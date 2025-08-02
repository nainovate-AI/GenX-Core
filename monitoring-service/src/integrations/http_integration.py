# monitoring-service/src/integrations/http_integration.py
"""HTTP integration for monitoring service"""

import time
import json
import logging
from typing import Optional, Dict, Any, Callable, List
from functools import wraps
import asyncio
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import httpx
from opentelemetry import trace, context, propagate
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.propagate import inject, extract
from opentelemetry.semconv.trace import SpanAttributes

from core.opentelemetry_setup import get_tracer, get_meter, metrics_collector
from core.config import settings

logger = logging.getLogger(__name__)

class MonitoringHTTPMiddleware(BaseHTTPMiddleware):
    """HTTP middleware for monitoring requests and responses"""
    
    def __init__(self, app: ASGIApp, service_name: str = None):
        super().__init__(app)
        self.service_name = service_name or settings.SERVICE_NAME
        self.tracer = get_tracer(f"{self.service_name}.http")
        self.meter = get_meter(f"{self.service_name}.http")
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Setup HTTP metrics"""
        self.request_counter = self.meter.create_counter(
            name="http_server_requests_total",
            description="Total number of HTTP requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="http_server_request_duration_seconds",
            description="HTTP request duration in seconds",
            unit="s"
        )
        
        self.request_size = self.meter.create_histogram(
            name="http_server_request_size_bytes",
            description="HTTP request size in bytes",
            unit="By"
        )
        
        self.response_size = self.meter.create_histogram(
            name="http_server_response_size_bytes",
            description="HTTP response size in bytes",
            unit="By"
        )
        
        self.active_requests = self.meter.create_up_down_counter(
            name="http_server_active_requests",
            description="Number of active HTTP requests",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="http_server_errors_total",
            description="Total number of HTTP errors",
            unit="1"
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process HTTP request with monitoring"""
        # Extract trace context from headers
        carrier = dict(request.headers)
        ctx = extract(carrier)
        
        # Generate request ID if not present
        request_id = request.headers.get("x-request-id", f"{time.time()}")
        
        # Start span
        with self.tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=ctx,
            kind=trace.SpanKind.SERVER,
            attributes={
                SpanAttributes.HTTP_METHOD: request.method,
                SpanAttributes.HTTP_URL: str(request.url),
                SpanAttributes.HTTP_TARGET: request.url.path,
                SpanAttributes.HTTP_HOST: request.url.hostname,
                SpanAttributes.HTTP_SCHEME: request.url.scheme,
                SpanAttributes.HTTP_USER_AGENT: request.headers.get("user-agent", ""),
                SpanAttributes.NET_PEER_IP: request.client.host if request.client else None,
                "http.request_id": request_id,
                "service.name": self.service_name
            }
        ) as span:
            # Record metrics
            start_time = time.time()
            self.active_requests.add(1)
            
            # Add request size
            content_length = request.headers.get("content-length", 0)
            if content_length:
                self.request_size.record(int(content_length), {
                    "method": request.method,
                    "endpoint": request.url.path
                })
                span.set_attribute(SpanAttributes.HTTP_REQUEST_CONTENT_LENGTH, int(content_length))
            
            try:
                # Call next middleware/endpoint
                response = await call_next(request)
                
                # Record response metrics
                duration = time.time() - start_time
                status_code = response.status_code
                
                # Set span attributes
                span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, status_code)
                span.set_attribute("http.response.duration", duration)
                
                # Set span status
                if 400 <= status_code < 600:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Record metrics
                labels = {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status_code": str(status_code)
                }
                
                self.request_counter.add(1, labels)
                self.request_duration.record(duration, labels)
                
                if 400 <= status_code < 600:
                    self.error_counter.add(1, {
                        "method": request.method,
                        "endpoint": request.url.path,
                        "status_code": str(status_code),
                        "error_type": "client_error" if status_code < 500 else "server_error"
                    })
                
                # Add response headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                
                # Inject trace context into response
                inject(dict(response.headers))
                
                return response
                
            except Exception as e:
                # Record error
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                self.error_counter.add(1, {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "error_type": "exception",
                    "exception": type(e).__name__
                })
                
                # Re-raise exception
                raise
                
            finally:
                self.active_requests.add(-1)


class MonitoringRoute(APIRoute):
    """Custom route class with built-in monitoring"""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def monitored_route_handler(request: Request) -> Response:
            # Add custom monitoring logic here if needed
            return await original_route_handler(request)
        
        return monitored_route_handler


class HTTPXMonitoringTransport(httpx.AsyncHTTPTransport):
    """Custom HTTPX transport with monitoring"""
    
    def __init__(self, *args, service_name: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_name = service_name or settings.SERVICE_NAME
        self.tracer = get_tracer(f"{self.service_name}.http.client")
        self.meter = get_meter(f"{self.service_name}.http.client")
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup HTTP client metrics"""
        self.request_counter = self.meter.create_counter(
            name="http_client_requests_total",
            description="Total number of HTTP client requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="http_client_request_duration_seconds",
            description="HTTP client request duration in seconds",
            unit="s"
        )
        
        self.error_counter = self.meter.create_counter(
            name="http_client_errors_total",
            description="Total number of HTTP client errors",
            unit="1"
        )
    
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle HTTP request with monitoring"""
        # Start span
        with self.tracer.start_as_current_span(
            f"{request.method} {request.url.host}{request.url.path}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.HTTP_METHOD: request.method,
                SpanAttributes.HTTP_URL: str(request.url),
                SpanAttributes.HTTP_HOST: request.url.host,
                SpanAttributes.HTTP_SCHEME: request.url.scheme,
                "service.name": self.service_name
            }
        ) as span:
            # Inject trace context
            inject(dict(request.headers))
            
            start_time = time.time()
            
            try:
                # Make request
                response = await super().handle_async_request(request)
                
                # Record metrics
                duration = time.time() - start_time
                status_code = response.status_code
                
                span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, status_code)
                
                if 400 <= status_code < 600:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                labels = {
                    "method": request.method,
                    "host": request.url.host,
                    "status_code": str(status_code)
                }
                
                self.request_counter.add(1, labels)
                self.request_duration.record(duration, labels)
                
                if 400 <= status_code < 600:
                    self.error_counter.add(1, labels)
                
                return response
                
            except Exception as e:
                # Record error
                duration = time.time() - start_time
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                self.error_counter.add(1, {
                    "method": request.method,
                    "host": request.url.host,
                    "error_type": type(e).__name__
                })
                
                raise


def create_monitored_http_client(
    service_name: Optional[str] = None,
    timeout: float = 30.0,
    **kwargs
) -> httpx.AsyncClient:
    """Create an HTTP client with monitoring"""
    transport = HTTPXMonitoringTransport(service_name=service_name)
    
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        **kwargs
    )


class MonitoringHTTPClient:
    """HTTP client with built-in monitoring and retry logic"""
    
    def __init__(
        self,
        service_name: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.service_name = service_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client = create_monitored_http_client(
            service_name=service_name,
            base_url=base_url,
            timeout=timeout
        )
        
        self.tracer = get_tracer(f"{service_name}.http.client")
        self.meter = get_meter(f"{service_name}.http.client")
        
        # Retry metrics
        self.retry_counter = self.meter.create_counter(
            name="http_client_retries_total",
            description="Total number of HTTP client retries",
            unit="1"
        )
    
    async def request(
        self,
        method: str,
        url: str,
        retry: bool = True,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries if retry else 1):
            try:
                response = await self.client.request(method, url, **kwargs)
                
                # Check if we should retry on this status code
                if retry and response.status_code in [502, 503, 504] and attempt < self.max_retries - 1:
                    self.retry_counter.add(1, {
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                        "status_code": str(response.status_code)
                    })
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                return response
                
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e
                
                if retry and attempt < self.max_retries - 1:
                    self.retry_counter.add(1, {
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__
                    })
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                raise
        
        # If we get here, all retries failed
        raise last_exception
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET request"""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST request"""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """PUT request"""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """DELETE request"""
        return await self.request("DELETE", url, **kwargs)
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Decorator for monitoring HTTP endpoints
def monitor_endpoint(
    operation_name: Optional[str] = None,
    track_request_body: bool = False,
    track_response_body: bool = False
):
    """Decorator to monitor FastAPI endpoints"""
    def decorator(func):
        name = operation_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request if present
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            tracer = get_tracer("endpoint")
            
            with tracer.start_as_current_span(
                f"endpoint.{name}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                try:
                    # Add endpoint metadata
                    span.set_attribute("endpoint.name", name)
                    span.set_attribute("endpoint.module", func.__module__)
                    
                    # Track request body if enabled
                    if track_request_body and request:
                        try:
                            body = await request.json()
                            span.set_attribute("request.body", json.dumps(body)[:1000])
                        except:
                            pass
                    
                    # Execute endpoint
                    result = await func(*args, **kwargs)
                    
                    # Track response if enabled
                    if track_response_body and isinstance(result, dict):
                        span.set_attribute("response.body", json.dumps(result)[:1000])
                    
                    return result
                    
                except HTTPException as e:
                    span.set_attribute("error.status_code", e.status_code)
                    span.set_attribute("error.detail", str(e.detail))
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {e.status_code}"))
                    raise
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        
        return wrapper
    return decorator


# Helper function to extract trace context from request
def get_trace_context(request: Request) -> Dict[str, str]:
    """Extract trace context from request headers"""
    return {
        "trace_id": request.headers.get("x-trace-id", ""),
        "span_id": request.headers.get("x-span-id", ""),
        "parent_span_id": request.headers.get("x-parent-span-id", ""),
        "request_id": request.headers.get("x-request-id", "")
    }


# Error handler with monitoring
async def monitored_error_handler(request: Request, exc: Exception):
    """Global error handler with monitoring"""
    tracer = get_tracer("error_handler")
    
    with tracer.start_as_current_span(
        "error_handler",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "error.type": type(exc).__name__,
            "error.message": str(exc),
            "request.path": request.url.path,
            "request.method": request.method
        }
    ) as span:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR))
        
        # Log error
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            exc_info=True,
            extra={
                "request_path": request.url.path,
                "request_method": request.method,
                "trace_id": span.get_span_context().trace_id
            }
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
                "request_id": request.headers.get("x-request-id", ""),
                "trace_id": format(span.get_span_context().trace_id, "032x")
            }
        )


# Context manager for HTTP operations
@asynccontextmanager
async def http_operation_context(
    operation: str,
    attributes: Optional[Dict[str, Any]] = None
):
    """Context manager for monitoring HTTP operations"""
    tracer = get_tracer("http.operation")
    
    with tracer.start_as_current_span(
        operation,
        kind=trace.SpanKind.INTERNAL,
        attributes=attributes or {}
    ) as span:
        start_time = time.time()
        
        try:
            yield span
            
            duration = time.time() - start_time
            span.set_attribute("operation.duration", duration)
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            duration = time.time() - start_time
            span.set_attribute("operation.duration", duration)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise