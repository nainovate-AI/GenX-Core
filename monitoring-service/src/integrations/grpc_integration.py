# monitoring-service/src/integrations/grpc_integration.py
"""gRPC integration for monitoring service"""

import time
import logging
from typing import Callable, Any, Optional, Dict
from functools import wraps

import grpc
from grpc import aio
from opentelemetry import trace, context, propagate
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.grpc import (
    GrpcInstrumentorClient,
    GrpcInstrumentorServer,
    _OpenTelemetryClientInterceptor,
    _OpenTelemetryServicerContext
)
from opentelemetry.semconv.trace import SpanAttributes
from google.protobuf.json_format import MessageToDict

from core.opentelemetry_setup import get_tracer, get_meter, metrics_collector

logger = logging.getLogger(__name__)

class MonitoringClientInterceptor(grpc.UnaryUnaryClientInterceptor,
                                 grpc.UnaryStreamClientInterceptor,
                                 grpc.StreamUnaryClientInterceptor,
                                 grpc.StreamStreamClientInterceptor):
    """Client interceptor for monitoring gRPC calls"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = get_tracer(f"{service_name}.grpc.client")
        self.meter = get_meter(f"{service_name}.grpc.client")
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup gRPC client metrics"""
        self.call_counter = self.meter.create_counter(
            name="grpc_client_calls_total",
            description="Total number of gRPC client calls",
            unit="1"
        )
        
        self.call_duration = self.meter.create_histogram(
            name="grpc_client_call_duration",
            description="Duration of gRPC client calls",
            unit="s"
        )
        
        self.error_counter = self.meter.create_counter(
            name="grpc_client_errors_total",
            description="Total number of gRPC client errors",
            unit="1"
        )
    
    def _intercept_call(
        self,
        continuation: Callable,
        client_call_details: grpc.ClientCallDetails,
        request_or_iterator: Any
    ):
        """Common interception logic"""
        method = client_call_details.method
        
        # Start span
        with self.tracer.start_as_current_span(
            f"grpc.{method}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                SpanAttributes.RPC_SYSTEM: "grpc",
                SpanAttributes.RPC_SERVICE: self.service_name,
                SpanAttributes.RPC_METHOD: method,
                "grpc.request_type": type(request_or_iterator).__name__
            }
        ) as span:
            start_time = time.time()
            
            # Inject trace context into metadata
            metadata = []
            if client_call_details.metadata:
                metadata = list(client_call_details.metadata)
            
            # Propagate trace context
            propagate.inject(dict(metadata))
            
            # Update call details with new metadata
            client_call_details = client_call_details._replace(
                metadata=metadata
            )
            
            try:
                # Make the actual call
                response = continuation(client_call_details, request_or_iterator)
                
                # Record success metrics
                duration = time.time() - start_time
                self.call_counter.add(1, {
                    "method": method,
                    "status": "success"
                })
                self.call_duration.record(duration, {"method": method})
                
                span.set_status(Status(StatusCode.OK))
                return response
                
            except grpc.RpcError as e:
                # Record error metrics
                duration = time.time() - start_time
                self.call_counter.add(1, {
                    "method": method,
                    "status": "error"
                })
                self.error_counter.add(1, {
                    "method": method,
                    "code": str(e.code())
                })
                
                # Record exception in span
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, f"gRPC error: {e.code()}")
                )
                span.set_attribute("grpc.status_code", str(e.code()))
                
                raise
    
    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)
    
    def intercept_unary_stream(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)
    
    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)
    
    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return self._intercept_call(continuation, client_call_details, request_iterator)


class MonitoringServerInterceptor(grpc.ServerInterceptor):
    """Server interceptor for monitoring gRPC calls"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = get_tracer(f"{service_name}.grpc.server")
        self.meter = get_meter(f"{service_name}.grpc.server")
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup gRPC server metrics"""
        self.call_counter = self.meter.create_counter(
            name="grpc_server_calls_total",
            description="Total number of gRPC server calls",
            unit="1"
        )
        
        self.call_duration = self.meter.create_histogram(
            name="grpc_server_call_duration",
            description="Duration of gRPC server calls",
            unit="s"
        )
        
        self.active_calls = self.meter.create_up_down_counter(
            name="grpc_server_active_calls",
            description="Number of active gRPC server calls",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="grpc_server_errors_total",
            description="Total number of gRPC server errors",
            unit="1"
        )
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming gRPC calls"""
        method = handler_call_details.method
        
        def wrapper(request_or_iterator, servicer_context):
            # Extract trace context from metadata
            metadata = dict(servicer_context.invocation_metadata())
            ctx = propagate.extract(metadata)
            
            # Start span with extracted context
            with self.tracer.start_as_current_span(
                f"grpc.{method}",
                kind=trace.SpanKind.SERVER,
                context=ctx,
                attributes={
                    SpanAttributes.RPC_SYSTEM: "grpc",
                    SpanAttributes.RPC_SERVICE: self.service_name,
                    SpanAttributes.RPC_METHOD: method,
                    "grpc.peer": servicer_context.peer()
                }
            ) as span:
                start_time = time.time()
                self.active_calls.add(1, {"method": method})
                
                try:
                    # Call the actual handler
                    response = continuation(request_or_iterator, servicer_context)
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    self.call_counter.add(1, {
                        "method": method,
                        "status": "success"
                    })
                    self.call_duration.record(duration, {"method": method})
                    
                    span.set_status(Status(StatusCode.OK))
                    return response
                    
                except Exception as e:
                    # Record error metrics
                    duration = time.time() - start_time
                    self.call_counter.add(1, {
                        "method": method,
                        "status": "error"
                    })
                    self.error_counter.add(1, {
                        "method": method,
                        "exception": type(e).__name__
                    })
                    
                    # Record exception in span
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    
                    raise
                    
                finally:
                    self.active_calls.add(-1, {"method": method})
        
        return wrapper


class AsyncMonitoringServerInterceptor(aio.ServerInterceptor):
    """Async server interceptor for monitoring gRPC calls"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = get_tracer(f"{service_name}.grpc.server")
        self.meter = get_meter(f"{service_name}.grpc.server")
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup gRPC server metrics"""
        self.call_counter = self.meter.create_counter(
            name="grpc_async_server_calls_total",
            description="Total number of async gRPC server calls",
            unit="1"
        )
        
        self.call_duration = self.meter.create_histogram(
            name="grpc_async_server_call_duration",
            description="Duration of async gRPC server calls",
            unit="s"
        )
        
        self.active_calls = self.meter.create_up_down_counter(
            name="grpc_async_server_active_calls",
            description="Number of active async gRPC server calls",
            unit="1"
        )
    
    async def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming async gRPC calls"""
        method = handler_call_details.method
        
        async def wrapper(request_or_iterator, servicer_context):
            # Extract trace context
            metadata = dict(servicer_context.invocation_metadata())
            ctx = propagate.extract(metadata)
            
            with self.tracer.start_as_current_span(
                f"grpc.{method}",
                kind=trace.SpanKind.SERVER,
                context=ctx,
                attributes={
                    SpanAttributes.RPC_SYSTEM: "grpc",
                    SpanAttributes.RPC_SERVICE: self.service_name,
                    SpanAttributes.RPC_METHOD: method
                }
            ) as span:
                start_time = time.time()
                self.active_calls.add(1, {"method": method})
                
                try:
                    response = await continuation(request_or_iterator, servicer_context)
                    
                    duration = time.time() - start_time
                    self.call_counter.add(1, {
                        "method": method,
                        "status": "success"
                    })
                    self.call_duration.record(duration, {"method": method})
                    
                    span.set_status(Status(StatusCode.OK))
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    self.active_calls.add(-1, {"method": method})
        
        return wrapper


def create_monitored_channel(
    target: str,
    service_name: str,
    options: Optional[list] = None,
    credentials: Optional[grpc.ChannelCredentials] = None
) -> grpc.Channel:
    """Create a gRPC channel with monitoring interceptors"""
    interceptors = [MonitoringClientInterceptor(service_name)]
    
    if credentials:
        return grpc.secure_channel(
            target,
            credentials,
            options=options,
            interceptors=interceptors
        )
    else:
        return grpc.insecure_channel(
            target,
            options=options,
            interceptors=interceptors
        )


def create_async_monitored_channel(
    target: str,
    service_name: str,
    options: Optional[list] = None,
    credentials: Optional[grpc.ChannelCredentials] = None
) -> aio.Channel:
    """Create an async gRPC channel with monitoring interceptors"""
    # Note: aio channels don't support interceptors yet in the same way
    # This is a placeholder for when they do
    if credentials:
        return aio.secure_channel(target, credentials, options=options)
    else:
        return aio.insecure_channel(target, options=options)


def add_monitoring_interceptors(
    server: grpc.Server,
    service_name: str
) -> grpc.Server:
    """Add monitoring interceptors to a gRPC server"""
    interceptor = MonitoringServerInterceptor(service_name)
    server = grpc.intercept_server(server, interceptor)
    return server


def add_async_monitoring_interceptors(
    server: aio.Server,
    service_name: str
) -> aio.Server:
    """Add monitoring interceptors to an async gRPC server"""
    interceptor = AsyncMonitoringServerInterceptor(service_name)
    server = grpc.aio.Server(interceptors=[interceptor])
    return server


# Decorator for monitoring gRPC service methods
def monitor_grpc_method(method_name: Optional[str] = None):
    """Decorator to monitor individual gRPC methods"""
    def decorator(func):
        name = method_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(self, request, context):
            tracer = get_tracer(self.__class__.__name__)
            
            with tracer.start_as_current_span(
                f"grpc.method.{name}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                try:
                    # Add request details to span
                    if hasattr(request, "DESCRIPTOR"):
                        span.set_attribute(
                            "grpc.request",
                            str(MessageToDict(request, preserving_proto_field_name=True))
                        )
                    
                    result = await func(self, request, context)
                    
                    # Add response details to span
                    if hasattr(result, "DESCRIPTOR"):
                        span.set_attribute(
                            "grpc.response_size",
                            len(str(result))
                        )
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        
        @wraps(func)
        def sync_wrapper(self, request, context):
            tracer = get_tracer(self.__class__.__name__)
            
            with tracer.start_as_current_span(
                f"grpc.method.{name}",
                kind=trace.SpanKind.INTERNAL
            ) as span:
                try:
                    result = func(self, request, context)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Helper class for gRPC service integration
class GrpcMonitoringMixin:
    """Mixin class for gRPC services to add monitoring capabilities"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = get_tracer(service_name)
        self.meter = get_meter(service_name)
        self._setup_service_metrics()
    
    def _setup_service_metrics(self):
        """Setup service-specific metrics"""
        self.operation_counter = self.meter.create_counter(
            name=f"{self.service_name}_operations_total",
            description=f"Total operations for {self.service_name}",
            unit="1"
        )
        
        self.operation_duration = self.meter.create_histogram(
            name=f"{self.service_name}_operation_duration",
            description=f"Operation duration for {self.service_name}",
            unit="s"
        )
    
    def record_operation(self, operation: str, duration: float, success: bool):
        """Record operation metrics"""
        labels = {
            "operation": operation,
            "success": str(success)
        }
        self.operation_counter.add(1, labels)
        self.operation_duration.record(duration, labels)