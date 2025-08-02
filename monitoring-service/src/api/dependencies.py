# monitoring-service/src/api/dependencies.py
"""API dependencies and common utilities"""

from fastapi import Depends, HTTPException, Header, Query
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache

from core.config import settings
from services import (
    prometheus_service,
    loki_service,
    jaeger_service,
    grafana_service,
    alertmanager_service,
    otel_collector_service
)

# Cache for service instances
@lru_cache()
def get_prometheus_service():
    """Get Prometheus service instance"""
    return prometheus_service

@lru_cache()
def get_loki_service():
    """Get Loki service instance"""
    return loki_service

@lru_cache()
def get_jaeger_service():
    """Get Jaeger service instance"""
    return jaeger_service

@lru_cache()
def get_grafana_service():
    """Get Grafana service instance"""
    return grafana_service

@lru_cache()
def get_alertmanager_service():
    """Get AlertManager service instance"""
    return alertmanager_service

@lru_cache()
def get_otel_collector_service():
    """Get OpenTelemetry Collector service instance"""
    return otel_collector_service

# Common query parameters
class TimeRangeParams:
    """Common time range parameters"""
    def __init__(
        self,
        start: Optional[str] = Query(None, description="Start time (ISO format)"),
        end: Optional[str] = Query(None, description="End time (ISO format)"),
        last: Optional[str] = Query(None, description="Last duration (e.g., 1h, 24h, 7d)")
    ):
        if last and (start or end):
            raise HTTPException(
                status_code=400,
                detail="Cannot specify 'last' with 'start' or 'end'"
            )
        
        if last:
            # Parse duration
            self.end_time = datetime.utcnow()
            self.start_time = self._parse_duration(last)
        else:
            self.end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
            self.start_time = datetime.fromisoformat(start) if start else self.end_time - timedelta(hours=1)
        
        if self.start_time >= self.end_time:
            raise HTTPException(
                status_code=400,
                detail="Start time must be before end time"
            )
    
    def _parse_duration(self, duration: str) -> datetime:
        """Parse duration string like 1h, 24h, 7d"""
        import re
        match = re.match(r'^(\d+)([hdwm])$', duration.lower())
        if not match:
            raise HTTPException(
                status_code=400,
                detail="Invalid duration format. Use format like: 1h, 24h, 7d, 1w, 1m"
            )
        
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'h':
            delta = timedelta(hours=value)
        elif unit == 'd':
            delta = timedelta(days=value)
        elif unit == 'w':
            delta = timedelta(weeks=value)
        elif unit == 'm':
            delta = timedelta(days=value * 30)  # Approximate
        
        return self.end_time - delta

class PaginationParams:
    """Common pagination parameters"""
    def __init__(
        self,
        limit: int = Query(100, ge=1, le=1000, description="Items per page"),
        offset: int = Query(0, ge=0, description="Number of items to skip")
    ):
        self.limit = limit
        self.offset = offset

# API key validation (placeholder for future implementation)
async def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for authentication")
) -> Optional[str]:
    """Verify API key if provided"""
    if not x_api_key:
        return None  # No authentication required for now
    
    # Placeholder for API key validation
    # In production, this would check against a database or external auth service
    if x_api_key == "invalid":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return x_api_key

# Rate limiting (simple in-memory implementation)
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old entries
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
        
        # Check rate limit
        if key not in self.requests:
            self.requests[key] = []
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit(
    x_api_key: Optional[str] = Depends(verify_api_key),
    x_forwarded_for: Optional[str] = Header(None)
) -> None:
    """Check rate limit for the request"""
    # Use API key or IP address as rate limit key
    key = x_api_key or x_forwarded_for or "anonymous"
    
    if not await rate_limiter.check_rate_limit(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

# Service health dependency
async def require_healthy_service(service_name: str) -> None:
    """Ensure a service is healthy before processing request"""
    service_map = {
        "prometheus": prometheus_service,
        "loki": loki_service,
        "jaeger": jaeger_service,
        "grafana": grafana_service,
        "alertmanager": alertmanager_service
    }
    
    if service_name not in service_map:
        return
    
    service = service_map[service_name]
    health = await service.health_check()
    
    if health.get("status") != "healthy":
        raise HTTPException(
            status_code=503,
            detail=f"{service_name} service is not healthy"
        )

# Request context for tracing
class RequestContext:
    """Request context for distributed tracing"""
    def __init__(
        self,
        x_request_id: Optional[str] = Header(None),
        x_trace_id: Optional[str] = Header(None),
        user_agent: Optional[str] = Header(None)
    ):
        self.request_id = x_request_id or self._generate_request_id()
        self.trace_id = x_trace_id
        self.user_agent = user_agent
    
    @staticmethod
    def _generate_request_id() -> str:
        """Generate a unique request ID"""
        import uuid
        return str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/tracing"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "user_agent": self.user_agent
        }

# Query validation
def validate_promql(query: str) -> str:
    """Validate PromQL query syntax"""
    # Basic validation - in production, use proper PromQL parser
    if not query or len(query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Invalid PromQL query"
        )
    
    # Check for common issues
    forbidden_keywords = ["drop", "delete", "truncate"]
    if any(keyword in query.lower() for keyword in forbidden_keywords):
        raise HTTPException(
            status_code=400,
            detail="Query contains forbidden keywords"
        )
    
    return query

def validate_logql(query: str) -> str:
    """Validate LogQL query syntax"""
    # Basic validation - in production, use proper LogQL parser
    if not query or len(query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Invalid LogQL query"
        )
    
    # Must start with stream selector
    if not query.startswith("{"):
        raise HTTPException(
            status_code=400,
            detail="LogQL query must start with a stream selector"
        )
    
    return query

# Async timeout decorator
def with_timeout(seconds: int = 30):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timeout after {seconds} seconds"
                )
        return wrapper
    return decorator

# Common response headers
def get_cache_headers(max_age: int = 60) -> Dict[str, str]:
    """Get cache control headers"""
    return {
        "Cache-Control": f"public, max-age={max_age}",
        "X-Content-Type-Options": "nosniff"
    }

# Service discovery helper
async def discover_services() -> Dict[str, str]:
    """Discover available monitoring services"""
    services = {}
    
    # Check each service
    service_checks = [
        ("prometheus", prometheus_service, settings.PROMETHEUS_URL),
        ("loki", loki_service, settings.LOKI_URL),
        ("jaeger", jaeger_service, settings.JAEGER_URL),
        ("grafana", grafana_service, settings.GRAFANA_URL),
        ("alertmanager", alertmanager_service, settings.ALERTMANAGER_URL)
    ]
    
    for name, service, url in service_checks:
        try:
            health = await service.health_check()
            services[name] = {
                "url": url,
                "status": health.get("status", "unknown")
            }
        except Exception:
            services[name] = {
                "url": url,
                "status": "unreachable"
            }
    
    return services