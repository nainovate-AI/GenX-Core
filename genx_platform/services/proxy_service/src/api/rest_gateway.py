# services/proxy_service/src/api/rest_gateway.py
"""
Secure REST Gateway for gRPC Services
Following OWASP security guidelines and enterprise best practices
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import grpc
import jwt
import structlog
from opentelemetry import trace
import redis.asyncio as redis
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..generated import metrics_service_pb2
from ..generated import metrics_service_pb2_grpc
from ..core.config import get_settings
from ..core.security import SecurityManager
from ..core.telemetry import get_tracer

settings = get_settings()
logger = structlog.get_logger()
tracer = get_tracer("proxy.rest_gateway")

# Security setup
security = HTTPBearer()
security_manager = SecurityManager()

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# API Router
router = APIRouter(prefix="/api/v1", tags=["metrics"])


class GrpcClientManager:
    """Manages gRPC client connections with connection pooling"""
    
    def __init__(self):
        self._channels = {}
        self._stubs = {}
        
    async def get_metrics_stub(self) -> metrics_service_pb2_grpc.MetricsServiceStub:
        """Get or create metrics service stub"""
        service_name = "metrics_service"
        
        if service_name not in self._channels:
            # Create secure channel with TLS in production
            if settings.ENVIRONMENT == "production":
                credentials = grpc.ssl_channel_credentials()
                self._channels[service_name] = grpc.aio.secure_channel(
                    settings.METRICS_SERVICE_URL,
                    credentials,
                    options=[
                        ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # 10MB
                        ('grpc.keepalive_time_ms', 10000),
                        ('grpc.keepalive_timeout_ms', 5000),
                        ('grpc.keepalive_permit_without_calls', True),
                        ('grpc.http2.max_pings_without_data', 0),
                    ]
                )
            else:
                self._channels[service_name] = grpc.aio.insecure_channel(
                    settings.METRICS_SERVICE_URL,
                    options=[
                        ('grpc.max_receive_message_length', 10 * 1024 * 1024),
                    ]
                )
            
            self._stubs[service_name] = metrics_service_pb2_grpc.MetricsServiceStub(
                self._channels[service_name]
            )
        
        return self._stubs[service_name]
    
    async def close(self):
        """Close all channels"""
        for channel in self._channels.values():
            await channel.close()


# Global gRPC client manager
grpc_client = GrpcClientManager()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token with additional security checks
    """
    try:
        # Decode and verify token
        payload = security_manager.verify_token(credentials.credentials)
        
        # Additional checks
        if not payload.get("user_id"):
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        # Check if token is blacklisted (e.g., user logged out)
        if await security_manager.is_token_blacklisted(credentials.credentials):
            raise HTTPException(status_code=401, detail="Token has been revoked")
        
        # Check user permissions
        if not await security_manager.has_permission(payload["user_id"], "metrics:read"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.get("/metrics/system")
@limiter.limit("30/minute")  # Rate limiting
@tracer.start_as_current_span("get_system_metrics")
async def get_system_metrics(
    request: Request,
    metric_types: Optional[str] = None,
    force_refresh: bool = False,
    user: Dict[str, Any] = Depends(verify_token),
    x_request_id: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None)
):
    """
    Get current system metrics
    
    Security measures:
    - JWT authentication required
    - Rate limiting (30 requests/minute)
    - Request ID tracking
    - IP logging for audit trail
    """
    span = trace.get_current_span()
    span.set_attributes({
        "user.id": user["user_id"],
        "request.id": x_request_id,
        "client.ip": x_forwarded_for or request.client.host
    })
    
    try:
        # Parse metric types
        requested_types = []
        if metric_types:
            for mt in metric_types.split(","):
                if mt == "cpu":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_CPU)
                elif mt == "memory":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_MEMORY)
                elif mt == "gpu":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_GPU)
                elif mt == "disk":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_DISK)
                elif mt == "network":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_NETWORK)
        
        if not requested_types:
            requested_types = [metrics_service_pb2.METRIC_TYPE_ALL]
        
        # Create gRPC request
        grpc_request = metrics_service_pb2.GetSystemMetricsRequest(
            metric_types=requested_types,
            force_refresh=force_refresh,
            request_id=x_request_id or ""
        )
        
        # Call gRPC service
        stub = await grpc_client.get_metrics_stub()
        response = await stub.GetSystemMetrics(grpc_request)
        
        # Convert proto to JSON-friendly format
        metrics_data = _proto_to_dict(response.metrics)
        
        # Log access for audit trail
        await security_manager.log_access(
            user_id=user["user_id"],
            resource="metrics:system",
            action="read",
            ip_address=x_forwarded_for or request.client.host,
            request_id=x_request_id
        )
        
        return {
            "status": "success",
            "data": metrics_data,
            "source": response.source,
            "timestamp": response.timestamp.ToDatetime().isoformat()
        }
        
    except grpc.RpcError as e:
        span.record_exception(e)
        logger.error("gRPC error", error=str(e), code=e.code())
        raise HTTPException(status_code=500, detail="Service unavailable")
    except Exception as e:
        span.record_exception(e)
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics/system/stream")
async def stream_system_metrics(
    request: Request,
    interval: int = 5,
    metric_types: Optional[str] = None,
    user: Dict[str, Any] = Depends(verify_token),
    x_request_id: Optional[str] = Header(None)
):
    """
    Stream system metrics using Server-Sent Events (SSE)
    More secure than WebSocket for read-only data
    """
    # Validate interval
    if interval < 1 or interval > 60:
        raise HTTPException(status_code=400, detail="Interval must be between 1 and 60 seconds")
    
    # Parse metric types
    requested_types = []
    if metric_types:
        for mt in metric_types.split(","):
            if mt == "cpu":
                requested_types.append(metrics_service_pb2.METRIC_TYPE_CPU)
            elif mt == "memory":
                requested_types.append(metrics_service_pb2.METRIC_TYPE_MEMORY)
            elif mt == "gpu":
                requested_types.append(metrics_service_pb2.METRIC_TYPE_GPU)
            elif mt == "disk":
                requested_types.append(metrics_service_pb2.METRIC_TYPE_DISK)
            elif mt == "network":
                requested_types.append(metrics_service_pb2.METRIC_TYPE_NETWORK)
    
    if not requested_types:
        requested_types = [metrics_service_pb2.METRIC_TYPE_ALL]
    
    async def event_generator():
        """Generate SSE events from gRPC stream"""
        try:
            # Create gRPC request
            grpc_request = metrics_service_pb2.StreamSystemMetricsRequest(
                interval_seconds=interval,
                metric_types=requested_types,
                request_id=x_request_id or ""
            )
            
            # Get gRPC stub and start streaming
            stub = await grpc_client.get_metrics_stub()
            stream = stub.StreamSystemMetrics(grpc_request)
            
            # Send events
            async for update in stream:
                metrics_data = _proto_to_dict(update.metrics)
                
                event_data = {
                    "type": update.type,
                    "data": metrics_data,
                    "timestamp": update.timestamp.ToDatetime().isoformat()
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'type': 'close', 'reason': 'client_disconnect'})}\n\n"
        except Exception as e:
            logger.error("Stream error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Access-Control-Allow-Origin": settings.CORS_ORIGINS[0],  # Restrict CORS
        }
    )


@router.get("/metrics/system/history")
@limiter.limit("10/minute")  # Stricter rate limit for expensive operation
@tracer.start_as_current_span("get_metrics_history")
async def get_metrics_history(
    request: Request,
    start_time: datetime,
    end_time: datetime,
    metric_types: Optional[str] = None,
    resolution: int = 60,
    user: Dict[str, Any] = Depends(verify_token),
    x_request_id: Optional[str] = Header(None)
):
    """
    Get historical metrics with validation
    """
    # Validate time range
    if end_time < start_time:
        raise HTTPException(status_code=400, detail="End time must be after start time")
    
    time_diff = end_time - start_time
    if time_diff.days > 7:
        raise HTTPException(status_code=400, detail="Time range cannot exceed 7 days")
    
    if resolution < 60 or resolution > 3600:
        raise HTTPException(status_code=400, detail="Resolution must be between 60 and 3600 seconds")
    
    try:
        # Parse metric types
        requested_types = []
        if metric_types:
            for mt in metric_types.split(","):
                if mt == "cpu":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_CPU)
                elif mt == "memory":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_MEMORY)
                elif mt == "gpu":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_GPU)
                elif mt == "disk":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_DISK)
                elif mt == "network":
                    requested_types.append(metrics_service_pb2.METRIC_TYPE_NETWORK)
        
        if not requested_types:
            requested_types = [metrics_service_pb2.METRIC_TYPE_ALL]
        
        # Create gRPC request
        grpc_request = metrics_service_pb2.GetMetricsHistoryRequest(
            metric_types=requested_types,
            resolution_seconds=resolution,
            request_id=x_request_id or ""
        )
        grpc_request.start_time.FromDatetime(start_time)
        grpc_request.end_time.FromDatetime(end_time)
        
        # Call gRPC service
        stub = await grpc_client.get_metrics_stub()
        response = await stub.GetMetricsHistory(grpc_request)
        
        # Convert to JSON format
        data_points = []
        for point in response.data_points:
            data_points.append({
                "timestamp": point.timestamp.ToDatetime().isoformat(),
                "metrics": _proto_to_dict(point.metrics)
            })
        
        # Log access
        await security_manager.log_access(
            user_id=user["user_id"],
            resource="metrics:history",
            action="read",
            metadata={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "data_points": len(data_points)
            }
        )
        
        return {
            "status": "success",
            "data": data_points,
            "metadata": {
                "start_time": response.metadata.start_time.ToDatetime().isoformat(),
                "end_time": response.metadata.end_time.ToDatetime().isoformat(),
                "count": response.metadata.count,
                "resolution_seconds": response.metadata.resolution_seconds
            }
        }
        
    except grpc.RpcError as e:
        logger.error("gRPC error", error=str(e))
        raise HTTPException(status_code=500, detail="Service unavailable")
    except Exception as e:
        logger.error("Failed to get history", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


def _proto_to_dict(proto_obj) -> Dict[str, Any]:
    """Convert protobuf object to dictionary"""
    # This is a simplified version - you'd want to use protobuf's built-in JSON conversion
    result = {}
    
    # Convert based on the specific proto message type
    if hasattr(proto_obj, 'cpu'):
        result['cpu'] = {
            'usage_percent': proto_obj.cpu.usage_percent,
            'per_core_percent': list(proto_obj.cpu.per_core_percent),
            'frequency': {
                'current': proto_obj.cpu.frequency.current_mhz,
                'min': proto_obj.cpu.frequency.min_mhz,
                'max': proto_obj.cpu.frequency.max_mhz
            },
            'load_average': {
                '1min': proto_obj.cpu.load_average.one_minute,
                '5min': proto_obj.cpu.load_average.five_minutes,
                '15min': proto_obj.cpu.load_average.fifteen_minutes
            },
            'count': proto_obj.cpu.count,
            'count_logical': proto_obj.cpu.count_logical
        }
    
    if hasattr(proto_obj, 'memory'):
        result['memory'] = {
            'total_bytes': proto_obj.memory.total_bytes,
            'available_bytes': proto_obj.memory.available_bytes,
            'used_bytes': proto_obj.memory.used_bytes,
            'free_bytes': proto_obj.memory.free_bytes,
            'percent': proto_obj.memory.percent,
            'swap': {
                'total_bytes': proto_obj.memory.swap.total_bytes,
                'used_bytes': proto_obj.memory.swap.used_bytes,
                'free_bytes': proto_obj.memory.swap.free_bytes,
                'percent': proto_obj.memory.swap.percent
            }
        }
    
    if hasattr(proto_obj, 'gpu'):
        result['gpu'] = []
        for gpu in proto_obj.gpu:
            result['gpu'].append({
                'id': gpu.id,
                'name': gpu.name,
                'load_percent': gpu.load_percent,
                'memory': {
                    'total_bytes': gpu.memory.total_bytes,
                    'used_bytes': gpu.memory.used_bytes,
                    'free_bytes': gpu.memory.free_bytes,
                    'percent': gpu.memory.percent
                },
                'temperature_celsius': gpu.temperature_celsius,
                'uuid': gpu.uuid,
                'driver_version': gpu.driver_version
            })
    
    # Add disk and network similarly...
    
    return result


# Health check endpoint (no auth required)
@router.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}