# services/proxy_service/src/api/websocket_gateway.py
"""
WebSocket Gateway for Real-time Metrics
Following Discord's Gateway architecture for scalability
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Set, Optional
from enum import Enum
import uuid
import grpc
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.exceptions import WebSocketException
import jwt
import structlog
from opentelemetry import trace
import prometheus_client

from ..generated import metrics_service_pb2
from ..generated import metrics_service_pb2_grpc
from ..core.config import get_settings
from ..core.security import SecurityManager
from ..core.telemetry import get_tracer
from .rest_gateway import grpc_client

settings = get_settings()
logger = structlog.get_logger()
tracer = get_tracer("proxy.websocket")
security_manager = SecurityManager()

# Prometheus metrics
ws_connections = prometheus_client.Gauge(
    'websocket_active_connections',
    'Number of active WebSocket connections'
)
ws_messages_sent = prometheus_client.Counter(
    'websocket_messages_sent_total',
    'Total WebSocket messages sent',
    ['message_type']
)
ws_messages_received = prometheus_client.Counter(
    'websocket_messages_received_total',
    'Total WebSocket messages received',
    ['message_type']
)


class WSMessageType(str, Enum):
    """WebSocket message types"""
    # Client -> Server
    AUTHENTICATE = "authenticate"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    REQUEST_METRICS = "request_metrics"
    REFRESH = "refresh"
    PING = "ping"
    
    # Server -> Client
    AUTHENTICATED = "authenticated"
    METRICS_UPDATE = "metrics_update"
    ERROR = "error"
    PONG = "pong"
    READY = "ready"
    RECONNECT = "reconnect"


class MetricSubscription:
    """Represents a metric subscription"""
    def __init__(
        self, 
        metric_types: list,
        interval: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ):
        self.metric_types = metric_types
        self.interval = max(1, min(interval, 60))  # 1-60 seconds
        self.filters = filters or {}
        self.last_update = 0


class WebSocketConnection:
    """Manages a single WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id: Optional[str] = None
        self.authenticated = False
        self.subscriptions: Dict[str, MetricSubscription] = {}
        self.last_heartbeat = time.time()
        self.message_queue = asyncio.Queue(maxsize=100)
        self._tasks: Set[asyncio.Task] = set()
        
    async def authenticate(self, token: str) -> bool:
        """Authenticate the WebSocket connection"""
        try:
            payload = security_manager.verify_token(token)
            self.user_id = payload.get("user_id")
            self.authenticated = True
            
            logger.info("WebSocket authenticated",
                       connection_id=self.connection_id,
                       user_id=self.user_id)
            
            return True
            
        except Exception as e:
            logger.error("WebSocket authentication failed",
                        connection_id=self.connection_id,
                        error=str(e))
            return False
    
    async def close(self, code: int = 1000, reason: str = "Normal closure"):
        """Close the connection gracefully"""
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close WebSocket
        try:
            await self.websocket.close(code=code, reason=reason)
        except:
            pass
        
        logger.info("WebSocket connection closed",
                   connection_id=self.connection_id,
                   user_id=self.user_id)


class WebSocketManager:
    """Manages all WebSocket connections"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the manager"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive())
    
    async def shutdown(self):
        """Shutdown the manager"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        
        # Close all connections
        tasks = []
        for conn in self.connections.values():
            tasks.append(conn.close(code=1001, reason="Server shutdown"))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def connect(self, websocket: WebSocket) -> WebSocketConnection:
        """Register a new connection"""
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id)
        
        self.connections[connection_id] = connection
        ws_connections.inc()
        
        logger.info("WebSocket connection established",
                   connection_id=connection_id,
                   total_connections=len(self.connections))
        
        return connection
    
    async def disconnect(self, connection: WebSocketConnection):
        """Remove a connection"""
        if connection.connection_id in self.connections:
            del self.connections[connection.connection_id]
            ws_connections.dec()
        
        await connection.close()
    
    async def _cleanup_inactive(self):
        """Cleanup inactive connections periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = time.time()
                inactive_connections = []
                
                for conn_id, conn in self.connections.items():
                    # Check for inactive connections (no heartbeat for 60 seconds)
                    if current_time - conn.last_heartbeat > 60:
                        inactive_connections.append(conn)
                
                # Remove inactive connections
                for conn in inactive_connections:
                    logger.warning("Removing inactive connection",
                                 connection_id=conn.connection_id)
                    await self.disconnect(conn)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))


# Global WebSocket manager
ws_manager = WebSocketManager()


async def handle_websocket_connection(websocket: WebSocket):
    """
    Handle a WebSocket connection with full lifecycle management
    """
    # Accept connection
    await websocket.accept()
    
    # Create connection object
    connection = await ws_manager.connect(websocket)
    
    # Create tasks for handling connection
    receive_task = asyncio.create_task(
        handle_receive(connection),
        name=f"receive-{connection.connection_id}"
    )
    send_task = asyncio.create_task(
        handle_send(connection),
        name=f"send-{connection.connection_id}"
    )
    
    connection._tasks.add(receive_task)
    connection._tasks.add(send_task)
    
    try:
        # Send initial ready message
        await send_message(connection, {
            "type": WSMessageType.READY,
            "connection_id": connection.connection_id,
            "heartbeat_interval": 30000,  # 30 seconds in ms
            "version": "1.0.0"
        })
        
        # Wait for tasks to complete
        await asyncio.gather(receive_task, send_task)
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", connection_id=connection.connection_id)
    except Exception as e:
        logger.error("WebSocket error", 
                    connection_id=connection.connection_id,
                    error=str(e))
    finally:
        await ws_manager.disconnect(connection)


async def handle_receive(connection: WebSocketConnection):
    """Handle incoming messages from client"""
    try:
        while True:
            # Receive message with timeout
            message = await asyncio.wait_for(
                connection.websocket.receive_json(),
                timeout=45.0  # 45 second timeout
            )
            
            ws_messages_received.labels(
                message_type=message.get("type", "unknown")
            ).inc()
            
            # Update heartbeat
            connection.last_heartbeat = time.time()
            
            # Handle message
            await handle_message(connection, message)
            
    except asyncio.TimeoutError:
        logger.warning("WebSocket receive timeout",
                      connection_id=connection.connection_id)
        await connection.close(code=1001, reason="Receive timeout")
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.error("Error handling receive",
                    connection_id=connection.connection_id,
                    error=str(e))
        await send_error(connection, "Internal error", str(e))


async def handle_send(connection: WebSocketConnection):
    """Handle sending messages to client"""
    try:
        while True:
            # Get message from queue
            message = await connection.message_queue.get()
            
            # Send message
            await connection.websocket.send_json(message)
            
            ws_messages_sent.labels(
                message_type=message.get("type", "unknown")
            ).inc()
            
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.error("Error handling send",
                    connection_id=connection.connection_id,
                    error=str(e))


async def handle_message(connection: WebSocketConnection, message: Dict[str, Any]):
    """Route and handle incoming messages"""
    msg_type = message.get("type")
    
    if msg_type == WSMessageType.AUTHENTICATE:
        await handle_authenticate(connection, message)
    
    elif msg_type == WSMessageType.SUBSCRIBE:
        if not connection.authenticated:
            await send_error(connection, "Not authenticated", 
                           "You must authenticate before subscribing")
            return
        await handle_subscribe(connection, message)
    
    elif msg_type == WSMessageType.UNSUBSCRIBE:
        if not connection.authenticated:
            await send_error(connection, "Not authenticated")
            return
        await handle_unsubscribe(connection, message)
    
    elif msg_type == WSMessageType.REQUEST_METRICS:
        if not connection.authenticated:
            await send_error(connection, "Not authenticated")
            return
        await handle_request_metrics(connection, message)
    
    elif msg_type == WSMessageType.REFRESH:
        if not connection.authenticated:
            await send_error(connection, "Not authenticated")
            return
        await handle_refresh(connection, message)
    
    elif msg_type == WSMessageType.PING:
        await send_message(connection, {
            "type": WSMessageType.PONG,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    else:
        await send_error(connection, "Unknown message type", f"Type: {msg_type}")


async def handle_authenticate(connection: WebSocketConnection, message: Dict[str, Any]):
    """Handle authentication message"""
    token = message.get("token")
    
    if not token:
        await send_error(connection, "Missing token", 
                        "Authentication token is required")
        return
    
    # Check if token is blacklisted
    if await security_manager.is_token_blacklisted(token):
        await send_error(connection, "Token revoked", 
                        "This token has been revoked")
        await connection.close(code=4001, reason="Token revoked")
        return
    
    # Authenticate
    if await connection.authenticate(token):
        await send_message(connection, {
            "type": WSMessageType.AUTHENTICATED,
            "user_id": connection.user_id,
            "permissions": ["metrics:read", "metrics:stream"]
        })
        
        # Log for audit
        await security_manager.log_access(
            user_id=connection.user_id,
            resource="websocket:metrics",
            action="connect",
            metadata={"connection_id": connection.connection_id}
        )
    else:
        await send_error(connection, "Authentication failed", 
                        "Invalid or expired token")
        await connection.close(code=4001, reason="Authentication failed")


async def handle_subscribe(connection: WebSocketConnection, message: Dict[str, Any]):
    """Handle metric subscription"""
    subscription_id = message.get("subscription_id", str(uuid.uuid4()))
    metric_types = message.get("metric_types", ["all"])
    interval = message.get("interval", 5)
    filters = message.get("filters", {})
    
    # Convert metric types to proto enum
    proto_types = []
    for mt in metric_types:
        if mt == "all":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_ALL)
        elif mt == "cpu":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_CPU)
        elif mt == "memory":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_MEMORY)
        elif mt == "gpu":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_GPU)
        elif mt == "disk":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_DISK)
        elif mt == "network":
            proto_types.append(metrics_service_pb2.METRIC_TYPE_NETWORK)
    
    if not proto_types:
        proto_types = [metrics_service_pb2.METRIC_TYPE_ALL]
    
    # Create subscription
    subscription = MetricSubscription(proto_types, interval, filters)
    connection.subscriptions[subscription_id] = subscription
    
    # Start streaming task
    task = asyncio.create_task(
        stream_metrics(connection, subscription_id, subscription),
        name=f"stream-{connection.connection_id}-{subscription_id}"
    )
    connection._tasks.add(task)
    
    # Send confirmation
    await send_message(connection, {
        "type": "subscribed",
        "subscription_id": subscription_id,
        "metric_types": metric_types,
        "interval": subscription.interval
    })
    
    logger.info("Metrics subscription created",
               connection_id=connection.connection_id,
               subscription_id=subscription_id,
               metric_types=metric_types)


async def handle_unsubscribe(connection: WebSocketConnection, message: Dict[str, Any]):
    """Handle unsubscribe request"""
    subscription_id = message.get("subscription_id")
    
    if subscription_id in connection.subscriptions:
        del connection.subscriptions[subscription_id]
        
        # Cancel streaming task
        for task in connection._tasks:
            if task.get_name() == f"stream-{connection.connection_id}-{subscription_id}":
                task.cancel()
                break
        
        await send_message(connection, {
            "type": "unsubscribed",
            "subscription_id": subscription_id
        })
        
        logger.info("Metrics subscription removed",
                   connection_id=connection.connection_id,
                   subscription_id=subscription_id)
    else:
        await send_error(connection, "Subscription not found",
                        f"No subscription with ID: {subscription_id}")


async def handle_request_metrics(connection: WebSocketConnection, message: Dict[str, Any]):
    """Handle one-time metrics request"""
    metric_types = message.get("metric_types", ["all"])
    force_refresh = message.get("force_refresh", False)
    
    try:
        # Convert to proto types
        proto_types = []
        for mt in metric_types:
            if mt == "all":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_ALL)
            elif mt == "cpu":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_CPU)
            elif mt == "memory":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_MEMORY)
            elif mt == "gpu":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_GPU)
            elif mt == "disk":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_DISK)
            elif mt == "network":
                proto_types.append(metrics_service_pb2.METRIC_TYPE_NETWORK)
        
        if not proto_types:
            proto_types = [metrics_service_pb2.METRIC_TYPE_ALL]
        
        # Get metrics via gRPC
        stub = await grpc_client.get_metrics_stub()
        grpc_request = metrics_service_pb2.GetSystemMetricsRequest(
            metric_types=proto_types,
            force_refresh=force_refresh,
            request_id=f"ws-{connection.connection_id}"
        )
        
        response = await stub.GetSystemMetrics(grpc_request)
        
        # Convert and send
        metrics_data = proto_to_dict(response.metrics)
        
        await send_message(connection, {
            "type": WSMessageType.METRICS_UPDATE,
            "request_id": message.get("request_id"),
            "metrics": metrics_data,
            "source": response.source,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error("Failed to get metrics",
                    connection_id=connection.connection_id,
                    error=str(e))
        await send_error(connection, "Failed to get metrics", str(e))


async def handle_refresh(connection: WebSocketConnection, message: Dict[str, Any]):
    """Force refresh all subscriptions"""
    for subscription_id, subscription in connection.subscriptions.items():
        subscription.last_update = 0  # Force immediate update
    
    await send_message(connection, {
        "type": "refresh_acknowledged",
        "subscription_count": len(connection.subscriptions)
    })


async def stream_metrics(
    connection: WebSocketConnection,
    subscription_id: str,
    subscription: MetricSubscription
):
    """Stream metrics for a subscription"""
    try:
        # Create gRPC request
        grpc_request = metrics_service_pb2.StreamSystemMetricsRequest(
            interval_seconds=subscription.interval,
            metric_types=subscription.metric_types,
            request_id=f"ws-{connection.connection_id}-{subscription_id}"
        )
        
        # Get gRPC stub and start streaming
        stub = await grpc_client.get_metrics_stub()
        stream = stub.StreamSystemMetrics(grpc_request)
        
        # Stream updates
        async for update in stream:
            # Check if subscription still exists
            if subscription_id not in connection.subscriptions:
                break
            
            # Convert proto to dict
            metrics_data = proto_to_dict(update.metrics)
            
            # Apply filters if any
            if subscription.filters:
                metrics_data = apply_filters(metrics_data, subscription.filters)
            
            # Send update
            await send_message(connection, {
                "type": WSMessageType.METRICS_UPDATE,
                "subscription_id": subscription_id,
                "update_type": update.type,
                "metrics": metrics_data,
                "timestamp": update.timestamp.ToDatetime().isoformat()
            })
            
            # Update last update time
            subscription.last_update = time.time()
            
    except asyncio.CancelledError:
        logger.info("Metrics stream cancelled",
                   connection_id=connection.connection_id,
                   subscription_id=subscription_id)
    except Exception as e:
        logger.error("Error in metrics stream",
                    connection_id=connection.connection_id,
                    subscription_id=subscription_id,
                    error=str(e))
        
        await send_error(connection, "Stream error", 
                        f"Failed to stream metrics: {str(e)}")


async def send_message(connection: WebSocketConnection, message: Dict[str, Any]):
    """Send a message to the client"""
    try:
        await connection.message_queue.put(message)
    except asyncio.QueueFull:
        logger.warning("Message queue full, dropping message",
                      connection_id=connection.connection_id,
                      message_type=message.get("type"))


async def send_error(connection: WebSocketConnection, error: str, details: str = ""):
    """Send error message to client"""
    await send_message(connection, {
        "type": WSMessageType.ERROR,
        "error": error,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    })


def proto_to_dict(proto_obj) -> Dict[str, Any]:
    """Convert protobuf metrics to dictionary"""
    result = {}
    
    # CPU metrics
    if hasattr(proto_obj, 'cpu') and proto_obj.HasField('cpu'):
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
    
    # Memory metrics
    if hasattr(proto_obj, 'memory') and proto_obj.HasField('memory'):
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
    
    # GPU metrics
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
    
    # Disk metrics
    if hasattr(proto_obj, 'disk') and proto_obj.HasField('disk'):
        result['disk'] = {
            'usage': {
                'total_bytes': proto_obj.disk.usage.total_bytes,
                'used_bytes': proto_obj.disk.usage.used_bytes,
                'free_bytes': proto_obj.disk.usage.free_bytes,
                'percent': proto_obj.disk.usage.percent
            },
            'io': {
                'read_count': proto_obj.disk.io.read_count,
                'write_count': proto_obj.disk.io.write_count,
                'read_bytes': proto_obj.disk.io.read_bytes,
                'write_bytes': proto_obj.disk.io.write_bytes
            },
            'partitions': [
                {
                    'device': p.device,
                    'mount_point': p.mount_point,
                    'filesystem_type': p.filesystem_type,
                    'total_bytes': p.total_bytes,
                    'used_bytes': p.used_bytes,
                    'free_bytes': p.free_bytes,
                    'percent': p.percent
                }
                for p in proto_obj.disk.partitions
            ],
            'model_storage': {
                'path': proto_obj.disk.model_storage.path,
                'total_bytes': proto_obj.disk.model_storage.total_bytes,
                'used_bytes': proto_obj.disk.model_storage.used_bytes,
                'free_bytes': proto_obj.disk.model_storage.free_bytes,
                'percent': proto_obj.disk.model_storage.percent
            }
        }
    
    # Network metrics
    if hasattr(proto_obj, 'network') and proto_obj.HasField('network'):
        result['network'] = {
            'io': {
                'bytes_sent': proto_obj.network.io.bytes_sent,
                'bytes_received': proto_obj.network.io.bytes_received,
                'packets_sent': proto_obj.network.io.packets_sent,
                'packets_received': proto_obj.network.io.packets_received,
                'errors_in': proto_obj.network.io.errors_in,
                'errors_out': proto_obj.network.io.errors_out,
                'dropped_in': proto_obj.network.io.dropped_in,
                'dropped_out': proto_obj.network.io.dropped_out
            },
            'connection_states': dict(proto_obj.network.connection_states)
        }
    
    return result


def apply_filters(metrics: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    """Apply filters to metrics data"""
    # Example filters:
    # - threshold filters (e.g., only show if CPU > 80%)
    # - selection filters (e.g., only specific GPU IDs)
    
    filtered = metrics.copy()
    
    # CPU threshold filter
    if 'cpu_threshold' in filters and 'cpu' in filtered:
        if filtered['cpu']['usage_percent'] < filters['cpu_threshold']:
            del filtered['cpu']
    
    # GPU selection filter
    if 'gpu_ids' in filters and 'gpu' in filtered:
        filtered['gpu'] = [
            gpu for gpu in filtered['gpu']
            if gpu['id'] in filters['gpu_ids']
        ]
    
    return filtered