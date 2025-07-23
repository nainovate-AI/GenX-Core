# services/proxy_service/src/api/websocket_router.py
"""
WebSocket Router for FastAPI Integration
"""
from fastapi import APIRouter, WebSocket, Query
from typing import Optional

from .websocket_gateway import handle_websocket_connection, ws_manager

# Create router
router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/metrics")
async def websocket_metrics_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT token for authentication")
):
    """
    WebSocket endpoint for real-time metrics streaming
    
    Connection URL: ws://localhost:8000/ws/metrics?token=<JWT_TOKEN>
    
    Or authenticate after connection with message:
    {
        "type": "authenticate",
        "token": "<JWT_TOKEN>"
    }
    
    Message Protocol:
    
    Client -> Server:
    - authenticate: {"type": "authenticate", "token": "..."}
    - subscribe: {"type": "subscribe", "metric_types": ["cpu", "memory"], "interval": 5}
    - unsubscribe: {"type": "unsubscribe", "subscription_id": "..."}
    - request_metrics: {"type": "request_metrics", "metric_types": ["all"]}
    - refresh: {"type": "refresh"}
    - ping: {"type": "ping"}
    
    Server -> Client:
    - ready: Initial connection ready message
    - authenticated: Authentication successful
    - metrics_update: Metrics data update
    - error: Error message
    - pong: Heartbeat response
    """
    await handle_websocket_connection(websocket)


# Add this to your main FastAPI app
def register_websocket_routes(app):
    """Register WebSocket routes with the FastAPI app"""
    app.include_router(router)
    
    # Initialize WebSocket manager on startup
    @app.on_event("startup")
    async def startup_websocket():
        await ws_manager.initialize()
    
    # Shutdown WebSocket manager
    @app.on_event("shutdown") 
    async def shutdown_websocket():
        await ws_manager.shutdown()