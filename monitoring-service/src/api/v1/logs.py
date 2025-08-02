# monitoring-service/src/api/v1/logs.py
"""Logs API endpoints for Loki integration"""

from fastapi import APIRouter, Query, HTTPException, WebSocket
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import json

from services.loki_service import LokiService
from models.logs import LogQuery, LogEntry, LogStreamResponse

router = APIRouter()
loki_service = LokiService()

# Request/Response models
class LogQueryRequest(BaseModel):
    """Log query request model"""
    query: str  # LogQL query
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: Optional[int] = 100
    direction: Optional[str] = "backward"  # forward or backward

class LogLabels(BaseModel):
    """Log labels model"""
    service: Optional[str] = None
    level: Optional[str] = None
    namespace: Optional[str] = None
    additional: Optional[Dict[str, str]] = None

@router.post("/query")
async def query_logs(request: LogQueryRequest):
    """
    Query logs from Loki using LogQL
    
    Example queries:
    - {service="monitoring-api"}
    - {service="monitoring-api"} |= "error"
    - {service=~"monitoring-.*"} | json | level="error"
    """
    try:
        # Default time range if not specified
        if not request.end:
            request.end = datetime.utcnow()
        if not request.start:
            request.start = request.end - timedelta(hours=1)
        
        result = await loki_service.query_range(
            query=request.query,
            start=request.start,
            end=request.end,
            limit=request.limit,
            direction=request.direction
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query logs: {str(e)}"
        )

@router.get("/query")
async def query_logs_simple(
    service: Optional[str] = Query(None, description="Service name to filter"),
    level: Optional[str] = Query(None, description="Log level to filter"),
    search: Optional[str] = Query(None, description="Text to search in logs"),
    start: Optional[str] = Query(None, description="Start time (ISO format)"),
    end: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, description="Maximum number of log entries")
):
    """Simple log query interface"""
    # Build LogQL query
    label_filters = []
    if service:
        label_filters.append(f'service="{service}"')
    if level:
        label_filters.append(f'level="{level}"')
    
    if not label_filters:
        label_filters.append('service=~".+"')  # Match any service
    
    query = "{" + ",".join(label_filters) + "}"
    
    # Add line filters
    if search:
        query += f' |= "{search}"'
    
    # Parse times
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(hours=1)
    
    try:
        result = await loki_service.query_range(
            query=query,
            start=start_time,
            end=end_time,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query logs: {str(e)}"
        )

@router.get("/labels")
async def get_labels():
    """Get all available log labels"""
    try:
        labels = await loki_service.get_labels()
        return labels
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get labels: {str(e)}"
        )

@router.get("/labels/{label}/values")
async def get_label_values(
    label: str,
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get all values for a specific label"""
    try:
        values = await loki_service.get_label_values(label, start, end)
        return values
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get label values: {str(e)}"
        )

@router.post("/push")
async def push_logs(entries: List[LogEntry]):
    """Push log entries to Loki"""
    try:
        result = await loki_service.push_logs(entries)
        return {"status": "success", "entries_pushed": len(entries)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to push logs: {str(e)}"
        )

@router.websocket("/stream")
async def stream_logs(websocket: WebSocket):
    """
    WebSocket endpoint for streaming logs in real-time
    
    Client sends:
    {
        "query": "{service=\"monitoring-api\"}",
        "start": "now"
    }
    """
    await websocket.accept()
    
    try:
        # Receive query parameters
        data = await websocket.receive_json()
        query = data.get("query", '{service=~".+"}')
        
        # Start streaming logs
        async for log_entry in loki_service.stream_logs(query):
            await websocket.send_json({
                "timestamp": log_entry.get("timestamp"),
                "line": log_entry.get("line"),
                "labels": log_entry.get("labels")
            })
            
    except Exception as e:
        await websocket.send_json({
            "error": str(e),
            "type": "error"
        })
    finally:
        await websocket.close()

@router.get("/tail")
async def tail_logs(
    service: Optional[str] = Query(None, description="Service to tail"),
    lines: int = Query(100, description="Number of recent lines"),
    follow: bool = Query(False, description="Follow log output")
):
    """Tail logs from a specific service"""
    query = f'{{service="{service}"}}' if service else '{service=~".+"}'
    
    try:
        if follow:
            # Return streaming response
            return {
                "message": "Use WebSocket endpoint /api/v1/logs/stream for live tailing",
                "query": query
            }
        else:
            # Return recent logs
            result = await loki_service.query_range(
                query=query,
                start=datetime.utcnow() - timedelta(minutes=5),
                end=datetime.utcnow(),
                limit=lines,
                direction="backward"
            )
            return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to tail logs: {str(e)}"
        )

@router.get("/services")
async def list_logging_services():
    """List all services that are sending logs"""
    try:
        services = await loki_service.get_label_values("service")
        return {
            "services": services.get("data", []),
            "count": len(services.get("data", []))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list services: {str(e)}"
        )

@router.get("/stats")
async def get_log_stats(
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get log statistics"""
    # Default to last 24 hours
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(days=1)
    
    try:
        # Get log volume by service
        volume_query = 'sum by (service) (rate({service=~".+"}[5m]))'
        volume_stats = await loki_service.query(volume_query)
        
        # Get error rate
        error_query = 'sum by (service) (rate({service=~".+"} |= "error"[5m]))'
        error_stats = await loki_service.query(error_query)
        
        return {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "volume_by_service": volume_stats,
            "error_rate_by_service": error_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get log stats: {str(e)}"
        )

@router.delete("/delete")
async def delete_logs(
    query: str = Query(..., description="LogQL query for logs to delete"),
    start: str = Query(..., description="Start time"),
    end: str = Query(..., description="End time"),
    dry_run: bool = Query(True, description="Only show what would be deleted")
):
    """Delete logs matching a query (admin only)"""
    # Note: Loki doesn't support deletion by default
    # This is a placeholder for custom implementation
    return {
        "message": "Log deletion not implemented",
        "note": "Loki does not support log deletion by design. Logs expire based on retention policies."
    }

@router.get("/export")
async def export_logs(
    query: str = Query(..., description="LogQL query"),
    start: str = Query(..., description="Start time"),
    end: str = Query(..., description="End time"),
    format: str = Query("json", description="Export format: json, csv")
):
    """Export logs matching a query"""
    try:
        result = await loki_service.query_range(
            query=query,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
            limit=5000  # Reasonable limit for export
        )
        
        if format == "csv":
            # Convert to CSV format
            return {
                "message": "CSV export not yet implemented",
                "data": result
            }
        else:
            return result
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export logs: {str(e)}"
        )