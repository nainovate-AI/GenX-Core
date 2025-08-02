# monitoring-service/src/api/v1/traces.py
"""Traces API endpoints for Jaeger integration"""

from fastapi import APIRouter, Query, HTTPException, Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from services.jaeger_service import JaegerService
from models.traces import Trace, Span, TraceSearchParams, ServiceDependencies

router = APIRouter()
jaeger_service = JaegerService()

# Request/Response models
class TraceQuery(BaseModel):
    """Trace query parameters"""
    service: str
    operation: Optional[str] = None
    tags: Optional[Dict[str, str]] = {}
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_duration: Optional[str] = None
    max_duration: Optional[str] = None
    limit: int = 20

class SpanFilter(BaseModel):
    """Span filter criteria"""
    service: Optional[str] = None
    operation: Optional[str] = None
    min_duration_us: Optional[int] = None
    tags: Optional[Dict[str, str]] = {}

@router.get("/search")
async def search_traces(
    service: str = Query(..., description="Service name"),
    operation: Optional[str] = Query(None, description="Operation name"),
    tags: Optional[str] = Query(None, description="Tags in format key1=value1,key2=value2"),
    start: Optional[str] = Query(None, description="Start time (ISO format)"),
    end: Optional[str] = Query(None, description="End time (ISO format)"),
    min_duration: Optional[str] = Query(None, description="Minimum duration (e.g., 100ms, 1s)"),
    max_duration: Optional[str] = Query(None, description="Maximum duration"),
    limit: int = Query(20, description="Maximum number of traces to return")
):
    """
    Search for traces based on criteria
    
    Example tags format: "http.status_code=500,http.method=GET"
    """
    # Parse tags if provided
    parsed_tags = {}
    if tags:
        for tag_pair in tags.split(','):
            if '=' in tag_pair:
                key, value = tag_pair.split('=', 1)
                parsed_tags[key.strip()] = value.strip()
    
    # Default time range if not specified
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(hours=1)
    
    try:
        traces = await jaeger_service.search_traces(
            service=service,
            operation=operation,
            tags=parsed_tags,
            start_time=start_time,
            end_time=end_time,
            min_duration=min_duration,
            max_duration=max_duration,
            limit=limit
        )
        return traces
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search traces: {str(e)}")

@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str = Path(..., description="Trace ID")):
    """Get a specific trace by ID"""
    try:
        trace = await jaeger_service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        return trace
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")

@router.get("/services")
async def list_services():
    """List all services that have reported traces"""
    try:
        services = await jaeger_service.get_services()
        return services
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list services: {str(e)}")

@router.get("/services/{service}/operations")
async def list_operations(service: str = Path(..., description="Service name")):
    """List all operations for a specific service"""
    try:
        operations = await jaeger_service.get_operations(service)
        return operations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list operations: {str(e)}")

@router.get("/dependencies")
async def get_dependencies(
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get service dependency graph"""
    # Default to last 24 hours
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(days=1)
    
    try:
        dependencies = await jaeger_service.get_dependencies(start_time, end_time)
        return dependencies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dependencies: {str(e)}")

@router.get("/trace/{trace_id}/spans")
async def get_trace_spans(
    trace_id: str = Path(..., description="Trace ID"),
    service: Optional[str] = Query(None, description="Filter by service"),
    operation: Optional[str] = Query(None, description="Filter by operation")
):
    """Get all spans for a specific trace"""
    try:
        trace = await jaeger_service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        # Extract and filter spans
        spans = trace.get("data", [{}])[0].get("spans", [])
        
        if service:
            spans = [s for s in spans if s.get("process", {}).get("serviceName") == service]
        if operation:
            spans = [s for s in spans if s.get("operationName") == operation]
        
        return {"trace_id": trace_id, "spans": spans, "count": len(spans)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get spans: {str(e)}")

@router.get("/trace/{trace_id}/timeline")
async def get_trace_timeline(trace_id: str = Path(..., description="Trace ID")):
    """Get trace timeline visualization data"""
    try:
        trace = await jaeger_service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        # Process trace data for timeline visualization
        timeline_data = await jaeger_service.build_trace_timeline(trace)
        return timeline_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build timeline: {str(e)}")

@router.get("/trace/{trace_id}/metrics")
async def get_trace_metrics(trace_id: str = Path(..., description="Trace ID")):
    """Get metrics for a specific trace"""
    try:
        trace = await jaeger_service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        # Calculate trace metrics
        metrics = await jaeger_service.calculate_trace_metrics(trace)
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")

@router.get("/services/{service}/metrics")
async def get_service_metrics(
    service: str = Path(..., description="Service name"),
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get aggregated metrics for a service"""
    # Default to last hour
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(hours=1)
    
    try:
        metrics = await jaeger_service.get_service_metrics(service, start_time, end_time)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service metrics: {str(e)}")

@router.get("/services/{service}/errors")
async def get_service_errors(
    service: str = Path(..., description="Service name"),
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time"),
    limit: int = Query(100, description="Maximum number of error traces")
):
    """Get error traces for a service"""
    # Default to last hour
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(hours=1)
    
    try:
        # Search for traces with error tags
        error_traces = await jaeger_service.search_traces(
            service=service,
            tags={"error": "true"},
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return error_traces
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error traces: {str(e)}")

@router.post("/compare")
async def compare_traces(trace_ids: List[str]):
    """Compare multiple traces"""
    if len(trace_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 trace IDs required for comparison")
    
    if len(trace_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 traces can be compared at once")
    
    try:
        comparison = await jaeger_service.compare_traces(trace_ids)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare traces: {str(e)}")

@router.get("/latency/percentiles")
async def get_latency_percentiles(
    service: str = Query(..., description="Service name"),
    operation: Optional[str] = Query(None, description="Operation name"),
    percentiles: List[float] = Query([0.5, 0.75, 0.95, 0.99], description="Percentiles to calculate"),
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get latency percentiles for a service/operation"""
    # Default to last hour
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(hours=1)
    
    try:
        latencies = await jaeger_service.get_latency_percentiles(
            service=service,
            operation=operation,
            percentiles=percentiles,
            start_time=start_time,
            end_time=end_time
        )
        return latencies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latency percentiles: {str(e)}")

@router.get("/flamegraph/{trace_id}")
async def get_trace_flamegraph(trace_id: str = Path(..., description="Trace ID")):
    """Get flamegraph data for a trace"""
    try:
        trace = await jaeger_service.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        flamegraph = await jaeger_service.build_flamegraph(trace)
        return flamegraph
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build flamegraph: {str(e)}")

@router.get("/service-map")
async def get_service_map(
    depth: int = Query(3, description="Depth of service dependencies to include"),
    start: Optional[str] = Query(None, description="Start time"),
    end: Optional[str] = Query(None, description="End time")
):
    """Get service dependency map"""
    # Default to last 24 hours
    end_time = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_time = datetime.fromisoformat(start) if start else end_time - timedelta(days=1)
    
    try:
        service_map = await jaeger_service.build_service_map(
            depth=depth,
            start_time=start_time,
            end_time=end_time
        )
        return service_map
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build service map: {str(e)}")

@router.get("/health")
async def jaeger_health():
    """Check Jaeger health status"""
    try:
        health = await jaeger_service.health_check()
        if health.get("status") == "healthy":
            return health
        else:
            raise HTTPException(status_code=503, detail=health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Jaeger unhealthy: {str(e)}")

@router.get("/trace-examples")
async def get_trace_examples():
    """Get example trace queries"""
    return {
        "examples": [
            {
                "description": "Find slow traces",
                "query": {
                    "service": "frontend",
                    "min_duration": "1s"
                }
            },
            {
                "description": "Find error traces",
                "query": {
                    "service": "backend",
                    "tags": {"error": "true"}
                }
            },
            {
                "description": "Find traces with specific HTTP status",
                "query": {
                    "service": "api-gateway",
                    "tags": {"http.status_code": "500"}
                }
            },
            {
                "description": "Find traces for specific user",
                "query": {
                    "service": "user-service",
                    "tags": {"user.id": "12345"}
                }
            }
        ]
    }