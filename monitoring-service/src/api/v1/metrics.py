# monitoring-service/src/api/v1/metrics.py
"""Metrics API endpoints for Prometheus integration"""

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

from services.prometheus_service import PrometheusService
from models.metrics import MetricQuery, MetricResponse, TargetsResponse, MetricMetadata

router = APIRouter()
prometheus_service = PrometheusService()

# Request/Response models
class MetricQueryRequest(BaseModel):
    """Metric query request model"""
    query: str
    time: Optional[datetime] = None
    timeout: Optional[str] = "30s"

class MetricRangeQueryRequest(BaseModel):
    """Metric range query request model"""
    query: str
    start: datetime
    end: datetime
    step: str = "15s"
    timeout: Optional[str] = "30s"

class RecordingRule(BaseModel):
    """Recording rule model"""
    name: str
    query: str
    labels: Optional[Dict[str, str]] = {}

@router.get("/query", response_model=MetricResponse)
async def query_metrics(
    query: str = Query(..., description="PromQL query"),
    time: Optional[str] = Query(None, description="Evaluation timestamp (RFC3339 or Unix timestamp)")
):
    """
    Execute a Prometheus instant query
    
    Example queries:
    - up
    - http_requests_total{job="api-server"}
    - rate(http_requests_total[5m])
    - histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
    """
    try:
        result = await prometheus_service.query(query, time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/query_range", response_model=MetricResponse)
async def query_metrics_range(
    query: str = Query(..., description="PromQL query"),
    start: Optional[str] = Query(None, description="Start timestamp (RFC3339 or Unix)"),
    end: Optional[str] = Query(None, description="End timestamp (RFC3339 or Unix)"),
    step: str = Query("15s", description="Query resolution step duration")
):
    """
    Execute a Prometheus range query
    
    Returns time series data over a time range
    """
    # Default to last hour if not specified
    if not end:
        end = datetime.utcnow().isoformat() + "Z"
    if not start:
        start = (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
    
    try:
        result = await prometheus_service.query_range(query, start, end, step)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Range query failed: {str(e)}")

@router.get("/targets", response_model=TargetsResponse)
async def get_targets(
    state: Optional[str] = Query(None, description="Filter by target state: active, dropped, any")
):
    """Get all configured Prometheus targets and their status"""
    try:
        result = await prometheus_service.get_targets()
        
        # Filter by state if requested
        if state and state != "any":
            filtered_targets = []
            for target_group in result.get("data", {}).get("activeTargets", []):
                if target_group.get("health") == state:
                    filtered_targets.append(target_group)
            result["data"]["activeTargets"] = filtered_targets
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get targets: {str(e)}")

@router.get("/metadata")
async def get_metadata(
    metric: Optional[str] = Query(None, description="Filter by metric name"),
    limit: int = Query(100, description="Maximum number of metrics to return")
):
    """Get metric metadata"""
    try:
        result = await prometheus_service.get_metadata(metric)
        
        # Apply limit
        if "data" in result and isinstance(result["data"], list):
            result["data"] = result["data"][:limit]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")

@router.get("/series")
async def get_series(
    match: List[str] = Query(..., description="Series selector(s)"),
    start: Optional[str] = Query(None, description="Start timestamp"),
    end: Optional[str] = Query(None, description="End timestamp")
):
    """Find series by label matchers"""
    try:
        result = await prometheus_service.get_series(match, start, end)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get series: {str(e)}")

@router.get("/labels")
async def get_label_names(
    start: Optional[str] = Query(None, description="Start timestamp"),
    end: Optional[str] = Query(None, description="End timestamp")
):
    """Get a list of label names"""
    try:
        result = await prometheus_service.get_labels(start, end)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get labels: {str(e)}")

@router.get("/label/{label_name}/values")
async def get_label_values(
    label_name: str,
    start: Optional[str] = Query(None, description="Start timestamp"),
    end: Optional[str] = Query(None, description="End timestamp")
):
    """Get all values for a label"""
    try:
        result = await prometheus_service.get_label_values(label_name, start, end)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get label values: {str(e)}")

@router.post("/reload")
async def reload_prometheus_config():
    """Reload Prometheus configuration"""
    success = await prometheus_service.reload_config()
    if success:
        return {"message": "Configuration reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload configuration")

@router.get("/rules")
async def get_rules(
    type: Optional[str] = Query(None, description="Filter by rule type: alert, record")
):
    """Get all configured alerting and recording rules"""
    try:
        result = await prometheus_service.get_rules()
        
        # Filter by type if requested
        if type and result.get("data", {}).get("groups"):
            for group in result["data"]["groups"]:
                if type == "alert":
                    group["rules"] = [r for r in group.get("rules", []) if r.get("type") == "alerting"]
                elif type == "record":
                    group["rules"] = [r for r in group.get("rules", []) if r.get("type") == "recording"]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")

@router.get("/alerts")
async def get_alerts(
    state: Optional[str] = Query(None, description="Filter by alert state: pending, firing")
):
    """Get all active alerts"""
    try:
        result = await prometheus_service.get_alerts()
        
        # Filter by state if requested
        if state and result.get("data", {}).get("alerts"):
            filtered_alerts = [
                alert for alert in result["data"]["alerts"]
                if alert.get("state") == state
            ]
            result["data"]["alerts"] = filtered_alerts
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.get("/tsdb-status")
async def get_tsdb_status():
    """Get TSDB (Time Series Database) statistics"""
    try:
        result = await prometheus_service.get_tsdb_status()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get TSDB status: {str(e)}")

@router.post("/recording-rule")
async def create_recording_rule(rule: RecordingRule):
    """Create a new recording rule (requires config reload)"""
    # This is a placeholder - actual implementation would write to rule files
    return {
        "message": "Recording rule creation not implemented",
        "note": "This would require writing to Prometheus rule files and reloading config"
    }

@router.get("/top-metrics")
async def get_top_metrics(
    query: str = Query("topk(10, count by (__name__)({__name__=~'.+'}))", description="Query for top metrics"),
    time: Optional[str] = Query(None, description="Evaluation time")
):
    """Get top metrics by cardinality or other criteria"""
    try:
        result = await prometheus_service.query(query, time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get top metrics: {str(e)}")

@router.get("/query-exemplars")
async def query_exemplars(
    query: str = Query(..., description="PromQL query"),
    start: str = Query(..., description="Start timestamp"),
    end: str = Query(..., description="End timestamp")
):
    """Query exemplars (trace IDs) for metrics"""
    try:
        result = await prometheus_service.query_exemplars(query, start, end)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query exemplars: {str(e)}")

@router.get("/snapshot")
async def create_snapshot(skip_head: bool = Query(False, description="Skip head block")):
    """Create a snapshot of the TSDB data"""
    try:
        result = await prometheus_service.create_snapshot(skip_head)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")

@router.delete("/series")
async def delete_series(
    match: List[str] = Query(..., description="Series selector(s)"),
    start: str = Query(..., description="Start timestamp"),
    end: str = Query(..., description="End timestamp")
):
    """Delete time series data (admin only)"""
    try:
        result = await prometheus_service.delete_series(match, start, end)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete series: {str(e)}")

@router.get("/health")
async def prometheus_health():
    """Check Prometheus health status"""
    try:
        health = await prometheus_service.health_check()
        if health.get("status") == "healthy":
            return health
        else:
            raise HTTPException(status_code=503, detail=health)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prometheus unhealthy: {str(e)}")

@router.get("/common-queries")
async def get_common_queries():
    """Get a list of common/useful PromQL queries"""
    return {
        "system": {
            "cpu_usage": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            "memory_usage": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
            "disk_usage": '100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)',
            "network_traffic": 'rate(node_network_receive_bytes_total[5m])'
        },
        "services": {
            "up_services": 'up',
            "request_rate": 'sum(rate(http_requests_total[5m])) by (service)',
            "error_rate": 'sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)',
            "response_time_p95": 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
        },
        "monitoring": {
            "prometheus_memory": 'process_resident_memory_bytes{job="prometheus"}',
            "prometheus_samples": 'prometheus_tsdb_symbol_table_size_bytes',
            "scrape_duration": 'prometheus_target_scrape_duration_seconds'
        }
    }