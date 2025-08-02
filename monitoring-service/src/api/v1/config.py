# monitoring-service/src/api/v1/config.py
"""Configuration API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Optional, List
from pydantic import BaseModel

from core.config import settings
from services import (
    prometheus_service,
    loki_service,
    jaeger_service,
    grafana_service,
    alertmanager_service,
    otel_collector_service
)

router = APIRouter()

# Models for configuration updates
class ServiceConfig(BaseModel):
    """Generic service configuration model"""
    service: str
    config: Dict
    
class ReloadResponse(BaseModel):
    """Response for configuration reload"""
    service: str
    status: str
    message: Optional[str] = None

@router.get("/services")
async def list_services():
    """List all available monitoring services and their status"""
    services = {
        "prometheus": {
            "url": settings.PROMETHEUS_URL,
            "description": "Time series metrics storage",
            "endpoints": {
                "ui": "http://localhost:9090",
                "api": "http://localhost:9090/api/v1"
            }
        },
        "loki": {
            "url": settings.LOKI_URL,
            "description": "Log aggregation system",
            "endpoints": {
                "api": "http://localhost:3100"
            }
        },
        "jaeger": {
            "url": settings.JAEGER_URL,
            "description": "Distributed tracing",
            "endpoints": {
                "ui": "http://localhost:16686",
                "api": "http://localhost:16686/api"
            }
        },
        "grafana": {
            "url": settings.GRAFANA_URL,
            "description": "Metrics visualization",
            "endpoints": {
                "ui": "http://localhost:3000"
            }
        },
        "alertmanager": {
            "url": settings.ALERTMANAGER_URL,
            "description": "Alert routing and management",
            "endpoints": {
                "ui": "http://localhost:9093",
                "api": "http://localhost:9093/api/v1"
            }
        },
        "otel-collector": {
            "url": settings.OTEL_COLLECTOR_URL,
            "description": "OpenTelemetry collector",
            "endpoints": {
                "grpc": "localhost:4317",
                "http": "localhost:4318",
                "metrics": "http://localhost:8888/metrics"
            }
        }
    }
    return services

@router.get("/environment")
async def get_environment():
    """Get current environment configuration"""
    return {
        "environment": settings.ENVIRONMENT,
        "service_name": settings.SERVICE_NAME,
        "data_retention_days": settings.DATA_RETENTION_DAYS,
        "docker_network": settings.DOCKER_NETWORK,
        "debug": settings.DEBUG
    }

@router.post("/reload/{service}")
async def reload_service_config(service: str):
    """Reload configuration for a specific service"""
    reload_functions = {
        "prometheus": prometheus_service.reload_config,
        "alertmanager": alertmanager_service.reload_config,
        # Add more services as they support config reload
    }
    
    if service not in reload_functions:
        raise HTTPException(
            status_code=400, 
            detail=f"Service '{service}' does not support configuration reload"
        )
    
    try:
        success = await reload_functions[service]()
        if success:
            return ReloadResponse(
                service=service,
                status="success",
                message=f"Configuration reloaded for {service}"
            )
        else:
            return ReloadResponse(
                service=service,
                status="failed",
                message=f"Failed to reload configuration for {service}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading {service} configuration: {str(e)}"
        )

@router.get("/prometheus/targets")
async def get_prometheus_targets():
    """Get all Prometheus scrape targets"""
    try:
        targets = await prometheus_service.get_targets()
        return targets
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Prometheus targets: {str(e)}"
        )

@router.get("/prometheus/rules")
async def get_prometheus_rules():
    """Get all Prometheus alerting rules"""
    try:
        rules = await prometheus_service.get_rules()
        return rules
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Prometheus rules: {str(e)}"
        )

@router.get("/grafana/dashboards")
async def list_grafana_dashboards():
    """List all Grafana dashboards"""
    try:
        dashboards = await grafana_service.list_dashboards()
        return dashboards
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list Grafana dashboards: {str(e)}"
        )

@router.post("/grafana/dashboards/import")
async def import_grafana_dashboard(dashboard: Dict):
    """Import a new Grafana dashboard"""
    try:
        result = await grafana_service.import_dashboard(dashboard)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import dashboard: {str(e)}"
        )

@router.get("/alertmanager/alerts")
async def get_active_alerts():
    """Get all active alerts from AlertManager"""
    try:
        alerts = await alertmanager_service.get_alerts()
        return alerts
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alerts: {str(e)}"
        )

@router.post("/test-connectivity")
async def test_service_connectivity():
    """Test connectivity to all monitoring services"""
    results = {}
    
    services_to_test = [
        ("prometheus", prometheus_service),
        ("loki", loki_service),
        ("jaeger", jaeger_service),
        ("grafana", grafana_service),
        ("alertmanager", alertmanager_service)
    ]
    
    for service_name, service in services_to_test:
        try:
            health = await service.health_check()
            results[service_name] = {
                "status": health.get("status", "unknown"),
                "reachable": health.get("status") == "healthy"
            }
        except Exception as e:
            results[service_name] = {
                "status": "error",
                "reachable": False,
                "error": str(e)
            }
    
    return results

@router.get("/datasources/test")
async def test_datasources():
    """Test all data source connections"""
    # This would test connections from monitoring service to data sources
    return {
        "message": "Data source testing endpoint",
        "note": "Implement specific data source tests as needed"
    }