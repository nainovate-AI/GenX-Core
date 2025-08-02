# monitoring-service/src/api/v1/health.py
"""Health check API endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

from core.health_checker import HealthChecker
from services import (
    prometheus_service,
    loki_service,
    jaeger_service,
    grafana_service,
    alertmanager_service,
    otel_collector_service
)

router = APIRouter()

# Health check models
class ServiceHealth(BaseModel):
    """Individual service health status"""
    name: str
    status: str  # healthy, unhealthy, degraded
    timestamp: datetime
    details: Optional[Dict] = None
    error: Optional[str] = None

class OverallHealth(BaseModel):
    """Overall system health status"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    services: List[ServiceHealth]
    healthy_count: int
    unhealthy_count: int
    total_count: int

# Initialize health checker
health_checker = HealthChecker()

@router.get("/", response_model=OverallHealth)
async def health_check():
    """
    Get overall health status of the monitoring system
    
    Returns health status for all monitoring components
    """
    health_status = await health_checker.check_all_services()
    
    services = []
    healthy_count = 0
    unhealthy_count = 0
    
    for service_name, status in health_status.items():
        service_health = ServiceHealth(
            name=service_name,
            status=status.get("status", "unknown"),
            timestamp=datetime.fromisoformat(status.get("timestamp", datetime.utcnow().isoformat())),
            details=status.get("details"),
            error=status.get("error")
        )
        services.append(service_health)
        
        if service_health.status == "healthy":
            healthy_count += 1
        else:
            unhealthy_count += 1
    
    # Determine overall status
    total_count = len(services)
    if unhealthy_count == 0:
        overall_status = "healthy"
    elif unhealthy_count < total_count / 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return OverallHealth(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services,
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        total_count=total_count
    )

@router.get("/service/{service_name}", response_model=ServiceHealth)
async def health_check_service(service_name: str):
    """Get health status for a specific service"""
    service_map = {
        "prometheus": prometheus_service,
        "loki": loki_service,
        "jaeger": jaeger_service,
        "grafana": grafana_service,
        "alertmanager": alertmanager_service,
        "otel-collector": otel_collector_service,
        "monitoring-api": None  # Special case for self
    }
    
    if service_name not in service_map:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service_name}' not found"
        )
    
    if service_name == "monitoring-api":
        # Self health check
        return ServiceHealth(
            name="monitoring-api",
            status="healthy",
            timestamp=datetime.utcnow(),
            details={
                "uptime": health_checker.get_uptime(),
                "version": "1.0.0"
            }
        )
    
    try:
        service = service_map[service_name]
        health_status = await service.health_check()
        
        return ServiceHealth(
            name=service_name,
            status=health_status.get("status", "unknown"),
            timestamp=datetime.fromisoformat(health_status.get("timestamp", datetime.utcnow().isoformat())),
            details=health_status.get("details"),
            error=health_status.get("error")
        )
    except Exception as e:
        return ServiceHealth(
            name=service_name,
            status="unhealthy",
            timestamp=datetime.utcnow(),
            error=str(e)
        )

@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    
    Returns 200 if the service is alive
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    
    Returns 200 if the service is ready to accept traffic
    """
    # Check if critical services are available
    critical_services = ["prometheus", "otel-collector"]
    
    for service_name in critical_services:
        if service_name == "prometheus":
            health = await prometheus_service.health_check()
        elif service_name == "otel-collector":
            health = await otel_collector_service.health_check()
        
        if health.get("status") != "healthy":
            raise HTTPException(
                status_code=503,
                detail=f"Critical service {service_name} is not healthy"
            )
    
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

@router.get("/metrics/health")
async def health_metrics():
    """Get health metrics in Prometheus format"""
    health_status = await health_checker.check_all_services()
    
    metrics = []
    metrics.append("# HELP monitoring_service_health Service health status (1=healthy, 0=unhealthy)")
    metrics.append("# TYPE monitoring_service_health gauge")
    
    for service_name, status in health_status.items():
        value = 1 if status.get("status") == "healthy" else 0
        metrics.append(f'monitoring_service_health{{service="{service_name}"}} {value}')
    
    metrics.append("")
    metrics.append("# HELP monitoring_service_uptime_seconds Service uptime in seconds")
    metrics.append("# TYPE monitoring_service_uptime_seconds counter")
    metrics.append(f"monitoring_service_uptime_seconds {health_checker.get_uptime()}")
    
    return "\n".join(metrics)

@router.get("/dependencies")
async def check_dependencies():
    """Check all service dependencies"""
    dependencies = {
        "docker": await health_checker.check_docker(),
        "network": await health_checker.check_network("monitoring-network"),
        "disk_space": await health_checker.check_disk_space(),
        "memory": await health_checker.check_memory()
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies
    }

@router.post("/maintenance/{action}")
async def maintenance_mode(action: str):
    """Enable or disable maintenance mode"""
    if action not in ["enable", "disable"]:
        raise HTTPException(
            status_code=400,
            detail="Action must be 'enable' or 'disable'"
        )
    
    if action == "enable":
        health_checker.enable_maintenance_mode()
        return {"status": "maintenance mode enabled"}
    else:
        health_checker.disable_maintenance_mode()
        return {"status": "maintenance mode disabled"}