# monitoring-service/src/core/health_checker.py
"""Health checking utilities for monitoring components"""

import asyncio
import httpx
import docker
import psutil
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

from core.config import settings

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class HealthChecker:
    """Centralized health checker for all monitoring components"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=settings.HEALTH_CHECK_TIMEOUT)
        self.docker_client = docker.from_env()
        self.start_time = time.time()
        self.maintenance_mode = False
        self.health_cache = {}
        self.cache_ttl = 10  # seconds
        
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all monitoring services"""
        if self.maintenance_mode:
            return {
                "maintenance": {
                    "status": "maintenance",
                    "message": "System is in maintenance mode",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        health_checks = {
            "monitoring-api": self._check_self(),
            "prometheus": self._check_prometheus(),
            "loki": self._check_loki(),
            "jaeger": self._check_jaeger(),
            "grafana": self._check_grafana(),
            "alertmanager": self._check_alertmanager(),
            "otel-collector": self._check_otel_collector()
        }
        
        # Run all health checks concurrently
        results = await asyncio.gather(
            *[check for check in health_checks.values()],
            return_exceptions=True
        )
        
        # Map results back to service names
        health_status = {}
        for (service, _), result in zip(health_checks.items(), results):
            if isinstance(result, Exception):
                health_status[service] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                health_status[service] = result
        
        return health_status
    
    async def _check_self(self) -> Dict[str, Any]:
        """Check monitoring API health"""
        return {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "uptime": self.get_uptime(),
                "version": settings.SERVICE_VERSION,
                "environment": settings.ENVIRONMENT,
                "maintenance_mode": self.maintenance_mode
            }
        }
    
    async def _check_prometheus(self) -> Dict[str, Any]:
        """Check Prometheus health"""
        try:
            # Check HTTP endpoint
            response = await self.client.get(f"{settings.PROMETHEUS_URL}/-/healthy")
            http_healthy = response.status_code == 200
            
            # Check readiness
            ready_response = await self.client.get(f"{settings.PROMETHEUS_URL}/-/ready")
            ready = ready_response.status_code == 200
            
            # Check TSDB status
            tsdb_response = await self.client.get(f"{settings.PROMETHEUS_URL}/api/v1/status/tsdb")
            tsdb_data = tsdb_response.json() if tsdb_response.status_code == 200 else {}
            
            # Check targets
            targets_response = await self.client.get(f"{settings.PROMETHEUS_URL}/api/v1/targets")
            targets_data = targets_response.json() if targets_response.status_code == 200 else {}
            
            # Calculate target health
            active_targets = targets_data.get("data", {}).get("activeTargets", [])
            healthy_targets = [t for t in active_targets if t.get("health") == "up"]
            
            status = HealthStatus.HEALTHY if http_healthy and ready else HealthStatus.UNHEALTHY
            if http_healthy and not ready:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.PROMETHEUS_URL,
                    "ready": ready,
                    "targets": {
                        "total": len(active_targets),
                        "healthy": len(healthy_targets),
                        "unhealthy": len(active_targets) - len(healthy_targets)
                    },
                    "storage": tsdb_data.get("data", {})
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.PROMETHEUS_URL}
            }
    
    async def _check_loki(self) -> Dict[str, Any]:
        """Check Loki health"""
        try:
            # Loki ready endpoint
            response = await self.client.get(f"{settings.LOKI_URL}/ready")
            ready = response.status_code == 200
            
            # Check metrics endpoint
            metrics_response = await self.client.get(f"{settings.LOKI_URL}/metrics")
            has_metrics = metrics_response.status_code == 200
            
            status = HealthStatus.HEALTHY if ready else HealthStatus.UNHEALTHY
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.LOKI_URL,
                    "ready": ready,
                    "metrics_available": has_metrics
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.LOKI_URL}
            }
    
    async def _check_jaeger(self) -> Dict[str, Any]:
        """Check Jaeger health"""
        try:
            # Jaeger health endpoint
            response = await self.client.get(f"{settings.JAEGER_URL}/")
            ui_healthy = response.status_code == 200
            
            # Check API
            api_response = await self.client.get(f"{settings.JAEGER_URL}/api/services")
            api_healthy = api_response.status_code == 200
            
            status = HealthStatus.HEALTHY if ui_healthy and api_healthy else HealthStatus.UNHEALTHY
            if ui_healthy and not api_healthy:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.JAEGER_URL,
                    "ui_healthy": ui_healthy,
                    "api_healthy": api_healthy
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.JAEGER_URL}
            }
    
    async def _check_grafana(self) -> Dict[str, Any]:
        """Check Grafana health"""
        try:
            # Grafana health endpoint
            response = await self.client.get(f"{settings.GRAFANA_URL}/api/health")
            health_data = response.json() if response.status_code == 200 else {}
            
            # Check database
            db_status = health_data.get("database", "unknown")
            
            status = HealthStatus.HEALTHY if response.status_code == 200 and db_status == "ok" else HealthStatus.UNHEALTHY
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.GRAFANA_URL,
                    "version": health_data.get("version", "unknown"),
                    "database": db_status
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.GRAFANA_URL}
            }
    
    async def _check_alertmanager(self) -> Dict[str, Any]:
        """Check AlertManager health"""
        try:
            # AlertManager health endpoint
            response = await self.client.get(f"{settings.ALERTMANAGER_URL}/-/healthy")
            healthy = response.status_code == 200
            
            # Check readiness
            ready_response = await self.client.get(f"{settings.ALERTMANAGER_URL}/-/ready")
            ready = ready_response.status_code == 200
            
            # Get status
            status_response = await self.client.get(f"{settings.ALERTMANAGER_URL}/api/v1/status")
            status_data = status_response.json() if status_response.status_code == 200 else {}
            
            status = HealthStatus.HEALTHY if healthy and ready else HealthStatus.UNHEALTHY
            if healthy and not ready:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.ALERTMANAGER_URL,
                    "ready": ready,
                    "cluster_status": status_data.get("data", {})
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.ALERTMANAGER_URL}
            }
    
    async def _check_otel_collector(self) -> Dict[str, Any]:
        """Check OpenTelemetry Collector health"""
        try:
            # OTEL Collector health endpoint
            response = await self.client.get(f"http://otel-collector:13133/")
            healthy = response.status_code == 200
            
            # Check metrics endpoint
            metrics_response = await self.client.get(f"http://otel-collector:8888/metrics")
            has_metrics = metrics_response.status_code == 200
            
            status = HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": settings.OTEL_COLLECTOR_URL,
                    "health_check_endpoint": "http://otel-collector:13133/",
                    "metrics_available": has_metrics
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": settings.OTEL_COLLECTOR_URL}
            }
    
    async def check_docker(self) -> Dict[str, Any]:
        """Check Docker daemon health"""
        try:
            # Check Docker info
            info = self.docker_client.info()
            
            # Check containers
            containers = self.docker_client.containers.list(all=True)
            monitoring_containers = [
                c for c in containers 
                if any(label.startswith("com.monitoring") for label in c.labels)
            ]
            
            running_containers = [c for c in monitoring_containers if c.status == "running"]
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "docker_version": info.get("ServerVersion", "unknown"),
                    "total_containers": len(containers),
                    "monitoring_containers": len(monitoring_containers),
                    "running_monitoring_containers": len(running_containers),
                    "storage_driver": info.get("Driver", "unknown")
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def check_network(self, network_name: str) -> Dict[str, Any]:
        """Check Docker network health"""
        try:
            network = self.docker_client.networks.get(network_name)
            
            # Get connected containers
            connected_containers = []
            for container_id in network.attrs.get("Containers", {}):
                try:
                    container = self.docker_client.containers.get(container_id)
                    connected_containers.append(container.name)
                except:
                    pass
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "network_name": network_name,
                    "driver": network.attrs.get("Driver", "unknown"),
                    "connected_containers": connected_containers,
                    "subnet": network.attrs.get("IPAM", {}).get("Config", [{}])[0].get("Subnet", "unknown")
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/')
            
            # Check if disk space is low
            status = HealthStatus.HEALTHY
            if disk_usage.percent > 90:
                status = HealthStatus.UNHEALTHY
            elif disk_usage.percent > 80:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "used_gb": round(disk_usage.used / (1024**3), 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "percent_used": disk_usage.percent
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            # Check if memory usage is high
            status = HealthStatus.HEALTHY
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent_used": memory.percent
                }
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    def enable_maintenance_mode(self):
        """Enable maintenance mode"""
        self.maintenance_mode = True
        logger.info("Maintenance mode enabled")
    
    def disable_maintenance_mode(self):
        """Disable maintenance mode"""
        self.maintenance_mode = False
        logger.info("Maintenance mode disabled")
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Global health checker instance
health_checker = HealthChecker()