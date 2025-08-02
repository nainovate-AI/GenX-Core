# monitoring-service/src/services/alertmanager_service.py
"""AlertManager service management"""

import asyncio
import httpx
import docker
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from core.config import settings
from core.opentelemetry_setup import trace_function
from models.alerts import (
    Alert, AlertGroup, AlertmanagerStatus, Silence,
    CreateSilenceRequest, AlertQueryParams, Receiver,
    Route, AlertmanagerConfig, AlertWebhookPayload
)

logger = logging.getLogger(__name__)

class AlertManagerService:
    """Service for managing AlertManager"""
    
    def __init__(self):
        self.base_url = settings.ALERTMANAGER_URL
        self.api_v1_url = f"{self.base_url}/api/v1"
        self.api_v2_url = f"{self.base_url}/api/v2"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.docker_client = docker.from_env()
        self.container_name = "alertmanager"
        self.config_path = settings.CONFIG_PATH / "alertmanager" / "alertmanager.yml"
        
    async def start(self):
        """Start AlertManager container if not already running"""
        try:
            # Check if container exists
            try:
                container = self.docker_client.containers.get(self.container_name)
                if container.status != "running":
                    logger.info(f"Starting existing {self.container_name} container")
                    container.start()
                    await self._wait_for_ready()
                else:
                    logger.info(f"{self.container_name} is already running")
            except docker.errors.NotFound:
                logger.info(f"Container {self.container_name} not found, will be created by docker-compose")
                await self._wait_for_ready()
                
        except Exception as e:
            logger.error(f"Failed to start AlertManager: {e}")
            raise
            
    async def stop(self):
        """Stop AlertManager gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop AlertManager: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for AlertManager to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/-/ready")
                if response.status_code == 200:
                    logger.info("AlertManager is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("AlertManager failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check AlertManager health"""
        try:
            # Check health endpoint
            health_response = await self.client.get(f"{self.base_url}/-/healthy")
            healthy = health_response.status_code == 200
            
            # Check readiness
            ready_response = await self.client.get(f"{self.base_url}/-/ready")
            ready = ready_response.status_code == 200
            
            # Get status
            status_data = await self.get_status()
            
            return {
                "status": "healthy" if healthy and ready else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "healthy": healthy,
                    "ready": ready,
                    "cluster_status": status_data.cluster_status if status_data else "unknown",
                    "version": status_data.version_info.get("version") if status_data else "unknown"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": self.base_url}
            }
    
    @trace_function()
    async def get_status(self) -> Optional[AlertmanagerStatus]:
        """Get AlertManager status"""
        try:
            response = await self.client.get(f"{self.api_v1_url}/status")
            response.raise_for_status()
            data = response.json()
            
            return AlertmanagerStatus(
                cluster_status=data.get("cluster", {}).get("status", "unknown"),
                version_info=data.get("versionInfo", {}),
                uptime=data.get("uptime", "0s"),
                config=data.get("config", {})
            )
        except Exception as e:
            logger.error(f"Failed to get AlertManager status: {e}")
            return None
    
    @trace_function()
    async def get_alerts(self, params: Optional[AlertQueryParams] = None) -> List[AlertGroup]:
        """Get all alerts grouped by their grouping labels"""
        try:
            # Build query parameters
            query_params = {}
            if params:
                if params.state:
                    query_params["filter"] = f'state="{params.state.value}"'
                if params.receiver:
                    query_params["receiver"] = params.receiver
                if params.silenced is not None:
                    query_params["silenced"] = str(params.silenced).lower()
                if params.inhibited is not None:
                    query_params["inhibited"] = str(params.inhibited).lower()
                if params.active is not None:
                    query_params["active"] = str(params.active).lower()
                if params.unprocessed is not None:
                    query_params["unprocessed"] = str(params.unprocessed).lower()
            
            response = await self.client.get(
                f"{self.api_v2_url}/alerts/groups",
                params=query_params
            )
            response.raise_for_status()
            
            groups = []
            for group_data in response.json():
                group = AlertGroup(
                    group_key=group_data["groupKey"],
                    group_labels=group_data["labels"],
                    alerts=[Alert(**alert) for alert in group_data["alerts"]],
                    total=len(group_data["alerts"]),
                    firing=sum(1 for a in group_data["alerts"] if a["state"] == "firing"),
                    resolved=sum(1 for a in group_data["alerts"] if a["state"] == "resolved"),
                    suppressed=sum(1 for a in group_data["alerts"] if a.get("status", {}).get("state") == "suppressed")
                )
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise
    
    @trace_function()
    async def get_alert_by_fingerprint(self, fingerprint: str) -> Optional[Alert]:
        """Get a specific alert by fingerprint"""
        try:
            groups = await self.get_alerts()
            for group in groups:
                for alert in group.alerts:
                    if alert.fingerprint == fingerprint:
                        return alert
            return None
        except Exception as e:
            logger.error(f"Failed to get alert {fingerprint}: {e}")
            return None
    
    @trace_function()
    async def create_silence(self, request: CreateSilenceRequest) -> str:
        """Create a new silence"""
        try:
            silence_data = {
                "matchers": request.matchers,
                "startsAt": request.starts_at.isoformat() if request.starts_at else datetime.utcnow().isoformat(),
                "endsAt": request.ends_at.isoformat(),
                "createdBy": request.created_by,
                "comment": request.comment
            }
            
            response = await self.client.post(
                f"{self.api_v2_url}/silences",
                json=silence_data
            )
            response.raise_for_status()
            
            result = response.json()
            silence_id = result.get("silenceID")
            logger.info(f"Created silence {silence_id}")
            return silence_id
            
        except Exception as e:
            logger.error(f"Failed to create silence: {e}")
            raise
    
    @trace_function()
    async def get_silences(self) -> List[Silence]:
        """Get all silences"""
        try:
            response = await self.client.get(f"{self.api_v2_url}/silences")
            response.raise_for_status()
            
            silences = []
            for silence_data in response.json():
                silence = Silence(
                    id=silence_data["id"],
                    matchers=silence_data["matchers"],
                    starts_at=datetime.fromisoformat(silence_data["startsAt"].replace('Z', '+00:00')),
                    ends_at=datetime.fromisoformat(silence_data["endsAt"].replace('Z', '+00:00')),
                    created_by=silence_data["createdBy"],
                    comment=silence_data["comment"],
                    status=silence_data.get("status", {})
                )
                silences.append(silence)
            
            return silences
            
        except Exception as e:
            logger.error(f"Failed to get silences: {e}")
            raise
    
    @trace_function()
    async def get_silence(self, silence_id: str) -> Optional[Silence]:
        """Get a specific silence"""
        try:
            response = await self.client.get(f"{self.api_v2_url}/silence/{silence_id}")
            response.raise_for_status()
            
            data = response.json()
            return Silence(
                id=data["id"],
                matchers=data["matchers"],
                starts_at=datetime.fromisoformat(data["startsAt"].replace('Z', '+00:00')),
                ends_at=datetime.fromisoformat(data["endsAt"].replace('Z', '+00:00')),
                created_by=data["createdBy"],
                comment=data["comment"],
                status=data.get("status", {})
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get silence {silence_id}: {e}")
            raise
    
    @trace_function()
    async def delete_silence(self, silence_id: str) -> bool:
        """Delete (expire) a silence"""
        try:
            response = await self.client.delete(f"{self.api_v2_url}/silence/{silence_id}")
            response.raise_for_status()
            logger.info(f"Deleted silence {silence_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete silence {silence_id}: {e}")
            return False
    
    @trace_function()
    async def get_receivers(self) -> List[str]:
        """Get list of configured receivers"""
        try:
            status = await self.get_status()
            if status and status.config:
                config = status.config
                # Parse receivers from config
                receivers = []
                if "receivers" in config:
                    for receiver in config["receivers"]:
                        receivers.append(receiver.get("name", ""))
                return receivers
            return []
            
        except Exception as e:
            logger.error(f"Failed to get receivers: {e}")
            return []
    
    @trace_function()
    async def reload_config(self) -> bool:
        """Reload AlertManager configuration"""
        try:
            response = await self.client.post(f"{self.base_url}/-/reload")
            success = response.status_code == 200
            
            if success:
                logger.info("AlertManager configuration reloaded successfully")
            else:
                logger.warning(f"AlertManager reload returned status {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to reload AlertManager config: {e}")
            return False
    
    @trace_function()
    async def get_config(self) -> Optional[AlertmanagerConfig]:
        """Get current AlertManager configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Parse configuration
                return AlertmanagerConfig(
                    global_config=config_data.get("global"),
                    route=Route(**config_data.get("route", {})),
                    receivers=[Receiver(**r) for r in config_data.get("receivers", [])],
                    inhibit_rules=config_data.get("inhibit_rules", []),
                    templates=config_data.get("templates", [])
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get AlertManager config: {e}")
            return None
    
    @trace_function()
    async def update_config(self, config: AlertmanagerConfig) -> bool:
        """Update AlertManager configuration"""
        try:
            # Convert config to dict
            config_dict = config.dict(by_alias=True, exclude_none=True)
            
            # Write to file
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            # Reload configuration
            return await self.reload_config()
            
        except Exception as e:
            logger.error(f"Failed to update AlertManager config: {e}")
            return False
    
    @trace_function()
    async def send_test_alert(self, receiver: str = "default") -> bool:
        """Send a test alert to verify configuration"""
        try:
            test_alert = {
                "labels": {
                    "alertname": "TestAlert",
                    "severity": "info",
                    "service": "monitoring-service",
                    "instance": "test"
                },
                "annotations": {
                    "summary": "Test alert from monitoring service",
                    "description": "This is a test alert to verify AlertManager configuration"
                },
                "generatorURL": f"{settings.PROMETHEUS_URL}/graph",
                "startsAt": datetime.utcnow().isoformat() + "Z"
            }
            
            response = await self.client.post(
                f"{self.api_v1_url}/alerts",
                json=[test_alert]
            )
            
            success = response.status_code in [200, 202]
            if success:
                logger.info(f"Test alert sent to receiver '{receiver}'")
            else:
                logger.warning(f"Test alert failed with status {response.status_code}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to send test alert: {e}")
            return False
    
    @trace_function()
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            groups = await self.get_alerts()
            
            total_alerts = sum(group.total for group in groups)
            firing_alerts = sum(group.firing for group in groups)
            resolved_alerts = sum(group.resolved for group in groups)
            suppressed_alerts = sum(group.suppressed for group in groups)
            
            # Count by severity
            severity_counts = {"critical": 0, "warning": 0, "info": 0}
            service_counts = {}
            
            for group in groups:
                for alert in group.alerts:
                    # Count by severity
                    severity = alert.labels.get("severity", "unknown")
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                    
                    # Count by service
                    service = alert.labels.get("service", "unknown")
                    service_counts[service] = service_counts.get(service, 0) + 1
            
            return {
                "total_alerts": total_alerts,
                "firing_alerts": firing_alerts,
                "resolved_alerts": resolved_alerts,
                "suppressed_alerts": suppressed_alerts,
                "pending_alerts": total_alerts - firing_alerts - resolved_alerts,
                "alert_groups": len(groups),
                "by_severity": severity_counts,
                "by_service": service_counts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {}
    
    @trace_function()
    async def handle_webhook(self, payload: AlertWebhookPayload) -> Dict[str, Any]:
        """Handle incoming webhook from AlertManager"""
        try:
            # Log the webhook
            logger.info(
                f"Received webhook: {payload.status} for group {payload.group_key} "
                f"with {len(payload.alerts)} alerts"
            )
            
            # Process alerts based on status
            for alert in payload.alerts:
                if alert.state == AlertState.FIRING:
                    logger.warning(
                        f"FIRING Alert: {alert.labels.get('alertname')} - "
                        f"{alert.annotations.get('summary', 'No summary')}"
                    )
                elif alert.state == AlertState.RESOLVED:
                    logger.info(
                        f"RESOLVED Alert: {alert.labels.get('alertname')} - "
                        f"{alert.annotations.get('summary', 'No summary')}"
                    )
            
            # You can add custom processing here:
            # - Send to external systems
            # - Store in database
            # - Trigger automated responses
            # - Update dashboards
            
            return {
                "status": "processed",
                "group_key": payload.group_key,
                "alerts_received": len(payload.alerts),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to handle webhook: {e}")
            raise
    
    @trace_function()
    async def get_alert_timeline(
        self,
        hours: int = 24,
        service: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alert timeline for visualization"""
        try:
            # This would typically query a database where alerts are stored
            # For now, return current alerts with estimated timeline
            groups = await self.get_alerts()
            timeline = []
            
            for group in groups:
                for alert in group.alerts:
                    if service and alert.labels.get("service") != service:
                        continue
                    
                    timeline.append({
                        "timestamp": alert.starts_at.isoformat(),
                        "event": "alert_started",
                        "alertname": alert.labels.get("alertname"),
                        "severity": alert.labels.get("severity"),
                        "service": alert.labels.get("service"),
                        "state": alert.state.value
                    })
                    
                    if alert.ends_at and alert.state == AlertState.RESOLVED:
                        timeline.append({
                            "timestamp": alert.ends_at.isoformat(),
                            "event": "alert_resolved",
                            "alertname": alert.labels.get("alertname"),
                            "severity": alert.labels.get("severity"),
                            "service": alert.labels.get("service"),
                            "state": "resolved"
                        })
            
            # Sort by timestamp
            timeline.sort(key=lambda x: x["timestamp"])
            
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to get alert timeline: {e}")
            return []
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
alertmanager_service = AlertManagerService()