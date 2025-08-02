# monitoring-service/src/services/grafana_service.py
"""Grafana service management"""

import asyncio
import httpx
import docker
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import base64

from core.config import settings
from core.opentelemetry_setup import trace_function

logger = logging.getLogger(__name__)

class GrafanaService:
    """Service for managing Grafana"""
    
    def __init__(self):
        self.base_url = settings.GRAFANA_URL
        self.api_url = f"{self.base_url}/api"
        self.admin_user = settings.GRAFANA_ADMIN_USER
        self.admin_password = settings.GRAFANA_ADMIN_PASSWORD
        self.org_id = settings.GRAFANA_ORG_ID
        
        # Create authenticated client
        auth_str = f"{self.admin_user}:{self.admin_password}"
        auth_bytes = auth_str.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/json"
            }
        )
        
        self.docker_client = docker.from_env()
        self.container_name = "grafana"
        self.dashboards_path = settings.DASHBOARDS_PATH
        
    async def start(self):
        """Start Grafana container if not already running"""
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
            logger.error(f"Failed to start Grafana: {e}")
            raise
            
    async def stop(self):
        """Stop Grafana gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop Grafana: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for Grafana to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    logger.info("Grafana is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("Grafana failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check Grafana health"""
        try:
            response = await self.client.get(f"{self.api_url}/health")
            health_data = response.json() if response.status_code == 200 else {}
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "database": health_data.get("database", "unknown"),
                    "version": health_data.get("version", "unknown"),
                    "commit": health_data.get("commit", "unknown")
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
    async def list_dashboards(self, folder_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all dashboards"""
        try:
            params = {}
            if folder_id is not None:
                params["folderIds"] = folder_id
                
            response = await self.client.get(
                f"{self.api_url}/search",
                params={**params, "type": "dash-db"}
            )
            response.raise_for_status()
            
            dashboards = response.json()
            return dashboards
            
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            raise
    
    @trace_function()
    async def get_dashboard(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get dashboard by UID"""
        try:
            response = await self.client.get(f"{self.api_url}/dashboards/uid/{uid}")
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get dashboard {uid}: {e}")
            raise
    
    @trace_function()
    async def import_dashboard(
        self,
        dashboard: Dict[str, Any],
        folder_id: int = 0,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """Import a dashboard"""
        try:
            # Prepare dashboard for import
            import_data = {
                "dashboard": dashboard,
                "folderId": folder_id,
                "overwrite": overwrite
            }
            
            # Remove id and uid from dashboard if present (for new dashboards)
            if "id" in dashboard:
                del dashboard["id"]
            if not overwrite and "uid" in dashboard:
                del dashboard["uid"]
            
            response = await self.client.post(
                f"{self.api_url}/dashboards/db",
                json=import_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Imported dashboard: {result.get('uid')} - {result.get('title')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to import dashboard: {e}")
            raise
    
    @trace_function()
    async def export_dashboard(self, uid: str) -> Optional[Dict[str, Any]]:
        """Export dashboard for backup or sharing"""
        try:
            dashboard_data = await self.get_dashboard(uid)
            if not dashboard_data:
                return None
            
            # Clean dashboard for export
            dashboard = dashboard_data.get("dashboard", {})
            
            # Remove instance-specific fields
            fields_to_remove = ["id", "uid", "version", "iteration"]
            for field in fields_to_remove:
                dashboard.pop(field, None)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to export dashboard {uid}: {e}")
            raise
    
    @trace_function()
    async def delete_dashboard(self, uid: str) -> bool:
        """Delete dashboard by UID"""
        try:
            response = await self.client.delete(f"{self.api_url}/dashboards/uid/{uid}")
            response.raise_for_status()
            
            logger.info(f"Deleted dashboard {uid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dashboard {uid}: {e}")
            return False
    
    @trace_function()
    async def list_datasources(self) -> List[Dict[str, Any]]:
        """List all data sources"""
        try:
            response = await self.client.get(f"{self.api_url}/datasources")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list datasources: {e}")
            raise
    
    @trace_function()
    async def get_datasource(self, name: str) -> Optional[Dict[str, Any]]:
        """Get data source by name"""
        try:
            response = await self.client.get(f"{self.api_url}/datasources/name/{name}")
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get datasource {name}: {e}")
            raise
    
    @trace_function()
    async def create_datasource(self, datasource: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new data source"""
        try:
            response = await self.client.post(
                f"{self.api_url}/datasources",
                json=datasource
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created datasource: {result.get('name')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create datasource: {e}")
            raise
    
    @trace_function()
    async def update_datasource(self, id: int, datasource: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing data source"""
        try:
            response = await self.client.put(
                f"{self.api_url}/datasources/{id}",
                json=datasource
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Updated datasource: {datasource.get('name')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update datasource: {e}")
            raise
    
    @trace_function()
    async def test_datasource(self, name: str) -> Dict[str, Any]:
        """Test data source connection"""
        try:
            # Get datasource first
            datasource = await self.get_datasource(name)
            if not datasource:
                return {"status": "error", "message": f"Datasource {name} not found"}
            
            # Test connection
            response = await self.client.post(
                f"{self.api_url}/datasources/{datasource['id']}/resources/test"
            )
            
            return {
                "status": "success" if response.status_code == 200 else "error",
                "message": response.json().get("message", ""),
                "details": response.json()
            }
            
        except Exception as e:
            logger.error(f"Failed to test datasource {name}: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_function()
    async def list_folders(self) -> List[Dict[str, Any]]:
        """List all folders"""
        try:
            response = await self.client.get(f"{self.api_url}/folders")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list folders: {e}")
            raise
    
    @trace_function()
    async def create_folder(self, title: str, uid: Optional[str] = None) -> Dict[str, Any]:
        """Create a new folder"""
        try:
            folder_data = {"title": title}
            if uid:
                folder_data["uid"] = uid
            
            response = await self.client.post(
                f"{self.api_url}/folders",
                json=folder_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created folder: {result.get('title')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            raise
    
    @trace_function()
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        try:
            response = await self.client.get(f"{self.api_url}/users")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise
    
    @trace_function()
    async def create_api_key(
        self,
        name: str,
        role: str = "Viewer",
        seconds_to_live: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create an API key"""
        try:
            key_data = {
                "name": name,
                "role": role
            }
            if seconds_to_live:
                key_data["secondsToLive"] = seconds_to_live
            
            response = await self.client.post(
                f"{self.api_url}/auth/keys",
                json=key_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created API key: {name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    @trace_function()
    async def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys"""
        try:
            response = await self.client.get(f"{self.api_url}/auth/keys")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            raise
    
    @trace_function()
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get Grafana alerts (legacy alerts)"""
        try:
            response = await self.client.get(f"{self.api_url}/alerts")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise
    
    @trace_function()
    async def get_alert_notifications(self) -> List[Dict[str, Any]]:
        """Get alert notification channels"""
        try:
            response = await self.client.get(f"{self.api_url}/alert-notifications")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get alert notifications: {e}")
            raise
    
    @trace_function()
    async def create_alert_notification(
        self,
        name: str,
        type: str,
        settings: Dict[str, Any],
        is_default: bool = False
    ) -> Dict[str, Any]:
        """Create alert notification channel"""
        try:
            notification_data = {
                "name": name,
                "type": type,
                "isDefault": is_default,
                "sendReminder": False,
                "disableResolveMessage": False,
                "settings": settings
            }
            
            response = await self.client.post(
                f"{self.api_url}/alert-notifications",
                json=notification_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created alert notification: {name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create alert notification: {e}")
            raise
    
    @trace_function()
    async def test_alert_notification(self, id: int) -> Dict[str, Any]:
        """Test alert notification channel"""
        try:
            response = await self.client.post(
                f"{self.api_url}/alert-notifications/test",
                json={"id": id}
            )
            
            return {
                "status": "success" if response.status_code == 200 else "error",
                "message": response.json().get("message", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to test alert notification: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_function()
    async def search_dashboards(
        self,
        query: str = "",
        tag: Optional[List[str]] = None,
        starred: Optional[bool] = None,
        folder_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Search dashboards with filters"""
        try:
            params = {"query": query}
            
            if tag:
                params["tag"] = tag
            if starred is not None:
                params["starred"] = str(starred).lower()
            if folder_ids:
                params["folderIds"] = ",".join(map(str, folder_ids))
            
            response = await self.client.get(
                f"{self.api_url}/search",
                params=params
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to search dashboards: {e}")
            raise
    
    @trace_function()
    async def get_dashboard_versions(self, uid: str) -> List[Dict[str, Any]]:
        """Get dashboard version history"""
        try:
            dashboard = await self.get_dashboard(uid)
            if not dashboard:
                return []
            
            dashboard_id = dashboard["dashboard"]["id"]
            
            response = await self.client.get(
                f"{self.api_url}/dashboards/id/{dashboard_id}/versions"
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get dashboard versions: {e}")
            raise
    
    @trace_function()
    async def restore_dashboard_version(
        self,
        uid: str,
        version: int
    ) -> Dict[str, Any]:
        """Restore dashboard to specific version"""
        try:
            dashboard = await self.get_dashboard(uid)
            if not dashboard:
                raise ValueError(f"Dashboard {uid} not found")
            
            dashboard_id = dashboard["dashboard"]["id"]
            
            # Get specific version
            response = await self.client.get(
                f"{self.api_url}/dashboards/id/{dashboard_id}/versions/{version}"
            )
            response.raise_for_status()
            
            version_data = response.json()
            
            # Restore the version
            return await self.import_dashboard(
                version_data["data"],
                folder_id=dashboard["meta"]["folderId"],
                overwrite=True
            )
            
        except Exception as e:
            logger.error(f"Failed to restore dashboard version: {e}")
            raise
    
    @trace_function()
    async def get_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            response = await self.client.get(f"{self.api_url}/user/preferences")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            raise
    
    @trace_function()
    async def update_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        try:
            response = await self.client.put(
                f"{self.api_url}/user/preferences",
                json=preferences
            )
            response.raise_for_status()
            
            return {"status": "success", "message": "Preferences updated"}
            
        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            raise
    
    @trace_function()
    async def get_org_preferences(self) -> Dict[str, Any]:
        """Get organization preferences"""
        try:
            response = await self.client.get(f"{self.api_url}/org/preferences")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get org preferences: {e}")
            raise
    
    @trace_function()
    async def import_dashboards_from_directory(
        self,
        directory: Optional[Path] = None,
        folder_id: int = 0
    ) -> List[Dict[str, Any]]:
        """Import all dashboards from a directory"""
        try:
            dashboard_dir = directory or self.dashboards_path
            if not dashboard_dir.exists():
                logger.warning(f"Dashboard directory not found: {dashboard_dir}")
                return []
            
            results = []
            
            for dashboard_file in dashboard_dir.glob("*.json"):
                try:
                    with open(dashboard_file, 'r') as f:
                        dashboard = json.load(f)
                    
                    result = await self.import_dashboard(dashboard, folder_id)
                    results.append({
                        "file": dashboard_file.name,
                        "status": "success",
                        "uid": result.get("uid"),
                        "title": result.get("title")
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to import {dashboard_file.name}: {e}")
                    results.append({
                        "file": dashboard_file.name,
                        "status": "error",
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to import dashboards from directory: {e}")
            raise
    
    @trace_function()
    async def export_all_dashboards(
        self,
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Export all dashboards to directory"""
        try:
            output_path = output_dir or Path("./dashboard_exports")
            output_path.mkdir(parents=True, exist_ok=True)
            
            dashboards = await self.list_dashboards()
            results = []
            
            for dashboard_meta in dashboards:
                try:
                    dashboard = await self.export_dashboard(dashboard_meta["uid"])
                    if dashboard:
                        filename = f"{dashboard_meta['uid']}_{dashboard_meta['title']}.json"
                        # Sanitize filename
                        filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                        
                        with open(output_path / filename, 'w') as f:
                            json.dump(dashboard, f, indent=2)
                        
                        results.append({
                            "uid": dashboard_meta["uid"],
                            "title": dashboard_meta["title"],
                            "file": filename,
                            "status": "success"
                        })
                    
                except Exception as e:
                    logger.error(f"Failed to export {dashboard_meta['uid']}: {e}")
                    results.append({
                        "uid": dashboard_meta["uid"],
                        "title": dashboard_meta["title"],
                        "status": "error",
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to export dashboards: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
grafana_service = GrafanaService()