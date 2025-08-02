# monitoring-service/src/services/prometheus_service.py
"""Prometheus service management for metrics storage and querying"""

import asyncio
import httpx
import docker
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

from core.config import settings
from core.opentelemetry_setup import trace_function
from models.metrics import (
    MetricResponse, MetricQuery, MetricRangeQuery,
    TargetsResponse, Target, RulesResponse, Rule, RuleGroup,
    AlertsResponse, Alert, TSDBStatus, BuildInfo, RuntimeInfo,
    SeriesMetadata, QueryExemplar, SnapshotResponse, MetricMetadata,
    ResultType, TargetHealth
)

logger = logging.getLogger(__name__)

class PrometheusService:
    """Service for managing Prometheus metrics storage"""
    
    def __init__(self):
        self.base_url = settings.PROMETHEUS_URL
        self.api_url = f"{self.base_url}/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for complex queries
        self.docker_client = docker.from_env()
        self.container_name = "prometheus"
        self.config_path = settings.CONFIG_PATH / "prometheus" / "prometheus.yml"
        self.alerts_path = settings.CONFIG_PATH / "prometheus" / "alerts"
        self.rules_path = settings.CONFIG_PATH / "prometheus" / "recording_rules"
        
    async def start(self):
        """Start Prometheus container if not already running"""
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
            logger.error(f"Failed to start Prometheus: {e}")
            raise
            
    async def stop(self):
        """Stop Prometheus gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop Prometheus: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for Prometheus to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/-/ready")
                if response.status_code == 200:
                    logger.info("Prometheus is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("Prometheus failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check Prometheus health"""
        try:
            # Check health endpoint
            health_response = await self.client.get(f"{self.base_url}/-/healthy")
            healthy = health_response.status_code == 200
            
            # Check readiness
            ready_response = await self.client.get(f"{self.base_url}/-/ready")
            ready = ready_response.status_code == 200
            
            # Get build info
            build_info = await self.get_build_info()
            
            # Get runtime info
            runtime_info = await self.get_runtime_info()
            
            # Get TSDB status for storage info
            tsdb_status = await self.get_tsdb_status()
            
            status = "healthy" if healthy and ready else "unhealthy"
            if healthy and not ready:
                status = "degraded"
            
            return {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "healthy": healthy,
                    "ready": ready,
                    "version": build_info.version if build_info else "unknown",
                    "uptime": runtime_info.startTime.isoformat() if runtime_info else "unknown",
                    "storage": {
                        "head_samples": tsdb_status.headStats.get("numSamples", 0) if tsdb_status else 0,
                        "head_series": tsdb_status.headStats.get("numSeries", 0) if tsdb_status else 0
                    }
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
    async def query(self, query: str, time: Optional[str] = None) -> MetricResponse:
        """Execute instant query"""
        try:
            params = {"query": query}
            if time:
                params["time"] = time
            
            response = await self.client.get(
                f"{self.api_url}/query",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return MetricResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    @trace_function()
    async def query_range(
        self,
        query: str,
        start: str,
        end: str,
        step: str = "15s"
    ) -> MetricResponse:
        """Execute range query"""
        try:
            params = {
                "query": query,
                "start": start,
                "end": end,
                "step": step
            }
            
            response = await self.client.get(
                f"{self.api_url}/query_range",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return MetricResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to execute range query: {e}")
            raise
    
    @trace_function()
    async def get_targets(self, state: Optional[str] = None) -> TargetsResponse:
        """Get all scrape targets and their status"""
        try:
            params = {}
            if state:
                params["state"] = state
            
            response = await self.client.get(
                f"{self.api_url}/targets",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse targets
            targets_data = {
                "activeTargets": [],
                "droppedTargets": data.get("data", {}).get("droppedTargets", [])
            }
            
            for target_data in data.get("data", {}).get("activeTargets", []):
                target = Target(**target_data)
                targets_data["activeTargets"].append(target)
            
            return TargetsResponse(data=targets_data)
            
        except Exception as e:
            logger.error(f"Failed to get targets: {e}")
            raise
    
    @trace_function()
    async def get_metadata(self, metric: Optional[str] = None) -> List[MetricMetadata]:
        """Get metric metadata"""
        try:
            params = {}
            if metric:
                params["metric"] = metric
            
            response = await self.client.get(
                f"{self.api_url}/metadata",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            metadata_list = []
            
            for metric_name, metric_info_list in data.get("data", {}).items():
                for metric_info in metric_info_list:
                    metadata = MetricMetadata(
                        metric=metric_name,
                        type=metric_info.get("type", "untyped"),
                        help=metric_info.get("help", ""),
                        unit=metric_info.get("unit")
                    )
                    metadata_list.append(metadata)
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            raise
    
    @trace_function()
    async def get_series(
        self,
        match: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[SeriesMetadata]:
        """Get time series that match label matchers"""
        try:
            params = {"match[]": match}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            
            response = await self.client.get(
                f"{self.api_url}/series",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            series_list = []
            
            for series_data in data.get("data", []):
                series = SeriesMetadata(labels=series_data)
                series_list.append(series)
            
            return series_list
            
        except Exception as e:
            logger.error(f"Failed to get series: {e}")
            raise
    
    @trace_function()
    async def get_labels(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[str]:
        """Get list of label names"""
        try:
            params = {}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            
            response = await self.client.get(
                f"{self.api_url}/labels",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to get labels: {e}")
            raise
    
    @trace_function()
    async def get_label_values(
        self,
        label: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[str]:
        """Get list of label values for a specific label"""
        try:
            params = {}
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            
            response = await self.client.get(
                f"{self.api_url}/label/{label}/values",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to get label values: {e}")
            raise
    
    @trace_function()
    async def get_rules(self, type: Optional[str] = None) -> RulesResponse:
        """Get all configured rules"""
        try:
            response = await self.client.get(f"{self.api_url}/rules")
            response.raise_for_status()
            
            data = response.json()
            rule_groups = []
            
            for group_data in data.get("data", {}).get("groups", []):
                rules = []
                for rule_data in group_data.get("rules", []):
                    rule = Rule(**rule_data)
                    
                    # Filter by type if specified
                    if type and rule.type.value != type:
                        continue
                    
                    rules.append(rule)
                
                # Only add group if it has rules after filtering
                if rules or not type:
                    group = RuleGroup(
                        name=group_data["name"],
                        file=group_data["file"],
                        interval=group_data["interval"],
                        evaluationTime=group_data.get("evaluationTime", 0),
                        lastEvaluation=datetime.fromisoformat(
                            group_data["lastEvaluation"].replace('Z', '+00:00')
                        ),
                        rules=rules
                    )
                    rule_groups.append(group)
            
            return RulesResponse(data={"groups": rule_groups})
            
        except Exception as e:
            logger.error(f"Failed to get rules: {e}")
            raise
    
    @trace_function()
    async def get_alerts(self) -> AlertsResponse:
        """Get all active alerts"""
        try:
            response = await self.client.get(f"{self.api_url}/alerts")
            response.raise_for_status()
            
            data = response.json()
            alerts = []
            
            for alert_data in data.get("data", {}).get("alerts", []):
                alert = Alert(**alert_data)
                alerts.append(alert)
            
            return AlertsResponse(data={"alerts": alerts})
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise
    
    @trace_function()
    async def reload_config(self) -> bool:
        """Reload Prometheus configuration"""
        try:
            response = await self.client.post(f"{self.base_url}/-/reload")
            success = response.status_code == 200
            
            if success:
                logger.info("Prometheus configuration reloaded successfully")
            else:
                logger.warning(f"Prometheus reload returned status {response.status_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to reload Prometheus config: {e}")
            return False
    
    @trace_function()
    async def get_tsdb_status(self) -> Optional[TSDBStatus]:
        """Get TSDB status information"""
        try:
            response = await self.client.get(f"{self.api_url}/status/tsdb")
            response.raise_for_status()
            
            data = response.json()
            return TSDBStatus(**data.get("data", {}))
            
        except Exception as e:
            logger.error(f"Failed to get TSDB status: {e}")
            return None
    
    @trace_function()
    async def get_build_info(self) -> Optional[BuildInfo]:
        """Get Prometheus build information"""
        try:
            response = await self.client.get(f"{self.api_url}/status/buildinfo")
            response.raise_for_status()
            
            data = response.json()
            return BuildInfo(**data.get("data", {}))
            
        except Exception as e:
            logger.error(f"Failed to get build info: {e}")
            return None
    
    @trace_function()
    async def get_runtime_info(self) -> Optional[RuntimeInfo]:
        """Get Prometheus runtime information"""
        try:
            response = await self.client.get(f"{self.api_url}/status/runtimeinfo")
            response.raise_for_status()
            
            data = response.json()
            runtime_data = data.get("data", {})
            
            # Parse timestamps
            if "startTime" in runtime_data:
                runtime_data["startTime"] = datetime.fromisoformat(
                    runtime_data["startTime"].replace('Z', '+00:00')
                )
            if "lastConfigTime" in runtime_data:
                runtime_data["lastConfigTime"] = datetime.fromisoformat(
                    runtime_data["lastConfigTime"].replace('Z', '+00:00')
                )
            
            return RuntimeInfo(**runtime_data)
            
        except Exception as e:
            logger.error(f"Failed to get runtime info: {e}")
            return None
    
    @trace_function()
    async def query_exemplars(
        self,
        query: str,
        start: str,
        end: str
    ) -> List[QueryExemplar]:
        """Query exemplars (trace IDs) for metrics"""
        try:
            params = {
                "query": query,
                "start": start,
                "end": end
            }
            
            response = await self.client.get(
                f"{self.api_url}/query_exemplars",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            exemplars = []
            
            for series_data in data.get("data", []):
                for exemplar_data in series_data.get("exemplars", []):
                    exemplar = QueryExemplar(**exemplar_data)
                    exemplars.append(exemplar)
            
            return exemplars
            
        except Exception as e:
            logger.error(f"Failed to query exemplars: {e}")
            raise
    
    @trace_function()
    async def create_snapshot(self, skip_head: bool = False) -> SnapshotResponse:
        """Create TSDB snapshot"""
        try:
            params = {"skip_head": str(skip_head).lower()}
            
            response = await self.client.post(
                f"{self.api_url}/admin/tsdb/snapshot",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return SnapshotResponse(**data)
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    @trace_function()
    async def delete_series(
        self,
        match: List[str],
        start: str,
        end: str
    ) -> Dict[str, Any]:
        """Delete time series data"""
        try:
            # Note: This requires admin API to be enabled
            params = {
                "match[]": match,
                "start": start,
                "end": end
            }
            
            response = await self.client.post(
                f"{self.api_url}/admin/tsdb/delete_series",
                params=params
            )
            response.raise_for_status()
            
            return {"status": "success", "message": "Series deleted"}
            
        except Exception as e:
            logger.error(f"Failed to delete series: {e}")
            raise
    
    @trace_function()
    async def clean_tombstones(self) -> Dict[str, Any]:
        """Clean up tombstones from deleted series"""
        try:
            response = await self.client.post(
                f"{self.api_url}/admin/tsdb/clean_tombstones"
            )
            response.raise_for_status()
            
            return {"status": "success", "message": "Tombstones cleaned"}
            
        except Exception as e:
            logger.error(f"Failed to clean tombstones: {e}")
            raise
    
    @trace_function()
    async def get_config(self) -> Optional[Dict[str, Any]]:
        """Get current Prometheus configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Prometheus config: {e}")
            return None
    
    @trace_function()
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """Update Prometheus configuration"""
        try:
            # Validate configuration structure
            required_keys = ["global", "scrape_configs"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
            
            # Write configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Reload configuration
            return await self.reload_config()
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus config: {e}")
            return False
    
    @trace_function()
    async def add_scrape_config(self, scrape_config: Dict[str, Any]) -> bool:
        """Add a new scrape configuration"""
        try:
            # Get current config
            config = await self.get_config()
            if not config:
                return False
            
            # Add new scrape config
            if "scrape_configs" not in config:
                config["scrape_configs"] = []
            
            # Check if job already exists
            job_name = scrape_config.get("job_name")
            for sc in config["scrape_configs"]:
                if sc.get("job_name") == job_name:
                    logger.warning(f"Scrape config for job {job_name} already exists")
                    return False
            
            config["scrape_configs"].append(scrape_config)
            
            # Update config
            return await self.update_config(config)
            
        except Exception as e:
            logger.error(f"Failed to add scrape config: {e}")
            return False
    
    @trace_function()
    async def remove_scrape_config(self, job_name: str) -> bool:
        """Remove a scrape configuration"""
        try:
            # Get current config
            config = await self.get_config()
            if not config:
                return False
            
            # Remove scrape config
            if "scrape_configs" in config:
                config["scrape_configs"] = [
                    sc for sc in config["scrape_configs"]
                    if sc.get("job_name") != job_name
                ]
            
            # Update config
            return await self.update_config(config)
            
        except Exception as e:
            logger.error(f"Failed to remove scrape config: {e}")
            return False
    
    @trace_function()
    async def add_alert_rule(
        self,
        group_name: str,
        rule: Dict[str, Any],
        file_name: Optional[str] = None
    ) -> bool:
        """Add a new alert rule"""
        try:
            # Determine file path
            if file_name:
                rule_file = self.alerts_path / file_name
            else:
                rule_file = self.alerts_path / f"{group_name}.yml"
            
            # Load existing rules or create new
            if rule_file.exists():
                with open(rule_file, 'r') as f:
                    rules_config = yaml.safe_load(f) or {}
            else:
                rules_config = {}
            
            # Ensure groups exist
            if "groups" not in rules_config:
                rules_config["groups"] = []
            
            # Find or create group
            group = None
            for g in rules_config["groups"]:
                if g.get("name") == group_name:
                    group = g
                    break
            
            if not group:
                group = {
                    "name": group_name,
                    "interval": "30s",
                    "rules": []
                }
                rules_config["groups"].append(group)
            
            # Add rule
            group["rules"].append(rule)
            
            # Save rules
            rule_file.parent.mkdir(parents=True, exist_ok=True)
            with open(rule_file, 'w') as f:
                yaml.dump(rules_config, f, default_flow_style=False)
            
            # Reload configuration
            return await self.reload_config()
            
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False
    
    @trace_function()
    async def get_metric_statistics(
        self,
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get statistics about metrics"""
        try:
            # Get TSDB status
            tsdb_status = await self.get_tsdb_status()
            if not tsdb_status:
                return {}
            
            stats = {
                "global": {
                    "total_series": tsdb_status.headStats.get("numSeries", 0),
                    "total_samples": tsdb_status.headStats.get("numSamples", 0),
                    "total_chunks": tsdb_status.headStats.get("chunks", 0),
                    "chunk_size_bytes": tsdb_status.headStats.get("chunkSize", 0)
                },
                "top_metrics_by_series": [],
                "top_labels_by_memory": []
            }
            
            # Add top metrics by series count
            for metric_info in tsdb_status.seriesCountByMetricName[:10]:
                stats["top_metrics_by_series"].append({
                    "metric": metric_info["name"],
                    "series_count": metric_info["value"]
                })
            
            # Add top labels by memory usage
            for label_info in tsdb_status.memoryInBytesByLabelName[:10]:
                stats["top_labels_by_memory"].append({
                    "label": label_info["name"],
                    "memory_bytes": label_info["value"]
                })
            
            # Get specific metric stats if requested
            if metric_name:
                # Query metric info
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                
                # Get series count
                series_query = f'count(count by (__name__)({{{metric_name}!=~""}}))'
                series_result = await self.query(series_query)
                
                # Get sample rate
                rate_query = f'sum(rate({metric_name}[5m]))'
                rate_result = await self.query(rate_query)
                
                stats["metric_specific"] = {
                    "metric_name": metric_name,
                    "series_count": self._extract_scalar_value(series_result),
                    "sample_rate_per_second": self._extract_scalar_value(rate_result),
                    "time_range_hours": hours
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get metric statistics: {e}")
            return {}
    
    def _extract_scalar_value(self, response: MetricResponse) -> float:
        """Extract scalar value from metric response"""
        try:
            if response.result_type == ResultType.VECTOR and response.result:
                return float(response.result[0].value.value)
            elif response.result_type == ResultType.SCALAR and response.result:
                return float(response.result.value)
            return 0.0
        except:
            return 0.0
    
    @trace_function()
    async def test_query(self, query: str) -> Dict[str, Any]:
        """Test a PromQL query and return information about it"""
        try:
            # Try to execute the query
            result = await self.query(query)
            
            # Analyze result
            analysis = {
                "valid": True,
                "result_type": result.result_type.value if result.result_type else "unknown",
                "sample_count": 0,
                "series_count": 0,
                "labels": set()
            }
            
            if result.result_type == ResultType.VECTOR:
                analysis["series_count"] = len(result.result)
                analysis["sample_count"] = len(result.result)
                
                # Collect all labels
                for series in result.result:
                    for label in series.metric.keys():
                        analysis["labels"].add(label)
                        
            elif result.result_type == ResultType.MATRIX:
                analysis["series_count"] = len(result.result)
                
                # Count total samples
                for series in result.result:
                    analysis["sample_count"] += len(series.values)
                
                # Collect all labels
                for series in result.result:
                    for label in series.metric.keys():
                        analysis["labels"].add(label)
            
            analysis["labels"] = sorted(list(analysis["labels"]))
            
            return analysis
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
prometheus_service = PrometheusService()