# monitoring-service/src/services/otel_collector_service.py
"""OpenTelemetry Collector service management"""

import asyncio
import httpx
import docker
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

from core.config import settings
from core.opentelemetry_setup import trace_function

logger = logging.getLogger(__name__)

class OTELCollectorService:
    """Service for managing OpenTelemetry Collector"""
    
    def __init__(self):
        self.base_url = f"http://otel-collector:13133"  # Health check port
        self.metrics_url = f"http://otel-collector:8888"  # Prometheus metrics
        self.zpages_url = f"http://otel-collector:55679"  # zPages debugging
        self.client = httpx.AsyncClient(timeout=30.0)
        self.docker_client = docker.from_env()
        self.container_name = "otel-collector"
        self.config_path = settings.CONFIG_PATH / "otel-collector" / "otel-collector-config.yaml"
        
    async def start(self):
        """Start OpenTelemetry Collector container if not already running"""
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
            logger.error(f"Failed to start OpenTelemetry Collector: {e}")
            raise
            
    async def stop(self):
        """Stop OpenTelemetry Collector gracefully"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info(f"Stopping {self.container_name} container")
            container.stop(timeout=30)
        except docker.errors.NotFound:
            logger.warning(f"{self.container_name} container not found")
        except Exception as e:
            logger.error(f"Failed to stop OpenTelemetry Collector: {e}")
            
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for OpenTelemetry Collector to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("OpenTelemetry Collector is ready")
                    return
            except:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("OpenTelemetry Collector failed to become ready")
        
    @trace_function()
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenTelemetry Collector health"""
        try:
            # Check health endpoint
            health_response = await self.client.get(f"{self.base_url}/health")
            healthy = health_response.status_code == 200
            
            # Get metrics
            metrics_healthy = False
            try:
                metrics_response = await self.client.get(f"{self.metrics_url}/metrics")
                metrics_healthy = metrics_response.status_code == 200
            except:
                pass
            
            # Get pipeline status
            pipeline_status = await self._get_pipeline_status()
            
            return {
                "status": "healthy" if healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "endpoint": self.base_url,
                    "health_check_endpoint": f"{self.base_url}/health",
                    "metrics_available": metrics_healthy,
                    "pipelines": pipeline_status
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "details": {"endpoint": self.base_url}
            }
    
    async def _get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of OTEL pipelines"""
        try:
            # Parse metrics to get pipeline info
            response = await self.client.get(f"{self.metrics_url}/metrics")
            if response.status_code != 200:
                return {}
            
            metrics_text = response.text
            
            # Extract pipeline metrics
            pipelines = {
                "traces": {
                    "received": self._parse_metric(metrics_text, "otelcol_receiver_accepted_spans"),
                    "processed": self._parse_metric(metrics_text, "otelcol_processor_batch_batch_send_size_sum"),
                    "exported": self._parse_metric(metrics_text, "otelcol_exporter_sent_spans"),
                    "dropped": self._parse_metric(metrics_text, "otelcol_processor_dropped_spans")
                },
                "metrics": {
                    "received": self._parse_metric(metrics_text, "otelcol_receiver_accepted_metric_points"),
                    "processed": self._parse_metric(metrics_text, "otelcol_processor_batch_metric_point_count"),
                    "exported": self._parse_metric(metrics_text, "otelcol_exporter_sent_metric_points"),
                    "dropped": self._parse_metric(metrics_text, "otelcol_processor_dropped_metric_points")
                },
                "logs": {
                    "received": self._parse_metric(metrics_text, "otelcol_receiver_accepted_log_records"),
                    "processed": self._parse_metric(metrics_text, "otelcol_processor_batch_log_record_count"),
                    "exported": self._parse_metric(metrics_text, "otelcol_exporter_sent_log_records"),
                    "dropped": self._parse_metric(metrics_text, "otelcol_processor_dropped_log_records")
                }
            }
            
            return pipelines
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {}
    
    def _parse_metric(self, metrics_text: str, metric_name: str) -> float:
        """Parse a metric value from Prometheus format text"""
        try:
            total = 0.0
            for line in metrics_text.split('\n'):
                if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
                    # Extract value after the metric
                    parts = line.split()
                    if len(parts) >= 2:
                        total += float(parts[-1])
            return total
        except:
            return 0.0
    
    @trace_function()
    async def get_config(self) -> Optional[Dict[str, Any]]:
        """Get current OpenTelemetry Collector configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            return None
            
        except Exception as e:
            logger.error(f"Failed to get OTEL Collector config: {e}")
            return None
    
    @trace_function()
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """Update OpenTelemetry Collector configuration"""
        try:
            # Validate configuration
            if not self._validate_config(config):
                logger.error("Invalid OTEL Collector configuration")
                return False
            
            # Write new configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Restart collector to apply changes
            await self.restart()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update OTEL Collector config: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate OpenTelemetry Collector configuration"""
        try:
            # Check required sections
            required_sections = ["receivers", "processors", "exporters", "service"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Check service pipelines
            if "pipelines" not in config["service"]:
                logger.error("Missing service.pipelines section")
                return False
            
            # Validate each pipeline
            for pipeline_name, pipeline_config in config["service"]["pipelines"].items():
                # Check pipeline has receivers, processors, and exporters
                if "receivers" not in pipeline_config:
                    logger.error(f"Pipeline {pipeline_name} missing receivers")
                    return False
                
                if "exporters" not in pipeline_config:
                    logger.error(f"Pipeline {pipeline_name} missing exporters")
                    return False
                
                # Validate referenced components exist
                for receiver in pipeline_config.get("receivers", []):
                    if receiver not in config["receivers"]:
                        logger.error(f"Receiver {receiver} not defined")
                        return False
                
                for processor in pipeline_config.get("processors", []):
                    if processor not in config["processors"]:
                        logger.error(f"Processor {processor} not defined")
                        return False
                
                for exporter in pipeline_config.get("exporters", []):
                    if exporter not in config["exporters"]:
                        logger.error(f"Exporter {exporter} not defined")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False
    
    async def restart(self):
        """Restart OpenTelemetry Collector"""
        try:
            await self.stop()
            await asyncio.sleep(2)
            await self.start()
            logger.info("OpenTelemetry Collector restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart OTEL Collector: {e}")
            raise
    
    @trace_function()
    async def get_receivers_info(self) -> Dict[str, Any]:
        """Get information about configured receivers"""
        try:
            config = await self.get_config()
            if not config:
                return {}
            
            receivers = config.get("receivers", {})
            receivers_info = {}
            
            for name, receiver_config in receivers.items():
                info = {
                    "type": self._get_receiver_type(name, receiver_config),
                    "config": receiver_config
                }
                
                # Add specific info based on receiver type
                if "otlp" in name:
                    protocols = receiver_config.get("protocols", {})
                    info["endpoints"] = {
                        "grpc": protocols.get("grpc", {}).get("endpoint", ""),
                        "http": protocols.get("http", {}).get("endpoint", "")
                    }
                elif "prometheus" in name:
                    info["scrape_configs"] = len(receiver_config.get("config", {}).get("scrape_configs", []))
                
                receivers_info[name] = info
            
            return receivers_info
            
        except Exception as e:
            logger.error(f"Failed to get receivers info: {e}")
            return {}
    
    def _get_receiver_type(self, name: str, config: Dict[str, Any]) -> str:
        """Determine receiver type from name and config"""
        if "otlp" in name:
            return "otlp"
        elif "prometheus" in name:
            return "prometheus"
        elif "jaeger" in name:
            return "jaeger"
        elif "zipkin" in name:
            return "zipkin"
        else:
            return "unknown"
    
    @trace_function()
    async def get_processors_info(self) -> Dict[str, Any]:
        """Get information about configured processors"""
        try:
            config = await self.get_config()
            if not config:
                return {}
            
            processors = config.get("processors", {})
            processors_info = {}
            
            for name, processor_config in processors.items():
                info = {
                    "type": name.split("/")[0],  # Extract base type
                    "config": processor_config
                }
                
                # Add specific info based on processor type
                if "batch" in name:
                    info["timeout"] = processor_config.get("timeout", "")
                    info["send_batch_size"] = processor_config.get("send_batch_size", 0)
                elif "memory_limiter" in name:
                    info["limit_mib"] = processor_config.get("limit_mib", 0)
                    info["spike_limit_mib"] = processor_config.get("spike_limit_mib", 0)
                elif "resource" in name:
                    info["attributes"] = len(processor_config.get("attributes", []))
                
                processors_info[name] = info
            
            return processors_info
            
        except Exception as e:
            logger.error(f"Failed to get processors info: {e}")
            return {}
    
    @trace_function()
    async def get_exporters_info(self) -> Dict[str, Any]:
        """Get information about configured exporters"""
        try:
            config = await self.get_config()
            if not config:
                return {}
            
            exporters = config.get("exporters", {})
            exporters_info = {}
            
            for name, exporter_config in exporters.items():
                info = {
                    "type": name.split("/")[0],  # Extract base type
                    "config": exporter_config
                }
                
                # Add specific info based on exporter type
                if "prometheusremotewrite" in name:
                    info["endpoint"] = exporter_config.get("endpoint", "")
                elif "otlp" in name:
                    info["endpoint"] = exporter_config.get("endpoint", "")
                    info["insecure"] = exporter_config.get("tls", {}).get("insecure", False)
                elif "loki" in name:
                    info["endpoint"] = exporter_config.get("endpoint", "")
                
                exporters_info[name] = info
            
            return exporters_info
            
        except Exception as e:
            logger.error(f"Failed to get exporters info: {e}")
            return {}
    
    @trace_function()
    async def get_pipelines_info(self) -> Dict[str, Any]:
        """Get information about configured pipelines"""
        try:
            config = await self.get_config()
            if not config:
                return {}
            
            pipelines = config.get("service", {}).get("pipelines", {})
            pipelines_info = {}
            
            for name, pipeline_config in pipelines.items():
                pipelines_info[name] = {
                    "receivers": pipeline_config.get("receivers", []),
                    "processors": pipeline_config.get("processors", []),
                    "exporters": pipeline_config.get("exporters", []),
                    "receiver_count": len(pipeline_config.get("receivers", [])),
                    "processor_count": len(pipeline_config.get("processors", [])),
                    "exporter_count": len(pipeline_config.get("exporters", []))
                }
            
            return pipelines_info
            
        except Exception as e:
            logger.error(f"Failed to get pipelines info: {e}")
            return {}
    
    @trace_function()
    async def get_zpages_info(self) -> Dict[str, Any]:
        """Get debugging information from zPages"""
        try:
            zpages_info = {}
            
            # Get trace information
            try:
                trace_response = await self.client.get(f"{self.zpages_url}/debug/tracez")
                zpages_info["traces_available"] = trace_response.status_code == 200
            except:
                zpages_info["traces_available"] = False
            
            # Get pipeline information
            try:
                pipeline_response = await self.client.get(f"{self.zpages_url}/debug/pipelinez")
                zpages_info["pipelines_available"] = pipeline_response.status_code == 200
            except:
                zpages_info["pipelines_available"] = False
            
            # Get service information
            try:
                service_response = await self.client.get(f"{self.zpages_url}/debug/servicez")
                zpages_info["service_available"] = service_response.status_code == 200
            except:
                zpages_info["service_available"] = False
            
            zpages_info["base_url"] = self.zpages_url
            
            return zpages_info
            
        except Exception as e:
            logger.error(f"Failed to get zPages info: {e}")
            return {}
    
    @trace_function()
    async def get_metrics(self) -> Dict[str, Any]:
        """Get OpenTelemetry Collector metrics"""
        try:
            response = await self.client.get(f"{self.metrics_url}/metrics")
            if response.status_code != 200:
                return {}
            
            metrics_text = response.text
            
            # Extract key metrics
            metrics = {
                "process": {
                    "uptime": self._parse_metric(metrics_text, "process_uptime_seconds"),
                    "memory_rss": self._parse_metric(metrics_text, "process_resident_memory_bytes"),
                    "cpu_seconds": self._parse_metric(metrics_text, "process_cpu_seconds_total")
                },
                "runtime": {
                    "goroutines": self._parse_metric(metrics_text, "go_goroutines"),
                    "gc_duration": self._parse_metric(metrics_text, "go_gc_duration_seconds_sum"),
                    "memory_allocated": self._parse_metric(metrics_text, "go_memstats_alloc_bytes")
                },
                "receivers": {
                    "accepted_spans": self._parse_metric(metrics_text, "otelcol_receiver_accepted_spans"),
                    "refused_spans": self._parse_metric(metrics_text, "otelcol_receiver_refused_spans"),
                    "accepted_metric_points": self._parse_metric(metrics_text, "otelcol_receiver_accepted_metric_points"),
                    "refused_metric_points": self._parse_metric(metrics_text, "otelcol_receiver_refused_metric_points"),
                    "accepted_log_records": self._parse_metric(metrics_text, "otelcol_receiver_accepted_log_records"),
                    "refused_log_records": self._parse_metric(metrics_text, "otelcol_receiver_refused_log_records")
                },
                "processors": {
                    "batch_timeout_trigger": self._parse_metric(metrics_text, "otelcol_processor_batch_timeout_trigger_send"),
                    "batch_size_trigger": self._parse_metric(metrics_text, "otelcol_processor_batch_size_trigger_send"),
                    "dropped_spans": self._parse_metric(metrics_text, "otelcol_processor_dropped_spans"),
                    "dropped_metric_points": self._parse_metric(metrics_text, "otelcol_processor_dropped_metric_points"),
                    "dropped_log_records": self._parse_metric(metrics_text, "otelcol_processor_dropped_log_records")
                },
                "exporters": {
                    "sent_spans": self._parse_metric(metrics_text, "otelcol_exporter_sent_spans"),
                    "failed_spans": self._parse_metric(metrics_text, "otelcol_exporter_send_failed_spans"),
                    "sent_metric_points": self._parse_metric(metrics_text, "otelcol_exporter_sent_metric_points"),
                    "failed_metric_points": self._parse_metric(metrics_text, "otelcol_exporter_send_failed_metric_points"),
                    "sent_log_records": self._parse_metric(metrics_text, "otelcol_exporter_sent_log_records"),
                    "failed_log_records": self._parse_metric(metrics_text, "otelcol_exporter_send_failed_log_records")
                },
                "queue": {
                    "size": self._parse_metric(metrics_text, "otelcol_exporter_queue_size"),
                    "capacity": self._parse_metric(metrics_text, "otelcol_exporter_queue_capacity")
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get OTEL Collector metrics: {e}")
            return {}
    
    @trace_function()
    async def add_receiver(
        self,
        name: str,
        receiver_type: str,
        config: Dict[str, Any],
        pipelines: List[str]
    ) -> bool:
        """Add a new receiver to the configuration"""
        try:
            current_config = await self.get_config()
            if not current_config:
                return False
            
            # Add receiver
            if "receivers" not in current_config:
                current_config["receivers"] = {}
            
            current_config["receivers"][name] = config
            
            # Add to pipelines
            for pipeline in pipelines:
                if pipeline in current_config["service"]["pipelines"]:
                    receivers = current_config["service"]["pipelines"][pipeline].get("receivers", [])
                    if name not in receivers:
                        receivers.append(name)
                        current_config["service"]["pipelines"][pipeline]["receivers"] = receivers
            
            # Update configuration
            return await self.update_config(current_config)
            
        except Exception as e:
            logger.error(f"Failed to add receiver: {e}")
            return False
    
    @trace_function()
    async def remove_receiver(self, name: str) -> bool:
        """Remove a receiver from the configuration"""
        try:
            current_config = await self.get_config()
            if not current_config:
                return False
            
            # Remove receiver
            if name in current_config.get("receivers", {}):
                del current_config["receivers"][name]
            
            # Remove from pipelines
            for pipeline in current_config["service"]["pipelines"].values():
                if "receivers" in pipeline and name in pipeline["receivers"]:
                    pipeline["receivers"].remove(name)
            
            # Update configuration
            return await self.update_config(current_config)
            
        except Exception as e:
            logger.error(f"Failed to remove receiver: {e}")
            return False
    
    @trace_function()
    async def test_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a configuration without applying it"""
        try:
            # Validate configuration structure
            is_valid = self._validate_config(config)
            
            if not is_valid:
                return {
                    "valid": False,
                    "error": "Configuration validation failed",
                    "details": "Check logs for specific validation errors"
                }
            
            # Additional checks
            warnings = []
            
            # Check for debug exporter in production
            if settings.ENVIRONMENT == "production":
                if "debug" in config.get("exporters", {}):
                    warnings.append("Debug exporter is enabled in production")
            
            # Check for missing recommended processors
            processors = config.get("processors", {})
            if not any("memory_limiter" in p for p in processors):
                warnings.append("Memory limiter processor is recommended")
            
            if not any("batch" in p for p in processors):
                warnings.append("Batch processor is recommended for performance")
            
            return {
                "valid": True,
                "warnings": warnings,
                "receivers": len(config.get("receivers", {})),
                "processors": len(config.get("processors", {})),
                "exporters": len(config.get("exporters", {})),
                "pipelines": len(config.get("service", {}).get("pipelines", {}))
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Create global instance
otel_collector_service = OTELCollectorService()