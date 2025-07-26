"""
Metrics Service gRPC Implementation
Production-grade service for system metrics collection
"""
import asyncio
import grpc
import time
import uuid
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, List, Optional
from google.protobuf.timestamp_pb2 import Timestamp

# Add genx_platform to path
current_file = os.path.abspath(__file__)
service_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(service_dir)
metrics_root = os.path.dirname(src_dir)
microservices_dir = os.path.dirname(metrics_root)
genx_components = os.path.dirname(microservices_dir)
genx_platform = os.path.dirname(genx_components)
sys.path.insert(0, genx_platform)

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    metrics_service_pb2,
    metrics_service_pb2_grpc,
)

# Import using absolute paths
from genx_components.microservices.metrics.src.collectors.metrics_collector import MetricsCollector
from genx_components.microservices.metrics.src.utils.cache import MetricsCache
from genx_components.microservices.metrics.src.utils.logger import setup_logging

logger = setup_logging(__name__)


class MetricsService(metrics_service_pb2_grpc.MetricsServiceServicer):
    """
    Production-grade gRPC service for system metrics
    """
    
    def __init__(self, config=None, telemetry=None):
        self.config = config
        self.telemetry = telemetry
        self.collector = MetricsCollector()
        
        # Get cache TTL from config or environment
        cache_ttl = 30
        if config and hasattr(config, 'cache_ttl_seconds'):
            cache_ttl = config.cache_ttl_seconds
        else:
            cache_ttl = int(os.environ.get('CACHE_TTL_SECONDS', '30'))
            
        self.cache = MetricsCache(ttl_seconds=cache_ttl)
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service"""
        if self._initialized:
            return
            
        await self.collector.initialize()
        self._initialized = True
        
        # Start background metrics collection
        asyncio.create_task(self._background_collector())
        
        logger.info("Metrics service initialized")
    
    async def _background_collector(self):
        """Background task to periodically collect metrics"""
        while True:
            try:
                # Collect metrics every 30 seconds
                metrics = await self.collector.collect_all()
                await self.cache.set("latest", metrics)
                logger.debug("Background metrics collection completed")
            except Exception as e:
                logger.error(f"Background collection failed: {e}")
            
            await asyncio.sleep(30)
    
    async def GetSystemMetrics(
        self,
        request: metrics_service_pb2.GetSystemMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetSystemMetricsResponse:
        """Get current system metrics"""
        
        # Use telemetry if available
        if self.telemetry:
            with self.telemetry.trace_operation("GetSystemMetrics", {
                "force_refresh": request.force_refresh,
                "metric_types_count": len(request.metric_types)
            }) as span:
                return await self._get_system_metrics_impl(request, context, span)
        else:
            return await self._get_system_metrics_impl(request, context, None)
    
    async def _get_system_metrics_impl(self, request, context, span=None):
        """Implementation of GetSystemMetrics"""
        try:
            # Create response metadata
            response_metadata = common_pb2.ResponseMetadata(
                request_id=request.metadata.request_id,
                service_name="metrics-service",
                service_version="1.0.0"
            )
            start_time = time.time()
            
            # Determine which metrics to collect
            metric_types = request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
            
            # Check cache if not forcing refresh
            source = "fresh"
            metrics_data = None
            
            if not request.force_refresh:
                cached = await self.cache.get("latest")
                if cached:
                    metrics_data = cached
                    source = "cache"
            
            # Collect fresh metrics if needed
            if metrics_data is None:
                metrics_data = await self.collector.collect_metrics(metric_types)
                await self.cache.set("latest", metrics_data)
            
            # Convert to protobuf
            metrics_proto = self._convert_to_proto(metrics_data)
            
            # Set response metadata timing
            response_metadata.duration_ms = int((time.time() - start_time) * 1000)
            response_metadata.timestamp.GetCurrentTime()
            
            return metrics_service_pb2.GetSystemMetricsResponse(
                metadata=response_metadata,
                metrics=metrics_proto,
                source=source
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}", exc_info=True)
            
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to collect metrics: {str(e)}"
            )
    
    async def StreamSystemMetrics(
        self,
        request: metrics_service_pb2.StreamSystemMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[metrics_service_pb2.SystemMetricsUpdate]:
        """Stream real-time metrics updates"""
        
        stream_id = str(uuid.uuid4())
        
        # Validate interval
        interval = max(1, min(300, request.interval_seconds or 5))
        max_duration = request.max_duration_seconds or 0
        
        logger.info(
            f"Starting metrics stream {stream_id}",
            extra={
                "interval": interval,
                "max_duration": max_duration,
                "metric_types": len(request.metric_types)
            }
        )
        
        try:
            start_time = time.time()
            last_metrics = None
            
            # Send initial update
            metrics_data = await self.collector.collect_metrics(
                request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
            )
            metrics_proto = self._convert_to_proto(metrics_data)
            
            yield metrics_service_pb2.SystemMetricsUpdate(
                type=metrics_service_pb2.SystemMetricsUpdate.UPDATE_TYPE_INITIAL,
                metrics=metrics_proto,
                timestamp=Timestamp(seconds=int(time.time()))
            )
            
            last_metrics = metrics_data
            
            # Stream periodic updates
            while not context.cancelled():
                await asyncio.sleep(interval)
                
                # Check max duration
                if max_duration > 0 and (time.time() - start_time) > max_duration:
                    logger.info(f"Stream {stream_id} reached max duration")
                    break
                
                # Collect new metrics
                metrics_data = await self.collector.collect_metrics(
                    request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
                )
                
                # Check if should send update
                should_send = True
                if request.include_deltas_only and last_metrics:
                    # Check if metrics changed significantly
                    should_send = self._has_significant_change(
                        last_metrics, metrics_data, request.change_threshold_percent
                    )
                
                if should_send:
                    metrics_proto = self._convert_to_proto(metrics_data)
                    
                    # Check for alerts if requested
                    alerts = []
                    if request.include_alerts:
                        alerts = self._check_alerts(metrics_data)
                    
                    update = metrics_service_pb2.SystemMetricsUpdate(
                        type=metrics_service_pb2.SystemMetricsUpdate.UPDATE_TYPE_PERIODIC,
                        metrics=metrics_proto,
                        timestamp=Timestamp(seconds=int(time.time()))
                    )
                    
                    if alerts:
                        update.alerts.extend(alerts)
                    
                    yield update
                    last_metrics = metrics_data
                    
        except asyncio.CancelledError:
            logger.info(f"Stream {stream_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Stream {stream_id} error: {e}", exc_info=True)
            
            # Send error update
            error_detail = common_pb2.ErrorDetail(
                code="INTERNAL",
                message=str(e)
            )
            
            yield metrics_service_pb2.SystemMetricsUpdate(
                type=metrics_service_pb2.SystemMetricsUpdate.UPDATE_TYPE_PERIODIC,
                error=error_detail,
                timestamp=Timestamp(seconds=int(time.time()))
            )
        finally:
            logger.info(f"Stream {stream_id} ended")
    
    async def GetMetricsHistory(
        self,
        request: metrics_service_pb2.GetMetricsHistoryRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetMetricsHistoryResponse:
        """Get historical metrics - for now returns empty as we focus on real-time"""
        
        # This is a placeholder - would integrate with time-series DB in production
        logger.info("Historical metrics requested - returning empty dataset")
        
        response_metadata = common_pb2.ResponseMetadata(
            request_id=request.metadata.request_id,
            service_name="metrics-service",
            service_version="1.0.0"
        )
        response_metadata.timestamp.GetCurrentTime()
        
        # Return empty history for now
        history_metadata = metrics_service_pb2.MetricsHistoryMetadata(
            count=0,
            resolution_seconds=request.resolution_seconds or 60,
            aggregation_type="none",
            completeness_percent=0.0
        )
        history_metadata.start_time.CopyFrom(request.start_time)
        history_metadata.end_time.CopyFrom(request.end_time)
        
        return metrics_service_pb2.GetMetricsHistoryResponse(
            metadata=response_metadata,
            data_points=[],
            history_metadata=history_metadata
        )
    
    async def RefreshMetrics(
        self,
        request: metrics_service_pb2.RefreshMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.RefreshMetricsResponse:
        """Force refresh metrics collection"""
        
        try:
            # Force fresh collection
            metrics_data = await self.collector.collect_all()
            await self.cache.set("latest", metrics_data)
            
            # Convert to proto
            metrics_proto = self._convert_to_proto(metrics_data)
            
            response_metadata = common_pb2.ResponseMetadata(
                request_id=request.metadata.request_id,
                service_name="metrics-service",
                service_version="1.0.0"
            )
            response_metadata.timestamp.GetCurrentTime()
            
            return metrics_service_pb2.RefreshMetricsResponse(
                metadata=response_metadata,
                success=True,
                metrics=metrics_proto
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh metrics: {e}", exc_info=True)
            
            error_detail = common_pb2.ErrorDetail(
                code="INTERNAL",
                message=f"Failed to refresh metrics: {str(e)}"
            )
            
            response_metadata = common_pb2.ResponseMetadata(
                request_id=request.metadata.request_id,
                service_name="metrics-service",
                service_version="1.0.0"
            )
            
            return metrics_service_pb2.RefreshMetricsResponse(
                metadata=response_metadata,
                success=False,
                error=error_detail
            )
    
    async def GetResourceSummary(
        self,
        request: metrics_service_pb2.GetResourceSummaryRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetResourceSummaryResponse:
        """Get resource summary for dashboard"""
        
        try:
            # Get latest metrics
            metrics_data = await self.cache.get("latest")
            if not metrics_data:
                metrics_data = await self.collector.collect_all()
            
            # Create summaries
            cpu_status = self._create_resource_status("cpu", metrics_data.get("cpu", {}))
            memory_status = self._create_resource_status("memory", metrics_data.get("memory", {}))
            disk_status = self._create_resource_status("disk", metrics_data.get("disk", {}))
            
            # GPU status (may be None)
            gpu_status = None
            if "gpu" in metrics_data and metrics_data["gpu"]:
                # Average across all GPUs
                gpu_usage = sum(g.get("usage_percent", 0) for g in metrics_data["gpu"]) / len(metrics_data["gpu"])
                gpu_status = self._create_resource_status("gpu", {"usage_percent": gpu_usage})
            
            # Overall health
            overall_health = self._determine_overall_health([cpu_status, memory_status, disk_status, gpu_status])
            
            response_metadata = common_pb2.ResponseMetadata(
                request_id=request.metadata.request_id,
                service_name="metrics-service",
                service_version="1.0.0"
            )
            response_metadata.timestamp.GetCurrentTime()
            
            response = metrics_service_pb2.GetResourceSummaryResponse(
                metadata=response_metadata,
                cpu_status=cpu_status,
                memory_status=memory_status,
                disk_status=disk_status,
                overall_health=overall_health
            )
            
            if gpu_status:
                response.gpu_status.CopyFrom(gpu_status)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get resource summary: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to get resource summary: {str(e)}"
            )
    
    def _convert_to_proto(self, metrics_data: Dict[str, Any]) -> metrics_service_pb2.SystemMetrics:
        """Convert collected metrics to protobuf format"""
        metrics = metrics_service_pb2.SystemMetrics()
        
        # CPU metrics
        if "cpu" in metrics_data:
            cpu_data = metrics_data["cpu"]
            metrics.cpu.usage_percent = cpu_data.get("usage_percent", 0.0)
            metrics.cpu.count = cpu_data.get("count", 0)
            metrics.cpu.count_logical = cpu_data.get("count_logical", 0)
            
            if "per_core_percent" in cpu_data:
                metrics.cpu.per_core_percent.extend(cpu_data["per_core_percent"])
            
            if "frequency" in cpu_data:
                freq = cpu_data["frequency"]
                metrics.cpu.frequency.current_mhz = freq.get("current", 0.0)
                metrics.cpu.frequency.min_mhz = freq.get("min", 0.0)
                metrics.cpu.frequency.max_mhz = freq.get("max", 0.0)
            
            if "load_average" in cpu_data:
                load = cpu_data["load_average"]
                metrics.cpu.load_average.one_minute = load.get("1m", 0.0)
                metrics.cpu.load_average.five_minutes = load.get("5m", 0.0)
                metrics.cpu.load_average.fifteen_minutes = load.get("15m", 0.0)
        
        # Memory metrics
        if "memory" in metrics_data:
            mem_data = metrics_data["memory"]
            metrics.memory.total_bytes = mem_data.get("total", 0)
            metrics.memory.available_bytes = mem_data.get("available", 0)
            metrics.memory.used_bytes = mem_data.get("used", 0)
            metrics.memory.free_bytes = mem_data.get("free", 0)
            metrics.memory.percent = mem_data.get("percent", 0.0)
            
            if "swap" in mem_data:
                swap = mem_data["swap"]
                metrics.memory.swap.total_bytes = swap.get("total", 0)
                metrics.memory.swap.used_bytes = swap.get("used", 0)
                metrics.memory.swap.free_bytes = swap.get("free", 0)
                metrics.memory.swap.percent = swap.get("percent", 0.0)
        
        # GPU metrics
        if "gpu" in metrics_data and metrics_data["gpu"]:
            for gpu_data in metrics_data["gpu"]:
                gpu = metrics.gpu.add()
                gpu.id = gpu_data.get("id", 0)
                gpu.name = gpu_data.get("name", "Unknown")
                gpu.load_percent = gpu_data.get("usage_percent", 0.0)
                gpu.temperature_celsius = gpu_data.get("temperature", 0.0)
                gpu.uuid = gpu_data.get("uuid", "")
                gpu.driver_version = gpu_data.get("driver_version", "")
                
                if "memory" in gpu_data:
                    mem = gpu_data["memory"]
                    gpu.memory.total_bytes = mem.get("total", 0)
                    gpu.memory.used_bytes = mem.get("used", 0)
                    gpu.memory.free_bytes = mem.get("free", 0)
                    gpu.memory.percent = mem.get("percent", 0.0)
        
        # Disk metrics
        if "disk" in metrics_data:
            disk_data = metrics_data["disk"]
            if "usage" in disk_data:
                usage = disk_data["usage"]
                metrics.disk.usage.total_bytes = usage.get("total", 0)
                metrics.disk.usage.used_bytes = usage.get("used", 0)
                metrics.disk.usage.free_bytes = usage.get("free", 0)
                metrics.disk.usage.percent = usage.get("percent", 0.0)
            
            # Model storage specific
            if "model_storage" in disk_data:
                ms = disk_data["model_storage"]
                metrics.disk.model_storage.path = ms.get("path", "/models")
                metrics.disk.model_storage.total_bytes = ms.get("total", 0)
                metrics.disk.model_storage.used_bytes = ms.get("used", 0)
                metrics.disk.model_storage.free_bytes = ms.get("free", 0)
                metrics.disk.model_storage.percent = ms.get("percent", 0.0)
        
        # Network metrics (simplified for now)
        if "network" in metrics_data:
            net_data = metrics_data["network"]
            if "io" in net_data:
                io = net_data["io"]
                metrics.network.io.bytes_sent = io.get("bytes_sent", 0)
                metrics.network.io.bytes_received = io.get("bytes_recv", 0)
                metrics.network.io.packets_sent = io.get("packets_sent", 0)
                metrics.network.io.packets_received = io.get("packets_recv", 0)
        
        # Set metadata
        metrics.collected_at.GetCurrentTime()
        
        return metrics
    
    def _has_significant_change(
        self, 
        old_metrics: Dict[str, Any], 
        new_metrics: Dict[str, Any], 
        threshold: float
    ) -> bool:
        """Check if metrics changed significantly"""
        if threshold <= 0:
            return True
        
        # Check CPU change
        old_cpu = old_metrics.get("cpu", {}).get("usage_percent", 0)
        new_cpu = new_metrics.get("cpu", {}).get("usage_percent", 0)
        if abs(new_cpu - old_cpu) > threshold:
            return True
        
        # Check memory change
        old_mem = old_metrics.get("memory", {}).get("percent", 0)
        new_mem = new_metrics.get("memory", {}).get("percent", 0)
        if abs(new_mem - old_mem) > threshold:
            return True
        
        return False
    
    def _check_alerts(self, metrics_data: Dict[str, Any]) -> List[metrics_service_pb2.ResourceAlert]:
        """Check for resource alerts"""
        alerts = []
        
        # CPU alert
        cpu_usage = metrics_data.get("cpu", {}).get("usage_percent", 0)
        if cpu_usage > 90:
            alert = metrics_service_pb2.ResourceAlert(
                id=f"cpu-high-{int(time.time())}",
                resource_type="cpu",
                severity=metrics_service_pb2.ResourceAlert.ALERT_SEVERITY_CRITICAL,
                message=f"CPU usage critical: {cpu_usage:.1f}%",
                current_value=cpu_usage,
                threshold=90.0
            )
            alert.triggered_at.GetCurrentTime()
            alerts.append(alert)
        elif cpu_usage > 80:
            alert = metrics_service_pb2.ResourceAlert(
                id=f"cpu-warn-{int(time.time())}",
                resource_type="cpu",
                severity=metrics_service_pb2.ResourceAlert.ALERT_SEVERITY_WARNING,
                message=f"CPU usage high: {cpu_usage:.1f}%",
                current_value=cpu_usage,
                threshold=80.0
            )
            alert.triggered_at.GetCurrentTime()
            alerts.append(alert)
        
        # Memory alert
        mem_usage = metrics_data.get("memory", {}).get("percent", 0)
        if mem_usage > 90:
            alert = metrics_service_pb2.ResourceAlert(
                id=f"memory-high-{int(time.time())}",
                resource_type="memory",
                severity=metrics_service_pb2.ResourceAlert.ALERT_SEVERITY_CRITICAL,
                message=f"Memory usage critical: {mem_usage:.1f}%",
                current_value=mem_usage,
                threshold=90.0
            )
            alert.triggered_at.GetCurrentTime()
            alerts.append(alert)
        
        # Disk alert
        disk_usage = metrics_data.get("disk", {}).get("usage", {}).get("percent", 0)
        if disk_usage > 85:
            alert = metrics_service_pb2.ResourceAlert(
                id=f"disk-high-{int(time.time())}",
                resource_type="disk",
                severity=metrics_service_pb2.ResourceAlert.ALERT_SEVERITY_WARNING,
                message=f"Disk usage high: {disk_usage:.1f}%",
                current_value=disk_usage,
                threshold=85.0
            )
            alert.triggered_at.GetCurrentTime()
            alerts.append(alert)
        
        return alerts
    
    def _create_resource_status(
        self, 
        resource_type: str, 
        resource_data: Dict[str, Any]
    ) -> metrics_service_pb2.ResourceStatus:
        """Create resource status summary"""
        usage_percent = 0.0
        
        if resource_type == "cpu":
            usage_percent = resource_data.get("usage_percent", 0.0)
        elif resource_type == "memory":
            usage_percent = resource_data.get("percent", 0.0)
        elif resource_type == "disk":
            usage_percent = resource_data.get("usage", {}).get("percent", 0.0)
        elif resource_type == "gpu":
            usage_percent = resource_data.get("usage_percent", 0.0)
        
        # Determine health status
        if usage_percent > 90:
            status = metrics_service_pb2.ResourceStatus.HEALTH_STATUS_CRITICAL
            message = f"{resource_type.upper()} usage critical: {usage_percent:.1f}%"
        elif usage_percent > 80:
            status = metrics_service_pb2.ResourceStatus.HEALTH_STATUS_WARNING
            message = f"{resource_type.upper()} usage high: {usage_percent:.1f}%"
        else:
            status = metrics_service_pb2.ResourceStatus.HEALTH_STATUS_HEALTHY
            message = f"{resource_type.upper()} usage normal: {usage_percent:.1f}%"
        
        return metrics_service_pb2.ResourceStatus(
            status=status,
            usage_percent=usage_percent,
            message=message,
            trend=metrics_service_pb2.ResourceStatus.TREND_STABLE
        )
    
    def _determine_overall_health(
        self, 
        statuses: List[Optional[metrics_service_pb2.ResourceStatus]]
    ) -> metrics_service_pb2.SystemHealth:
        """Determine overall system health from resource statuses"""
        valid_statuses = [s for s in statuses if s is not None]
        
        if not valid_statuses:
            return metrics_service_pb2.SYSTEM_HEALTH_UNKNOWN
        
        # If any critical, system is critical
        if any(s.status == metrics_service_pb2.ResourceStatus.HEALTH_STATUS_CRITICAL for s in valid_statuses):
            return metrics_service_pb2.SYSTEM_HEALTH_CRITICAL
        
        # If any warning, system is degraded
        if any(s.status == metrics_service_pb2.ResourceStatus.HEALTH_STATUS_WARNING for s in valid_statuses):
            return metrics_service_pb2.SYSTEM_HEALTH_DEGRADED
        
        return metrics_service_pb2.SYSTEM_HEALTH_HEALTHY