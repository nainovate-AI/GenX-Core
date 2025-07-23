# services/proxy_service/src/grpc_services/metrics_service.py
"""
gRPC Metrics Service Implementation
Collects and serves system metrics with caching and history
"""
import asyncio
import grpc
from datetime import datetime, timedelta
from typing import AsyncIterator, List, Dict, Any
import psutil
import GPUtil
import structlog
from google.protobuf.timestamp_pb2 import Timestamp
from opentelemetry import trace

from ..generated import metrics_service_pb2
from ..generated import metrics_service_pb2_grpc
from ..collectors.metrics_collector import MetricsCollector
from ..data_access.cache_manager import get_cache_manager

logger = structlog.get_logger(__name__)


class MetricsService(metrics_service_pb2_grpc.MetricsServiceServicer):
    """
    gRPC service for system metrics
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.cache_manager = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service"""
        if self._initialized:
            return
            
        self.cache_manager = await get_cache_manager()
        await self.collector.initialize(self.cache_manager)
        self._initialized = True
        logger.info("Metrics service initialized")
    
    async def GetSystemMetrics(
        self,
        request: metrics_service_pb2.GetSystemMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetSystemMetricsResponse:
        """Get current system metrics"""
        
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()
        
        logger.info("Getting system metrics", 
                   request_id=request.request_id,
                   force_refresh=request.force_refresh)
        
        try:
            # Check cache first if not forcing refresh
            source = "fresh"
            if not request.force_refresh:
                cached_metrics = await self.collector.get_cached_metrics()
                if cached_metrics:
                    source = "cache"
                    metrics_proto = self._convert_to_proto(cached_metrics)
                    
                    response = metrics_service_pb2.GetSystemMetricsResponse(
                        metrics=metrics_proto,
                        source=source
                    )
                    
                    # Set timestamp
                    response.timestamp.GetCurrentTime()
                    
                    return response
            
            # Collect fresh metrics
            metric_types = request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
            metrics_data = await self.collector.collect_metrics(metric_types)
            
            # Convert to proto format
            metrics_proto = self._convert_to_proto(metrics_data)
            
            # Create response
            response = metrics_service_pb2.GetSystemMetricsResponse(
                metrics=metrics_proto,
                source=source
            )
            
            # Set timestamp
            response.timestamp.GetCurrentTime()
            
            return response
            
        except Exception as e:
            logger.error("Failed to get system metrics", 
                        error=str(e),
                        request_id=request.request_id)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def StreamSystemMetrics(
        self,
        request: metrics_service_pb2.StreamSystemMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[metrics_service_pb2.SystemMetricsUpdate]:
        """Stream real-time metrics updates"""
        
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()
        
        logger.info("Starting metrics stream",
                   request_id=request.request_id,
                   interval=request.interval_seconds)
        
        try:
            # Send initial metrics
            initial_metrics = await self.collector.collect_metrics(
                request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
            )
            
            update = metrics_service_pb2.SystemMetricsUpdate(
                type=metrics_service_pb2.SystemMetricsUpdate.INITIAL,
                metrics=self._convert_to_proto(initial_metrics)
            )
            update.timestamp.GetCurrentTime()
            
            yield update
            
            # Stream periodic updates
            interval = max(request.interval_seconds, 1)  # Minimum 1 second
            
            while not context.is_active():
                await asyncio.sleep(interval)
                
                # Collect metrics
                metrics_data = await self.collector.collect_metrics(
                    request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL]
                )
                
                # Create update
                update = metrics_service_pb2.SystemMetricsUpdate(
                    type=metrics_service_pb2.SystemMetricsUpdate.PERIODIC,
                    metrics=self._convert_to_proto(metrics_data)
                )
                update.timestamp.GetCurrentTime()
                
                yield update
                
        except asyncio.CancelledError:
            logger.info("Metrics stream cancelled", request_id=request.request_id)
            raise
        except Exception as e:
            logger.error("Error in metrics stream", 
                        error=str(e),
                        request_id=request.request_id)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetMetricsHistory(
        self,
        request: metrics_service_pb2.GetMetricsHistoryRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.GetMetricsHistoryResponse:
        """Get historical metrics data"""
        
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()
        
        logger.info("Getting metrics history",
                   request_id=request.request_id,
                   start_time=request.start_time.ToDatetime().isoformat(),
                   end_time=request.end_time.ToDatetime().isoformat())
        
        try:
            # Convert timestamps
            start_time = request.start_time.ToDatetime()
            end_time = request.end_time.ToDatetime()
            
            # Validate time range
            if end_time < start_time:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, 
                            "End time must be after start time")
            
            if (end_time - start_time).days > 7:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                            "Time range cannot exceed 7 days")
            
            # Get historical data from collector
            history_data = await self.collector.get_historical_metrics(
                start_time=start_time,
                end_time=end_time,
                metric_types=request.metric_types or [metrics_service_pb2.METRIC_TYPE_ALL],
                resolution_seconds=request.resolution_seconds or 60
            )
            
            # Convert to proto format
            data_points = []
            for point in history_data:
                dp = metrics_service_pb2.MetricsDataPoint()
                
                # Set timestamp
                ts = Timestamp()
                ts.FromDatetime(point['timestamp'])
                dp.timestamp.CopyFrom(ts)
                
                # Set metrics
                dp.metrics.CopyFrom(self._convert_to_proto(point['metrics']))
                data_points.append(dp)
            
            # Create metadata
            metadata = metrics_service_pb2.MetricsHistoryMetadata(
                count=len(data_points),
                resolution_seconds=request.resolution_seconds or 60
            )
            
            if data_points:
                metadata.start_time.CopyFrom(data_points[0].timestamp)
                metadata.end_time.CopyFrom(data_points[-1].timestamp)
            
            return metrics_service_pb2.GetMetricsHistoryResponse(
                data_points=data_points,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error("Failed to get metrics history",
                        error=str(e),
                        request_id=request.request_id)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def RefreshMetrics(
        self,
        request: metrics_service_pb2.RefreshMetricsRequest,
        context: grpc.aio.ServicerContext
    ) -> metrics_service_pb2.RefreshMetricsResponse:
        """Force refresh metrics collection"""
        
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()
        
        logger.info("Refreshing metrics", request_id=request.request_id)
        
        try:
            # Force collect all metrics
            metrics_data = await self.collector.collect_metrics(
                [metrics_service_pb2.METRIC_TYPE_ALL],
                force_refresh=True
            )
            
            # Convert to proto
            metrics_proto = self._convert_to_proto(metrics_data)
            
            return metrics_service_pb2.RefreshMetricsResponse(
                success=True,
                metrics=metrics_proto
            )
            
        except Exception as e:
            logger.error("Failed to refresh metrics",
                        error=str(e),
                        request_id=request.request_id)
            
            return metrics_service_pb2.RefreshMetricsResponse(
                success=False,
                metrics=None
            )
    
    def _convert_to_proto(self, metrics_data: Dict[str, Any]) -> metrics_service_pb2.SystemMetrics:
        """Convert collected metrics to proto format"""
        
        metrics = metrics_service_pb2.SystemMetrics()
        
        # CPU metrics
        if 'cpu' in metrics_data:
            cpu_data = metrics_data['cpu']
            metrics.cpu.usage_percent = cpu_data.get('usage_percent', 0)
            metrics.cpu.per_core_percent.extend(cpu_data.get('per_core_percent', []))
            
            if 'frequency' in cpu_data:
                freq = cpu_data['frequency']
                metrics.cpu.frequency.current_mhz = freq.get('current', 0)
                metrics.cpu.frequency.min_mhz = freq.get('min', 0)
                metrics.cpu.frequency.max_mhz = freq.get('max', 0)
            
            if 'load_average' in cpu_data:
                load = cpu_data['load_average']
                metrics.cpu.load_average.one_minute = load.get('1min', 0)
                metrics.cpu.load_average.five_minutes = load.get('5min', 0)
                metrics.cpu.load_average.fifteen_minutes = load.get('15min', 0)
            
            metrics.cpu.count = cpu_data.get('count', 0)
            metrics.cpu.count_logical = cpu_data.get('count_logical', 0)
        
        # Memory metrics
        if 'memory' in metrics_data:
            mem_data = metrics_data['memory']
            metrics.memory.total_bytes = mem_data.get('total', 0)
            metrics.memory.available_bytes = mem_data.get('available', 0)
            metrics.memory.used_bytes = mem_data.get('used', 0)
            metrics.memory.free_bytes = mem_data.get('free', 0)
            metrics.memory.percent = mem_data.get('percent', 0)
            
            if 'swap' in mem_data:
                swap = mem_data['swap']
                metrics.memory.swap.total_bytes = swap.get('total', 0)
                metrics.memory.swap.used_bytes = swap.get('used', 0)
                metrics.memory.swap.free_bytes = swap.get('free', 0)
                metrics.memory.swap.percent = swap.get('percent', 0)
        
        # GPU metrics
        if 'gpu' in metrics_data:
            for gpu_data in metrics_data['gpu']:
                gpu = metrics.gpu.add()
                gpu.id = gpu_data.get('id', 0)
                gpu.name = gpu_data.get('name', '')
                gpu.load_percent = gpu_data.get('load', 0)
                gpu.temperature_celsius = gpu_data.get('temperature', 0)
                gpu.uuid = gpu_data.get('uuid', '')
                gpu.driver_version = gpu_data.get('driver', '')
                
                if 'memory' in gpu_data:
                    mem = gpu_data['memory']
                    gpu.memory.total_bytes = int(mem.get('total', 0) * 1024 * 1024)  # MB to bytes
                    gpu.memory.used_bytes = int(mem.get('used', 0) * 1024 * 1024)
                    gpu.memory.free_bytes = int(mem.get('free', 0) * 1024 * 1024)
                    gpu.memory.percent = mem.get('percent', 0)
        
        # Disk metrics
        if 'disk' in metrics_data:
            disk_data = metrics_data['disk']
            
            if 'usage' in disk_data:
                usage = disk_data['usage']
                metrics.disk.usage.total_bytes = usage.get('total', 0)
                metrics.disk.usage.used_bytes = usage.get('used', 0)
                metrics.disk.usage.free_bytes = usage.get('free', 0)
                metrics.disk.usage.percent = usage.get('percent', 0)
            
            if 'io' in disk_data:
                io = disk_data['io']
                metrics.disk.io.read_count = io.get('read_count', 0)
                metrics.disk.io.write_count = io.get('write_count', 0)
                metrics.disk.io.read_bytes = io.get('read_bytes', 0)
                metrics.disk.io.write_bytes = io.get('write_bytes', 0)
            
            if 'partitions' in disk_data:
                for part_data in disk_data['partitions']:
                    partition = metrics.disk.partitions.add()
                    partition.device = part_data.get('device', '')
                    partition.mount_point = part_data.get('mountpoint', '')
                    partition.filesystem_type = part_data.get('fstype', '')
                    partition.total_bytes = part_data.get('total', 0)
                    partition.used_bytes = part_data.get('used', 0)
                    partition.free_bytes = part_data.get('free', 0)
                    partition.percent = part_data.get('percent', 0)
            
            if 'model_storage' in disk_data:
                storage = disk_data['model_storage']
                metrics.disk.model_storage.path = storage.get('path', '')
                metrics.disk.model_storage.total_bytes = storage.get('total', 0)
                metrics.disk.model_storage.used_bytes = storage.get('used', 0)
                metrics.disk.model_storage.free_bytes = storage.get('free', 0)
                metrics.disk.model_storage.percent = storage.get('percent', 0)
        
        # Network metrics
        if 'network' in metrics_data:
            net_data = metrics_data['network']
            
            if 'io' in net_data:
                io = net_data['io']
                metrics.network.io.bytes_sent = io.get('bytes_sent', 0)
                metrics.network.io.bytes_received = io.get('bytes_recv', 0)
                metrics.network.io.packets_sent = io.get('packets_sent', 0)
                metrics.network.io.packets_received = io.get('packets_recv', 0)
                metrics.network.io.errors_in = io.get('errin', 0)
                metrics.network.io.errors_out = io.get('errout', 0)
                metrics.network.io.dropped_in = io.get('dropin', 0)
                metrics.network.io.dropped_out = io.get('dropout', 0)
            
            if 'connections' in net_data:
                for state, count in net_data['connections'].items():
                    metrics.network.connection_states[state] = count
        
        return metrics