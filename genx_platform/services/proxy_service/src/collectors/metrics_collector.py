# services/proxy_service/src/collectors/metrics_collector.py
"""
System Metrics Collector
Collects CPU, Memory, GPU, Disk, and Network metrics
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psutil
import GPUtil
import shutil
import structlog
from opentelemetry import trace, metrics as otel_metrics
import prometheus_client

from ..generated import metrics_service_pb2
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class MetricsCollector:
    """
    Collects system metrics with caching and history tracking
    """
    
    def __init__(self):
        self.cache_manager = None
        self.collection_interval = 5  # seconds
        self._collection_task = None
        self._last_collection_time = 0
        self._collection_lock = asyncio.Lock()
        
        # Prometheus metrics
        self.prom_collection_duration = prometheus_client.Histogram(
            'proxy_metrics_collection_duration_seconds',
            'Time spent collecting metrics',
            ['metric_type']
        )
        
        self.prom_collection_errors = prometheus_client.Counter(
            'proxy_metrics_collection_errors_total',
            'Total number of metric collection errors',
            ['metric_type']
        )
    
    async def initialize(self, cache_manager):
        """Initialize the collector with cache manager"""
        self.cache_manager = cache_manager
        
        # Start periodic collection
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._periodic_collection())
            logger.info("Started periodic metrics collection")
    
    async def shutdown(self):
        """Shutdown the collector"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped periodic metrics collection")
    
    async def _periodic_collection(self):
        """Background task for periodic metrics collection"""
        while True:
            try:
                # Collect all metrics
                metrics = await self.collect_metrics([metrics_service_pb2.METRIC_TYPE_ALL])
                
                # Store in cache
                await self._store_metrics(metrics)
                
                # Store in history
                await self._store_history(metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic collection", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def collect_metrics(
        self, 
        metric_types: List[int],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Collect specified system metrics
        """
        async with self._collection_lock:
            # Check if we need ALL metrics
            collect_all = metrics_service_pb2.METRIC_TYPE_ALL in metric_types
            
            metrics = {}
            
            # Collect each metric type
            if collect_all or metrics_service_pb2.METRIC_TYPE_CPU in metric_types:
                metrics['cpu'] = await self._collect_cpu_metrics()
            
            if collect_all or metrics_service_pb2.METRIC_TYPE_MEMORY in metric_types:
                metrics['memory'] = await self._collect_memory_metrics()
            
            if collect_all or metrics_service_pb2.METRIC_TYPE_GPU in metric_types:
                metrics['gpu'] = await self._collect_gpu_metrics()
            
            if collect_all or metrics_service_pb2.METRIC_TYPE_DISK in metric_types:
                metrics['disk'] = await self._collect_disk_metrics()
            
            if collect_all or metrics_service_pb2.METRIC_TYPE_NETWORK in metric_types:
                metrics['network'] = await self._collect_network_metrics()
            
            self._last_collection_time = time.time()
            
            return metrics
    
    async def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cached metrics if available and fresh"""
        if not self.cache_manager:
            return None
        
        # Check if cache is fresh (less than collection interval old)
        if time.time() - self._last_collection_time < self.collection_interval:
            return await self.cache_manager.get("metrics:system:latest")
        
        return None
    
    async def get_historical_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_types: List[int],
        resolution_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Get historical metrics from Redis"""
        if not self.cache_manager:
            return []
        
        history = []
        
        # Calculate time buckets
        current_time = start_time
        while current_time < end_time:
            bucket_key = f"metrics:history:{int(current_time.timestamp()) // 60}"
            
            # Get data from this bucket
            data = await self.cache_manager.get(bucket_key)
            if data:
                # Filter by metric types if needed
                if metrics_service_pb2.METRIC_TYPE_ALL not in metric_types:
                    filtered_data = {'timestamp': current_time}
                    for metric_type in metric_types:
                        if metric_type == metrics_service_pb2.METRIC_TYPE_CPU and 'cpu' in data:
                            filtered_data['cpu'] = data['cpu']
                        elif metric_type == metrics_service_pb2.METRIC_TYPE_MEMORY and 'memory' in data:
                            filtered_data['memory'] = data['memory']
                        elif metric_type == metrics_service_pb2.METRIC_TYPE_GPU and 'gpu' in data:
                            filtered_data['gpu'] = data['gpu']
                        elif metric_type == metrics_service_pb2.METRIC_TYPE_DISK and 'disk' in data:
                            filtered_data['disk'] = data['disk']
                        elif metric_type == metrics_service_pb2.METRIC_TYPE_NETWORK and 'network' in data:
                            filtered_data['network'] = data['network']
                    data = filtered_data
                
                history.append({
                    'timestamp': current_time,
                    'metrics': data
                })
            
            current_time += timedelta(seconds=resolution_seconds)
        
        return history
    
    @prom_collection_duration.labels(metric_type="cpu").time()
    async def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics"""
        try:
            return await asyncio.to_thread(self._get_cpu_metrics)
        except Exception as e:
            self.prom_collection_errors.labels(metric_type="cpu").inc()
            logger.error("Failed to collect CPU metrics", error=str(e))
            return {}
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics (blocking)"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        load_avg = psutil.getloadavg()
        
        return {
            'usage_percent': sum(cpu_percent) / len(cpu_percent),
            'per_core_percent': cpu_percent,
            'frequency': {
                'current': cpu_freq.current if cpu_freq else 0,
                'min': cpu_freq.min if cpu_freq else 0,
                'max': cpu_freq.max if cpu_freq else 0
            },
            'load_average': {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            },
            'count': psutil.cpu_count(logical=False),
            'count_logical': psutil.cpu_count(logical=True)
        }
    
    @prom_collection_duration.labels(metric_type="memory").time()
    async def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics"""
        try:
            return await asyncio.to_thread(self._get_memory_metrics)
        except Exception as e:
            self.prom_collection_errors.labels(metric_type="memory").inc()
            logger.error("Failed to collect memory metrics", error=str(e))
            return {}
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics (blocking)"""
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'total': virtual_mem.total,
            'available': virtual_mem.available,
            'used': virtual_mem.used,
            'free': virtual_mem.free,
            'percent': virtual_mem.percent,
            'swap': {
                'total': swap_mem.total,
                'used': swap_mem.used,
                'free': swap_mem.free,
                'percent': swap_mem.percent
            }
        }
    
    @prom_collection_duration.labels(metric_type="gpu").time()
    async def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect GPU metrics"""
        try:
            return await asyncio.to_thread(self._get_gpu_metrics)
        except Exception as e:
            self.prom_collection_errors.labels(metric_type="gpu").inc()
            logger.error("Failed to collect GPU metrics", error=str(e))
            return []
    
    def _get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU metrics (blocking)"""
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory': {
                        'total': gpu.memoryTotal,
                        'used': gpu.memoryUsed,
                        'free': gpu.memoryFree,
                        'percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0
                    },
                    'temperature': gpu.temperature,
                    'uuid': gpu.uuid,
                    'driver': gpu.driver
                }
                for gpu in gpus
            ]
        except Exception:
            return []
    
    @prom_collection_duration.labels(metric_type="disk").time()
    async def _collect_disk_metrics(self) -> Dict[str, Any]:
        """Collect disk metrics"""
        try:
            return await asyncio.to_thread(self._get_disk_metrics)
        except Exception as e:
            self.prom_collection_errors.labels(metric_type="disk").inc()
            logger.error("Failed to collect disk metrics", error=str(e))
            return {}
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk metrics (blocking)"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Get all disk partitions
        partitions = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                })
            except PermissionError:
                continue
        
        # Model storage path metrics
        model_storage_path = settings.MODEL_STORAGE_PATH
        try:
            model_storage = shutil.disk_usage(model_storage_path)
            model_storage_info = {
                'path': model_storage_path,
                'total': model_storage.total,
                'used': model_storage.used,
                'free': model_storage.free,
                'percent': (model_storage.used / model_storage.total * 100) if model_storage.total > 0 else 0
            }
        except:
            model_storage_info = {
                'path': model_storage_path,
                'total': 0,
                'used': 0,
                'free': 0,
                'percent': 0
            }
        
        return {
            'usage': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent
            },
            'io': {
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            },
            'partitions': partitions,
            'model_storage': model_storage_info
        }
    
    @prom_collection_duration.labels(metric_type="network").time()
    async def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics"""
        try:
            return await asyncio.to_thread(self._get_network_metrics)
        except Exception as e:
            self.prom_collection_errors.labels(metric_type="network").inc()
            logger.error("Failed to collect network metrics", error=str(e))
            return {}
    
    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics (blocking)"""
        net_io = psutil.net_io_counters()
        
        # Count connections by status
        connection_states = {}
        try:
            for conn in psutil.net_connections():
                state = conn.status
                connection_states[state] = connection_states.get(state, 0) + 1
        except:
            pass
        
        return {
            'io': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            },
            'connections': connection_states
        }
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store latest metrics in cache"""
        if not self.cache_manager:
            return
        
        try:
            await self.cache_manager.set(
                "metrics:system:latest",
                metrics,
                ttl=300  # 5 minutes
            )
        except Exception as e:
            logger.error("Failed to store metrics in cache", error=str(e))
    
    async def _store_history(self, metrics: Dict[str, Any]):
        """Store metrics in time-series history"""
        if not self.cache_manager:
            return
        
        try:
            timestamp = int(datetime.utcnow().timestamp())
            bucket_key = f"metrics:history:{timestamp // 60}"
            
            # Store metrics in bucket
            await self.cache_manager.set(
                bucket_key,
                metrics,
                ttl=86400  # 24 hours
            )
            
            # Update Prometheus gauges for Grafana
            self._update_prometheus_metrics(metrics)
            
        except Exception as e:
            logger.error("Failed to store metrics history", error=str(e))
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus metrics for Grafana"""
        # CPU metrics
        if 'cpu' in metrics:
            cpu_gauge = prometheus_client.Gauge(
                'system_cpu_usage_percent',
                'CPU usage percentage',
                registry=prometheus_client.REGISTRY
            )
            cpu_gauge.set(metrics['cpu'].get('usage_percent', 0))
        
        # Memory metrics  
        if 'memory' in metrics:
            memory_gauge = prometheus_client.Gauge(
                'system_memory_usage_bytes',
                'Memory usage in bytes',
                registry=prometheus_client.REGISTRY
            )
            memory_gauge.set(metrics['memory'].get('used', 0))
        
        # GPU metrics
        for i, gpu in enumerate(metrics.get('gpu', [])):
            gpu_load_gauge = prometheus_client.Gauge(
                f'system_gpu_{i}_load_percent',
                f'GPU {i} load percentage',
                registry=prometheus_client.REGISTRY
            )
            gpu_load_gauge.set(gpu.get('load', 0))