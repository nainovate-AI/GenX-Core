"""
Main Metrics Collector
Orchestrates all individual collectors
"""
import asyncio
from typing import Dict, Any, List, Optional
import time
import sys
import os

# Add genx_platform to path
current_file = os.path.abspath(__file__)
collectors_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(collectors_dir)
metrics_root = os.path.dirname(src_dir)
microservices_dir = os.path.dirname(metrics_root)
genx_components = os.path.dirname(microservices_dir)
genx_platform = os.path.dirname(genx_components)
sys.path.insert(0, genx_platform)

from genx_components.microservices.grpc import metrics_service_pb2

# Local imports
sys.path.insert(0, src_dir)
from collectors.cpu import CPUCollector
from collectors.memory import MemoryCollector
from collectors.gpu import GPUCollector
from collectors.disk import DiskCollector
from collectors.network import NetworkCollector
from utils.logger import setup_logging

logger = setup_logging(__name__)


class MetricsCollector:
    """
    Main collector that orchestrates all metric collectors
    """
    
    def __init__(self):
        self._collectors = {
            'cpu': CPUCollector(),
            'memory': MemoryCollector(),
            'gpu': GPUCollector(),
            'disk': DiskCollector(),
            'network': NetworkCollector()
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize all collectors"""
        if self._initialized:
            return
        
        logger.info("Initializing metrics collectors...")
        
        # Initialize collectors concurrently
        init_tasks = []
        for name, collector in self._collectors.items():
            init_tasks.append(self._init_collector(name, collector))
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check initialization results
        failed_collectors = []
        for i, (name, _) in enumerate(self._collectors.items()):
            if isinstance(results[i], Exception):
                failed_collectors.append(name)
                logger.error(f"Failed to initialize {name} collector: {results[i]}")
        
        if failed_collectors:
            logger.warning(f"Some collectors failed to initialize: {failed_collectors}")
        
        self._initialized = True
        logger.info("Metrics collectors initialized")
    
    async def _init_collector(self, name: str, collector):
        """Initialize a single collector with error handling"""
        try:
            await collector.initialize()
            return name, True
        except Exception as e:
            logger.error(f"Error initializing {name} collector: {e}")
            return name, False
    
    async def collect_all(self) -> Dict[str, Any]:
        """Collect all available metrics"""
        return await self.collect_metrics([metrics_service_pb2.METRIC_TYPE_ALL])
    
    async def collect_metrics(self, metric_types: List[int]) -> Dict[str, Any]:
        """
        Collect specified metrics
        
        Args:
            metric_types: List of MetricType enums from protobuf
            
        Returns:
            Dictionary with collected metrics
        """
        if not self._initialized:
            await self.initialize()
        
        # Determine what to collect
        collect_all = metrics_service_pb2.METRIC_TYPE_ALL in metric_types
        collectors_to_run = []
        
        if collect_all:
            collectors_to_run = list(self._collectors.items())
        else:
            # Map metric types to collectors
            type_mapping = {
                metrics_service_pb2.METRIC_TYPE_CPU: 'cpu',
                metrics_service_pb2.METRIC_TYPE_MEMORY: 'memory',
                metrics_service_pb2.METRIC_TYPE_GPU: 'gpu',
                metrics_service_pb2.METRIC_TYPE_DISK: 'disk',
                metrics_service_pb2.METRIC_TYPE_NETWORK: 'network'
            }
            
            for metric_type in metric_types:
                collector_name = type_mapping.get(metric_type)
                if collector_name and collector_name in self._collectors:
                    collectors_to_run.append((collector_name, self._collectors[collector_name]))
        
        # Collect metrics concurrently
        start_time = time.time()
        tasks = []
        
        for name, collector in collectors_to_run:
            tasks.append(self._collect_single(name, collector))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        metrics = {}
        for i, (name, _) in enumerate(collectors_to_run):
            if isinstance(results[i], Exception):
                logger.error(f"Failed to collect {name} metrics: {results[i]}")
                # Add error placeholder
                metrics[name] = {
                    'error': True,
                    'error_message': str(results[i])
                }
            else:
                metrics[name] = results[i]
        
        # Add collection metadata
        metrics['_metadata'] = {
            'collection_time': time.time(),
            'duration_ms': int((time.time() - start_time) * 1000),
            'collectors_run': len(collectors_to_run),
            'collectors_failed': sum(1 for m in metrics.values() if isinstance(m, dict) and m.get('error'))
        }
        
        return metrics
    
    async def _collect_single(self, name: str, collector) -> Dict[str, Any]:
        """Collect metrics from a single collector"""
        try:
            return await collector.collect()
        except Exception as e:
            logger.error(f"Error collecting {name} metrics: {e}")
            raise
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """Get statistics for all collectors"""
        stats = {}
        
        for name, collector in self._collectors.items():
            stats[name] = collector.get_stats()
        
        return {
            'collectors': stats,
            'initialized': self._initialized,
            'total_collectors': len(self._collectors)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up metrics collectors...")
        
        # Cleanup GPU collector specifically (NVML shutdown)
        if 'gpu' in self._collectors:
            gpu_collector = self._collectors['gpu']
            if hasattr(gpu_collector, 'cleanup'):
                await gpu_collector.cleanup()
        
        logger.info("Metrics collectors cleanup complete")