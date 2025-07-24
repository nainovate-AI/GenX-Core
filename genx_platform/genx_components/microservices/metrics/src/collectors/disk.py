"""
Disk Metrics Collector
Collects disk usage, I/O statistics, and partition information
"""
import os
import time
from typing import Dict, Any, List, Optional
import psutil

from .base import BaseCollector
from ..utils.logger import setup_logging

logger = setup_logging(__name__)


class DiskCollector(BaseCollector):
    """
    Collects disk-related metrics including usage, I/O, and partitions
    """
    
    def __init__(self):
        super().__init__("disk")
        self._model_storage_path = os.environ.get("MODEL_STORAGE_PATH", "/models")
        self._last_io_counters = None
        self._last_io_time = None
        
    async def _initialize(self) -> None:
        """Initialize disk collector"""
        # Get initial I/O counters for rate calculation
        try:
            self._last_io_counters = psutil.disk_io_counters()
            self._last_io_time = time.time()
        except Exception as e:
            logger.warning(f"Failed to get initial I/O counters: {e}")
        
        # Log disk partitions
        partitions = psutil.disk_partitions(all=False)
        logger.info(
            f"Disk collector initialized with {len(partitions)} partitions",
            model_storage_path=self._model_storage_path
        )
    
    async def _collect(self) -> Dict[str, Any]:
        """Collect disk metrics"""
        metrics = {}
        
        # Overall disk usage (root partition)
        with self._time_operation("disk_usage"):
            root_usage = psutil.disk_usage('/')
            metrics['usage'] = {
                'total': root_usage.total,
                'used': root_usage.used,
                'free': root_usage.free,
                'percent': round(root_usage.percent, 2),
                
                # Human-readable values
                'total_gb': self._bytes_to_gb(root_usage.total),
                'used_gb': self._bytes_to_gb(root_usage.used),
                'free_gb': self._bytes_to_gb(root_usage.free)
            }
        
        # Disk I/O statistics
        with self._time_operation("disk_io"):
            metrics['io'] = await self._collect_io_metrics()
        
        # Partition information
        with self._time_operation("partitions"):
            metrics['partitions'] = await self._collect_partition_metrics()
        
        # Model storage specific metrics
        with self._time_operation("model_storage"):
            metrics['model_storage'] = await self._collect_model_storage_metrics()
        
        # Disk health indicators
        metrics['health'] = self._calculate_disk_health(metrics)
        
        return metrics
    
    async def _collect_io_metrics(self) -> Dict[str, Any]:
        """Collect disk I/O metrics with rate calculation"""
        io_metrics = {}
        
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if current_io:
                io_metrics = {
                    'read_count': current_io.read_count,
                    'write_count': current_io.write_count,
                    'read_bytes': current_io.read_bytes,
                    'write_bytes': current_io.write_bytes,
                    'read_time_ms': current_io.read_time,
                    'write_time_ms': current_io.write_time,
                    
                    # Human-readable cumulative values
                    'read_gb': self._bytes_to_gb(current_io.read_bytes),
                    'write_gb': self._bytes_to_gb(current_io.write_bytes)
                }
                
                # Calculate rates if we have previous data
                if self._last_io_counters and self._last_io_time:
                    time_delta = current_time - self._last_io_time
                    
                    if time_delta > 0:
                        # Bytes per second
                        read_rate = (current_io.read_bytes - self._last_io_counters.read_bytes) / time_delta
                        write_rate = (current_io.write_bytes - self._last_io_counters.write_bytes) / time_delta
                        
                        # Operations per second
                        read_ops = (current_io.read_count - self._last_io_counters.read_count) / time_delta
                        write_ops = (current_io.write_count - self._last_io_counters.write_count) / time_delta
                        
                        io_metrics['rates'] = {
                            'read_bytes_per_sec': round(read_rate, 2),
                            'write_bytes_per_sec': round(write_rate, 2),
                            'read_mb_per_sec': round(read_rate / (1024 * 1024), 2),
                            'write_mb_per_sec': round(write_rate / (1024 * 1024), 2),
                            'read_ops_per_sec': round(read_ops, 2),
                            'write_ops_per_sec': round(write_ops, 2),
                            'total_ops_per_sec': round(read_ops + write_ops, 2)
                        }
                        
                        # I/O utilization (simplified)
                        if hasattr(current_io, 'busy_time') and hasattr(self._last_io_counters, 'busy_time'):
                            busy_delta = current_io.busy_time - self._last_io_counters.busy_time
                            io_metrics['utilization_percent'] = round(
                                min(100, (busy_delta / (time_delta * 1000)) * 100), 2
                            )
                
                # Update last values
                self._last_io_counters = current_io
                self._last_io_time = current_time
                
            else:
                io_metrics = self._get_empty_io_metrics()
                
        except Exception as e:
            logger.error(f"Failed to collect I/O metrics: {e}")
            io_metrics = self._get_empty_io_metrics()
        
        return io_metrics
    
    async def _collect_partition_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics for all disk partitions"""
        partitions_data = []
        
        try:
            partitions = psutil.disk_partitions(all=False)
            
            for partition in partitions:
                try:
                    # Skip certain filesystem types
                    if partition.fstype in ['squashfs', 'tmpfs', 'devtmpfs']:
                        continue
                    
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    partition_data = {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'filesystem': partition.fstype,
                        'options': partition.opts.split(','),
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': round(usage.percent, 2),
                        
                        # Human-readable values
                        'total_gb': self._bytes_to_gb(usage.total),
                        'used_gb': self._bytes_to_gb(usage.used),
                        'free_gb': self._bytes_to_gb(usage.free),
                        
                        # Flags
                        'is_readonly': 'ro' in partition.opts,
                        'is_removable': self._is_removable_device(partition.device)
                    }
                    
                    # Add partition-specific I/O if available
                    try:
                        partition_io = psutil.disk_io_counters(perdisk=True).get(
                            os.path.basename(partition.device).replace('/dev/', '')
                        )
                        if partition_io:
                            partition_data['io'] = {
                                'read_count': partition_io.read_count,
                                'write_count': partition_io.write_count,
                                'read_mb': self._bytes_to_mb(partition_io.read_bytes),
                                'write_mb': self._bytes_to_mb(partition_io.write_bytes)
                            }
                    except Exception:
                        pass
                    
                    partitions_data.append(partition_data)
                    
                except PermissionError:
                    logger.debug(f"Permission denied for partition {partition.mountpoint}")
                except Exception as e:
                    logger.warning(f"Failed to get metrics for partition {partition.device}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to collect partition metrics: {e}")
        
        return partitions_data
    
    async def _collect_model_storage_metrics(self) -> Dict[str, Any]:
        """Collect model storage specific metrics"""
        model_metrics = {
            'path': self._model_storage_path,
            'exists': os.path.exists(self._model_storage_path)
        }
        
        try:
            if model_metrics['exists'] and os.path.isdir(self._model_storage_path):
                # Get storage stats
                stat = os.statvfs(self._model_storage_path)
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_avail * stat.f_frsize
                used = total - free
                
                model_metrics.update({
                    'total': total,
                    'used': used,
                    'free': free,
                    'percent': round((used / total * 100) if total > 0 else 0, 2),
                    'total_gb': self._bytes_to_gb(total),
                    'used_gb': self._bytes_to_gb(used),
                    'free_gb': self._bytes_to_gb(free)
                })
                
                # Count and measure models
                model_info = await self._analyze_model_directory()
                model_metrics.update(model_info)
                
            else:
                model_metrics.update({
                    'total': 0,
                    'used': 0,
                    'free': 0,
                    'percent': 0,
                    'error': 'Model storage path not found or not a directory'
                })
                
        except Exception as e:
            logger.error(f"Failed to collect model storage metrics: {e}")
            model_metrics['error'] = str(e)
        
        return model_metrics
    
    async def _analyze_model_directory(self) -> Dict[str, Any]:
        """Analyze model directory for model count and sizes"""
        info = {
            'model_count': 0,
            'total_model_size': 0,
            'models': [],
            'largest_model': None
        }
        
        try:
            # Walk through model directory
            for entry in os.scandir(self._model_storage_path):
                if entry.is_dir():
                    model_size = self._get_directory_size(entry.path)
                    model_data = {
                        'name': entry.name,
                        'size': model_size,
                        'size_gb': self._bytes_to_gb(model_size)
                    }
                    
                    info['models'].append(model_data)
                    info['model_count'] += 1
                    info['total_model_size'] += model_size
                    
                    # Track largest model
                    if not info['largest_model'] or model_size > info['largest_model']['size']:
                        info['largest_model'] = model_data
            
            # Sort models by size
            info['models'].sort(key=lambda x: x['size'], reverse=True)
            
            # Keep only top 10 for response
            info['top_models'] = info['models'][:10]
            del info['models']  # Remove full list to save space
            
            # Add human-readable total
            info['total_model_size_gb'] = self._bytes_to_gb(info['total_model_size'])
            
        except Exception as e:
            logger.warning(f"Failed to analyze model directory: {e}")
            info['error'] = str(e)
        
        return info
    
    def _get_directory_size(self, path: str) -> int:
        """Recursively get directory size in bytes"""
        total = 0
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat().st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += self._get_directory_size(entry.path)
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access {path}: {e}")
        
        return total
    
    def _is_removable_device(self, device: str) -> bool:
        """Check if device is removable (USB, etc.)"""
        # Simplified check - in production would check /sys/block/*/removable
        removable_patterns = ['/media/', '/mnt/', '/Volumes/']
        return any(pattern in device for pattern in removable_patterns)
    
    def _calculate_disk_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disk health indicators"""
        health = {
            'status': 'healthy',
            'warnings': [],
            'critical': []
        }
        
        # Check root disk usage
        root_percent = metrics['usage']['percent']
        if root_percent > 95:
            health['critical'].append(f"Root disk critically full: {root_percent}%")
            health['status'] = 'critical'
        elif root_percent > 90:
            health['warnings'].append(f"Root disk nearly full: {root_percent}%")
            health['status'] = 'warning' if health['status'] == 'healthy' else health['status']
        elif root_percent > 80:
            health['warnings'].append(f"Root disk usage high: {root_percent}%")
        
        # Check model storage
        if 'model_storage' in metrics and metrics['model_storage'].get('exists'):
            model_percent = metrics['model_storage'].get('percent', 0)
            if model_percent > 90:
                health['warnings'].append(f"Model storage nearly full: {model_percent}%")
                health['status'] = 'warning' if health['status'] == 'healthy' else health['status']
        
        # Check I/O rates (if available)
        if 'io' in metrics and 'rates' in metrics['io']:
            write_rate = metrics['io']['rates'].get('write_mb_per_sec', 0)
            if write_rate > 500:  # > 500 MB/s sustained write
                health['warnings'].append(f"High disk write rate: {write_rate} MB/s")
        
        return health
    
    def _get_empty_io_metrics(self) -> Dict[str, Any]:
        """Return empty I/O metrics structure"""
        return {
            'read_count': 0,
            'write_count': 0,
            'read_bytes': 0,
            'write_bytes': 0,
            'read_time_ms': 0,
            'write_time_ms': 0,
            'read_gb': 0,
            'write_gb': 0
        }
    
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Return disk-specific error metrics"""
        return {
            'usage': {
                'total': 0,
                'used': 0,
                'free': 0,
                'percent': 0.0
            },
            'io': self._get_empty_io_metrics(),
            'partitions': [],
            'model_storage': {
                'path': self._model_storage_path,
                'exists': False,
                'error': error_message
            },
            'error': True,
            'error_message': error_message,
            '_metadata': {
                'collector': self.name,
                'error': True
            }
        }