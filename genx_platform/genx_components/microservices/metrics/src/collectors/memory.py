"""
Memory Metrics Collector
Collects RAM, swap, and memory usage metrics
"""
from typing import Dict, Any
import psutil
import os
import sys

from .base import BaseCollector

current_file = os.path.abspath(__file__)
collectors_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(collectors_dir)
sys.path.insert(0, src_dir)

from genx_components.microservices.metrics.src.utils.logger import setup_logging

logger = setup_logging(__name__)


class MemoryCollector(BaseCollector):
    """
    Collects memory-related metrics including RAM and swap
    """
    
    def __init__(self):
        super().__init__("memory")
        self._total_memory = 0
        self._total_swap = 0
        
    async def _initialize(self) -> None:
        """Initialize memory collector"""
        # Get total memory for reference
        mem = psutil.virtual_memory()
        self._total_memory = mem.total
        
        swap = psutil.swap_memory()
        self._total_swap = swap.total
        
        logger.info(
            "Memory collector initialized",
            total_memory_gb=self._bytes_to_gb(self._total_memory),
            total_swap_gb=self._bytes_to_gb(self._total_swap)
        )
    
    async def _collect(self) -> Dict[str, Any]:
        """Collect memory metrics"""
        metrics = {}
        
        # Virtual memory (RAM)
        with self._time_operation("virtual_memory"):
            mem = psutil.virtual_memory()
            
            metrics['ram'] = {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'free': mem.free,
                'percent': round(mem.percent, 2),
                
                # Human-readable values
                'total_gb': self._bytes_to_gb(mem.total),
                'available_gb': self._bytes_to_gb(mem.available),
                'used_gb': self._bytes_to_gb(mem.used),
                'free_gb': self._bytes_to_gb(mem.free),
                
                # Platform-specific fields
                'active': getattr(mem, 'active', 0),
                'inactive': getattr(mem, 'inactive', 0),
                'buffers': getattr(mem, 'buffers', 0),
                'cached': getattr(mem, 'cached', 0),
                'shared': getattr(mem, 'shared', 0),
                'slab': getattr(mem, 'slab', 0),
            }
            
            # Calculate additional metrics
            metrics['ram']['used_excluding_cache'] = mem.used - getattr(mem, 'cached', 0) - getattr(mem, 'buffers', 0)
            metrics['ram']['percent_excluding_cache'] = round(
                self._safe_divide(metrics['ram']['used_excluding_cache'], mem.total) * 100, 2
            )
        
        # Swap memory
        with self._time_operation("swap_memory"):
            swap = psutil.swap_memory()
            
            metrics['swap'] = {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': round(swap.percent, 2),
                'sin': swap.sin,    # Swap in (bytes)
                'sout': swap.sout,  # Swap out (bytes)
                
                # Human-readable values
                'total_gb': self._bytes_to_gb(swap.total),
                'used_gb': self._bytes_to_gb(swap.used),
                'free_gb': self._bytes_to_gb(swap.free),
            }
            
            # Swap activity indicators
            metrics['swap']['is_active'] = swap.sin > 0 or swap.sout > 0
            metrics['swap']['enabled'] = swap.total > 0
        
        # Combined memory metrics
        total_memory = mem.total + swap.total
        used_memory = mem.used + swap.used
        
        metrics['combined'] = {
            'total': total_memory,
            'used': used_memory,
            'percent': round(self._safe_divide(used_memory, total_memory) * 100, 2),
            'total_gb': self._bytes_to_gb(total_memory),
            'used_gb': self._bytes_to_gb(used_memory)
        }
        
        # Memory pressure indicators
        metrics['pressure'] = self._calculate_memory_pressure(mem, swap)
        
        # Process memory stats (top consumers)
        try:
            metrics['top_consumers'] = await self._get_top_memory_consumers(5)
        except Exception as e:
            logger.warning(f"Failed to get top memory consumers: {e}")
            metrics['top_consumers'] = []
        
        # For backward compatibility
        metrics['total'] = mem.total
        metrics['available'] = mem.available
        metrics['used'] = mem.used
        metrics['free'] = mem.free
        metrics['percent'] = round(mem.percent, 2)
        metrics['swap'] = {
            'total': swap.total,
            'used': swap.used,
            'free': swap.free,
            'percent': round(swap.percent, 2)
        }
        
        return metrics
    
    def _calculate_memory_pressure(self, mem: Any, swap: Any) -> Dict[str, Any]:
        """Calculate memory pressure indicators"""
        pressure = {
            'level': 'normal',
            'swap_pressure': False,
            'cache_pressure': False,
            'oom_risk': False
        }
        
        # Determine pressure level
        if mem.percent > 95:
            pressure['level'] = 'critical'
            pressure['oom_risk'] = True
        elif mem.percent > 90:
            pressure['level'] = 'high'
            pressure['oom_risk'] = True
        elif mem.percent > 80:
            pressure['level'] = 'medium'
        elif mem.percent > 70:
            pressure['level'] = 'low'
        
        # Check swap pressure
        if swap.total > 0 and swap.percent > 50:
            pressure['swap_pressure'] = True
            
        # Check cache pressure (Linux specific)
        cached = getattr(mem, 'cached', 0)
        if cached > 0:
            cache_ratio = cached / mem.total
            if cache_ratio < 0.1:  # Less than 10% cache
                pressure['cache_pressure'] = True
        
        return pressure
    
    async def _get_top_memory_consumers(self, limit: int = 5) -> list:
        """Get top memory consuming processes"""
        processes = []
        
        try:
            # Iterate through all processes
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['memory_percent'] and pinfo['memory_percent'] > 0.1:  # Only include if using > 0.1%
                        processes.append({
                            'pid': pinfo['pid'],
                            'name': pinfo['name'],
                            'memory_mb': self._bytes_to_mb(pinfo['memory_info'].rss),
                            'memory_percent': round(pinfo['memory_percent'], 2)
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Sort by memory usage and return top N
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            return processes[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top memory consumers: {e}")
            return []
    
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Return memory-specific error metrics"""
        return {
            'total': self._total_memory,
            'available': 0,
            'used': 0,
            'free': 0,
            'percent': 0.0,
            'ram': {
                'total': self._total_memory,
                'available': 0,
                'used': 0,
                'free': 0,
                'percent': 0.0,
                'total_gb': self._bytes_to_gb(self._total_memory)
            },
            'swap': {
                'total': self._total_swap,
                'used': 0,
                'free': self._total_swap,
                'percent': 0.0
            },
            'error': True,
            'error_message': error_message,
            '_metadata': {
                'collector': self.name,
                'error': True
            }
        }