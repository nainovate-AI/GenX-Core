"""
CPU Metric Collector
"""

import os 
import platform
from typing import Dict, Any, List
import psutil

from .base import BaseCollector
from ..utils.logger import setup_logging

logger = setup_logging(__name__)

class CPUCollector(BaseCollector):
    """
    Collects CPU-related metrics
    """

    def __init__(self):
        super().__init__("cpu")
        self._platform = platform.system()
        self._processor = platform.processor()
        self._cpu_count_logical = 0
        self._cpu_count_physical = 0

    async def _initialize(self) -> None:
        """
        Initialize the CPU collector.
        """
        # Get CPU counts
        self._cpu_count_logical = psutil.cpu_count(logical=True)
        self._cpu_count_physical = psutil.cpu_count(logical=False)

        logger.info(
            "CPU collector initialized",
            platform = self._platform,
            processor = self._processor,
            logical_cpu = self._cpu_count_logical,
            physical_cpu = self._cpu_count_physical
        )

    async def _collect(self) -> Dict[str, Any]:
        """Collect CPU metrics."""
        metrics = {}

        # CPU usage percentage
        # Use interval for more accurate reading
        with self._time_operation("cpu_percent"):
            metrics["usage_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["per_core_percent"] = psutil.cpu_percent(interval=0.1, percpu=True)

        # CPU counts
        metrics["count"] = self._cpu_count_physical
        metrics["count_logical"] = self._cpu_count_logical

        # CPU frequency
        with self._time_operation("cpu_freq"):
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics["frequency"] = {
                    "current": round(cpu_freq.current, 2),
                    "min": round(cpu_freq.min, 2),
                    "max": round(cpu_freq.max, 2)
                }

                # Per-Cpu frequency if available
                try:
                    per_cpu_freq = psutil.cpu_freq(percpu=True)
                    if per_cpu_freq:
                        metrics["per_core_frequency"] = [
                            {
                                "current": round(freq.current, 2),
                                "min": round(freq.min, 2),
                                "max": round(freq.max, 2)
                            } for freq in per_cpu_freq
                        ]
                except Exception:
                    # Not supported on all platforms
                    pass
            else:
                metrics["frequency"] = {
                    "current": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }
        
        # Load average (Unix-like systems)
        with self._time_operation("load_average"):
            if hasattr(os, 'getloadavg'):
                load = os.getloadavg()
                metrics["load_average"] = {
                    "1m": round(load[0], 2),
                    "5m": round(load[1], 2),
                    "15m": round(load[2], 2)
                }

                # Normalized load (load per CPU)
                metrics['load_average_normalized'] = {
                    '1m': round(load[0] / self._cpu_count_logical, 2),
                    '5m': round(load[1] / self._cpu_count_logical, 2),
                    '15m': round(load[2] / self._cpu_count_logical, 2)
                }
            else:
                # Windows does not support load averages
                # Could calculate process-based load as an alternative
                metrics["load_average"] = {
                    "1m": 0.0,
                    "5m": 0.0,
                    "15m": 0.0
                }
        
        # CPU stats
        with self._time_operation("cpu_stats"):
            try:
                cpu_stats = psutil.cpu_stats()
                metrics["stats"] = {
                    "ctx_switches": cpu_stats.ctx_switches,
                    "interrupts": cpu_stats.interrupts,
                    "soft_interrupts": getattr(cpu_stats, 'soft_interrupts', 0),
                    "syscalls": getattr(cpu_stats, 'syscalls', 0),
                }
            except Exception as e:
                logger.warning(
                    "Failed to collect CPU stats",
                    error=str(e)
                )
                metrics["stats"] = {}\
        
        # CPU times
        with self._time_operation("cpu_times"):
            try:
                cpu_times = psutil.cpu_times()
                total_time = sum([getattr(cpu_times, field) for field in cpu_times._fields])
                
                metrics['times'] = {
                    'user': round(cpu_times.user, 2),
                    'system': round(cpu_times.system, 2),
                    'idle': round(cpu_times.idle, 2),
                    'iowait': round(getattr(cpu_times, 'iowait', 0), 2),
                    'irq': round(getattr(cpu_times, 'irq', 0), 2),
                    'softirq': round(getattr(cpu_times, 'softirq', 0), 2),
                    'steal': round(getattr(cpu_times, 'steal', 0), 2),
                    'guest': round(getattr(cpu_times, 'guest', 0), 2)
                }
                
                # Calculate percentages
                if total_time > 0:
                    metrics['times_percent'] = {
                        field: round((getattr(cpu_times, field) / total_time) * 100, 2)
                        for field in cpu_times._fields
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get CPU times: {e}")
                metrics['times'] = {}

        # Process and thread counts
        with self._time_operation("process_count"):
            try:
                metrics['process_count'] = len(psutil.pids())
                
                # Get thread count (more expensive operation)
                thread_count = 0
                for proc in psutil.process_iter(['num_threads']):
                    try:
                        thread_count += proc.info['num_threads']
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                metrics['thread_count'] = thread_count
                
            except Exception as e:
                logger.warning(f"Failed to get process/thread count: {e}")
                metrics['process_count'] = 0
                metrics['thread_count'] = 0
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                cpu_temps = temps.get('coretemp', temps.get('cpu_thermal', []))
                if cpu_temps:
                    metrics['temperature'] = {
                        'current': round(cpu_temps[0].current, 1),
                        'high': round(cpu_temps[0].high, 1) if cpu_temps[0].high else None,
                        'critical': round(cpu_temps[0].critical, 1) if cpu_temps[0].critical else None
                    }
        except Exception:
            # Not available on all platforms
            pass
        
        return metrics
    
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Return CPU-specific error metrics"""
        return {
            'usage_percent': 0.0,
            'per_core_percent': [0.0] * self._cpu_count_logical,
            'count': self._cpu_count_physical,
            'count_logical': self._cpu_count_logical,
            'frequency': {'current': 0.0, 'min': 0.0, 'max': 0.0},
            'load_average': {'1m': 0.0, '5m': 0.0, '15m': 0.0},
            'error': True,
            'error_message': error_message,
            '_metadata': {
                'collector': self.name,
                'error': True
            }
        }