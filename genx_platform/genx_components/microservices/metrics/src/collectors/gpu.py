"""
GPU Metrics Collector
Collects NVIDIA GPU metrics using GPUtil and pynvml
"""
from typing import Dict, Any, List, Optional
import platform
import os
import sys

from .base import BaseCollector

current_file = os.path.abspath(__file__)
collectors_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(collectors_dir)
sys.path.insert(0, src_dir)

from genx_components.microservices.metrics.src.utils.logger import setup_logging

logger = setup_logging(__name__)

# Try to import GPU libraries
GPU_AVAILABLE = False
NVIDIA_ML_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
    logger.info("GPUtil available for GPU monitoring")
except ImportError:
    logger.warning("GPUtil not available - GPU metrics disabled")

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
    logger.info("pynvml available for detailed NVIDIA metrics")
except ImportError:
    logger.warning("pynvml not available - detailed GPU metrics disabled")


class GPUCollector(BaseCollector):
    """
    Collects GPU metrics, primarily for NVIDIA GPUs
    """
    
    def __init__(self):
        super().__init__("gpu")
        self._gpu_count = 0
        self._gpu_list = []
        self._nvml_initialized = False
        
    async def _initialize(self) -> None:
        """Initialize GPU collector"""
        if not GPU_AVAILABLE:
            logger.info("GPU monitoring not available - GPUtil not installed")
            return
        
        try:
            # Detect GPUs
            self._gpu_list = GPUtil.getGPUs()
            self._gpu_count = len(self._gpu_list)
            
            if self._gpu_count == 0:
                logger.info("No GPUs detected")
                return
            
            # Initialize NVIDIA ML if available
            if NVIDIA_ML_AVAILABLE and self._gpu_count > 0:
                try:
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                    
                    # Get driver version
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    logger.info(f"NVIDIA driver version: {driver_version}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize NVIDIA ML: {e}")
                    self._nvml_initialized = False
            
            # Log GPU information
            for gpu in self._gpu_list:
                logger.info(
                    f"Detected GPU: {gpu.name}",
                    gpu_id=gpu.id,
                    memory_total_mb=gpu.memoryTotal,
                    driver=gpu.driver
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize GPU collector: {e}")
            self._gpu_count = 0
    
    async def _collect(self) -> Dict[str, Any]:
        """Collect GPU metrics"""
        if not GPU_AVAILABLE or self._gpu_count == 0:
            return {'gpus': [], 'available': False, 'count': 0}
        
        metrics = {
            'available': True,
            'count': self._gpu_count,
            'gpus': []
        }
        
        try:
            # Get updated GPU list
            gpu_list = GPUtil.getGPUs()
            
            for gpu in gpu_list:
                gpu_metrics = await self._collect_single_gpu(gpu)
                metrics['gpus'].append(gpu_metrics)
            
            # Add aggregate metrics
            if metrics['gpus']:
                metrics['aggregate'] = self._calculate_aggregate_metrics(metrics['gpus'])
            
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _collect_single_gpu(self, gpu) -> Dict[str, Any]:
        """Collect metrics for a single GPU"""
        gpu_data = {
            'id': gpu.id,
            'name': gpu.name,
            'uuid': gpu.uuid,
            'driver_version': gpu.driver,
            
            # Basic metrics from GPUtil
            'load_percent': round(gpu.load * 100, 2),
            'memory': {
                'total': gpu.memoryTotal * 1024 * 1024,  # Convert MB to bytes
                'used': gpu.memoryUsed * 1024 * 1024,
                'free': gpu.memoryFree * 1024 * 1024,
                'percent': round((gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0, 2),
                'total_mb': gpu.memoryTotal,
                'used_mb': gpu.memoryUsed,
                'free_mb': gpu.memoryFree
            },
            'temperature': gpu.temperature,
            
            # For backward compatibility
            'usage_percent': round(gpu.load * 100, 2)
        }
        
        # Get detailed metrics from NVIDIA ML if available
        if self._nvml_initialized:
            detailed_metrics = await self._get_detailed_nvidia_metrics(gpu.id)
            gpu_data.update(detailed_metrics)
        
        return gpu_data
    
    async def _get_detailed_nvidia_metrics(self, gpu_id: int) -> Dict[str, Any]:
        """Get detailed metrics using NVIDIA ML"""
        detailed = {}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Power metrics
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                detailed['power'] = {
                    'current_watts': round(power, 2),
                    'limit_watts': round(power_limit, 2),
                    'percent': round((power / power_limit * 100) if power_limit > 0 else 0, 2)
                }
            except pynvml.NVMLError:
                logger.debug(f"Power metrics not available for GPU {gpu_id}")
            
            # Clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                detailed['clocks'] = {
                    'graphics_mhz': gpu_clock,
                    'memory_mhz': mem_clock
                }
                
                # Max clocks
                max_gpu_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                detailed['clocks']['graphics_max_mhz'] = max_gpu_clock
                detailed['clocks']['memory_max_mhz'] = max_mem_clock
                
            except pynvml.NVMLError:
                logger.debug(f"Clock metrics not available for GPU {gpu_id}")
            
            # PCIe information
            try:
                pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                detailed['pcie'] = {
                    'generation': pcie_gen,
                    'width': pcie_width
                }
            except pynvml.NVMLError:
                logger.debug(f"PCIe metrics not available for GPU {gpu_id}")
            
            # Utilization rates
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                detailed['utilization'] = {
                    'gpu_percent': util.gpu,
                    'memory_percent': util.memory
                }
            except pynvml.NVMLError:
                logger.debug(f"Utilization metrics not available for GPU {gpu_id}")
            
            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                detailed['fan_speed_percent'] = fan_speed
            except pynvml.NVMLError:
                logger.debug(f"Fan speed not available for GPU {gpu_id}")
            
            # Compute mode
            try:
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                compute_mode_str = {
                    pynvml.NVML_COMPUTEMODE_DEFAULT: "Default",
                    pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
                    pynvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
                    pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process"
                }.get(compute_mode, "Unknown")
                detailed['compute_mode'] = compute_mode_str
            except pynvml.NVMLError:
                pass
            
            # Performance state
            try:
                perf_state = pynvml.nvmlDeviceGetPerformanceState(handle)
                detailed['performance_state'] = f"P{perf_state}"
            except pynvml.NVMLError:
                pass
            
            # Processes using GPU
            try:
                processes = []
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    processes.append({
                        'pid': proc.pid,
                        'memory_mb': self._bytes_to_mb(proc.usedGpuMemory),
                        'type': 'compute'
                    })
                
                # Also get graphics processes
                graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                for proc in graphics_procs:
                    processes.append({
                        'pid': proc.pid,
                        'memory_mb': self._bytes_to_mb(proc.usedGpuMemory),
                        'type': 'graphics'
                    })
                
                detailed['processes'] = processes
                detailed['process_count'] = len(processes)
                
            except pynvml.NVMLError:
                logger.debug(f"Process information not available for GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Error getting detailed GPU metrics: {e}")
        
        return detailed
    
    def _calculate_aggregate_metrics(self, gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all GPUs"""
        if not gpus:
            return {}
        
        total_memory = sum(gpu['memory']['total'] for gpu in gpus)
        used_memory = sum(gpu['memory']['used'] for gpu in gpus)
        
        aggregate = {
            'avg_load_percent': round(sum(gpu['load_percent'] for gpu in gpus) / len(gpus), 2),
            'max_load_percent': max(gpu['load_percent'] for gpu in gpus),
            'total_memory': total_memory,
            'total_memory_used': used_memory,
            'total_memory_percent': round((used_memory / total_memory * 100) if total_memory > 0 else 0, 2),
            'avg_temperature': round(sum(gpu['temperature'] for gpu in gpus) / len(gpus), 1),
            'max_temperature': max(gpu['temperature'] for gpu in gpus)
        }
        
        # Add power metrics if available
        if any('power' in gpu for gpu in gpus):
            total_power = sum(gpu.get('power', {}).get('current_watts', 0) for gpu in gpus)
            total_power_limit = sum(gpu.get('power', {}).get('limit_watts', 0) for gpu in gpus)
            aggregate['total_power_watts'] = round(total_power, 2)
            aggregate['total_power_limit_watts'] = round(total_power_limit, 2)
        
        return aggregate
    
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Return GPU-specific error metrics"""
        return {
            'available': False,
            'count': 0,
            'gpus': [],
            'error': True,
            'error_message': error_message,
            '_metadata': {
                'collector': self.name,
                'error': True
            }
        }
    
    async def cleanup(self):
        """Cleanup GPU resources"""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("NVIDIA ML shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down NVIDIA ML: {e}")