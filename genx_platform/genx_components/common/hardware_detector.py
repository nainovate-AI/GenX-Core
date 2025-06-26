"""
genx_platform/genx_components/common/hardware_detector.py
Hardware Detection Module
Detects available hardware and recommends optimal backends
"""
import platform
import subprocess
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
import os

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Supported hardware types"""
    NVIDIA_GPU = "nvidia_gpu"
    APPLE_SILICON = "apple_silicon"
    AMD_GPU = "amd_gpu"
    INTEL_CPU = "intel_cpu"
    AMD_CPU = "amd_cpu"
    GENERIC_CPU = "generic_cpu"


class BackendType(Enum):
    """Available backend types"""
    VLLM = "vllm"
    TGI = "tgi"
    TRANSFORMERS = "transformers"
    MLX = "mlx"
    ONNX = "onnx"
    CLOUD = "cloud"


class HardwareDetector:
    """Detects available hardware and recommends optimal backends"""
    
    @staticmethod
    def detect_platform() -> Dict[str, Any]:
        """
        Detect current hardware platform and capabilities
        
        Returns:
            Dictionary containing hardware info and recommendations
        """
        result = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "hardware_type": HardwareType.GENERIC_CPU,
            "gpu_available": False,
            "gpu_info": None,
            "cpu_info": HardwareDetector._get_cpu_info(),
            "memory_gb": HardwareDetector._get_memory_info(),
            "recommended_backends": []
        }
        
        # Detect GPU
        gpu_info = HardwareDetector._detect_gpu()
        if gpu_info:
            result["gpu_available"] = True
            result["gpu_info"] = gpu_info
            result["hardware_type"] = gpu_info["type"]
            
        # Get recommendations
        result["recommended_backends"] = HardwareDetector._get_backend_recommendations(result)
        
        return result
    
    @staticmethod
    def _detect_gpu() -> Optional[Dict[str, Any]]:
        """Detect GPU type and capabilities"""
        # We'll implement the actual detection logic next
        # For now, return None
        return None
    
    @staticmethod
    def _get_cpu_info() -> Dict[str, Any]:
        """Get CPU information"""
        return {
            "model": platform.processor(),
            "cores": os.cpu_count(),
        }
    
    @staticmethod
    def _get_memory_info() -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 0.0
    
    @staticmethod
    def _get_backend_recommendations(hardware_info: Dict[str, Any]) -> List[BackendType]:
        """Recommend backends based on hardware"""
        recommendations = []
        
        if hardware_info["hardware_type"] == HardwareType.NVIDIA_GPU:
            recommendations.extend([BackendType.VLLM, BackendType.TGI])
        elif hardware_info["hardware_type"] == HardwareType.APPLE_SILICON:
            recommendations.append(BackendType.MLX)
        
        # Always include transformers as fallback
        recommendations.append(BackendType.TRANSFORMERS)
        
        return recommendations
    
    # Add to the _detect_gpu method in HardwareDetector class
    @staticmethod
    def _detect_gpu() -> Optional[Dict[str, Any]]:
        """Detect GPU type and capabilities"""
        gpu_info = None
        
        # Check for Apple Silicon first
        if platform.system() == "Darwin":
            try:
                # Check if it's Apple Silicon
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.arm64"], 
                    capture_output=True, 
                    text=True
                )
                if result.stdout.strip() == "1":
                    # Get chip info
                    chip_info = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True
                    ).stdout.strip()
                    
                    gpu_info = {
                        "type": HardwareType.APPLE_SILICON,
                        "name": chip_info,
                        "memory_gb": HardwareDetector._get_memory_info(),  # Unified memory
                        "compute_capability": "metal",
                    }
                    return gpu_info
            except Exception:
                pass
        
        # Check for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "type": HardwareType.NVIDIA_GPU,
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "compute_capability": torch.cuda.get_device_capability(0),
                }
                return gpu_info
        except ImportError:
            pass
        
        # Add checks for AMD GPU, etc.
        
        return gpu_info