"""
Backend Factory for LLM Service
Manages creation and selection of appropriate backends with device support
"""
import logging
from typing import Dict, Type, Optional, Any, List
from enum import Enum

from genx_components.common.hardware_detector import HardwareDetector, BackendType
from ..backends.base import LLMBackend
from ..backends.transformers_backend import TransformersBackend

# Conditional imports for optional backends
try:
    from ..backends.mlx_backend import MLXBackend
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX backend not available")

try:
    from ..backends.vllm_backend import VLLMBackend
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM backend not available")

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating LLM backends with device support"""
    
    # Registry of available backends
    _backends: Dict[str, Type[LLMBackend]] = {
        BackendType.TRANSFORMERS.value: TransformersBackend,
    }
    
    # Add optional backends if available
    if MLX_AVAILABLE:
        _backends[BackendType.MLX.value] = MLXBackend
    
    if VLLM_AVAILABLE:
        _backends[BackendType.VLLM.value] = VLLMBackend
    
    # Backend device compatibility
    _backend_device_support = {
        BackendType.TRANSFORMERS.value: ['cuda', 'cpu', 'mps', 'auto'],
        BackendType.MLX.value: ['mps', 'auto'],  # MLX only works on Apple Silicon
        BackendType.VLLM.value: ['cuda', 'auto'],  # vLLM only works on NVIDIA GPUs
        # Future backends
        # BackendType.ONNX.value: ['cpu', 'cuda', 'auto'],
        # BackendType.TGI.value: ['cuda', 'auto'],
    }
    
    @classmethod
    def create_backend(
        cls,
        backend_type: Optional[str],
        model_id: str,
        device: str = "auto",
        auto_select: bool = True,
        **kwargs
    ) -> LLMBackend:
        """
        Create an LLM backend with device support
        
        Args:
            backend_type: Specific backend type to use (optional)
            model_id: Model identifier
            device: Target device (cuda, cpu, mps, auto)
            auto_select: Auto-select best backend if type not specified
            **kwargs: Additional backend-specific arguments
            
        Returns:
            LLMBackend instance
        """
        # Add device to kwargs
        kwargs['device'] = device
        
        # Determine backend type
        if backend_type:
            selected_backend = backend_type
            
            # Validate device compatibility
            if not cls._is_device_compatible(selected_backend, device):
                raise ValueError(
                    f"Backend {selected_backend} does not support device {device}. "
                    f"Supported devices: {cls._backend_device_support.get(selected_backend, [])}"
                )
        elif auto_select:
            selected_backend = cls._auto_select_backend(model_id, device)
        else:
            selected_backend = BackendType.TRANSFORMERS.value
        
        # Validate backend type
        if selected_backend not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend type: {selected_backend}. "
                f"Available backends: {available}"
            )
        
        # Resolve device if auto
        if device == "auto":
            resolved_device = cls._resolve_auto_device(selected_backend)
            kwargs['device'] = resolved_device
            logger.info(f"Auto-resolved device to: {resolved_device}")
        
        # Create backend instance
        backend_class = cls._backends[selected_backend]
        logger.info(f"Creating {selected_backend} backend for model {model_id} on device {kwargs['device']}")
        
        return backend_class(model_id=model_id, **kwargs)
    
    @classmethod
    def _is_device_compatible(cls, backend_type: str, device: str) -> bool:
        """Check if a backend supports a specific device"""
        if device == "auto":
            return True  # Auto is always compatible
        
        supported_devices = cls._backend_device_support.get(backend_type, [])
        return device in supported_devices
    
    @classmethod
    def _resolve_auto_device(cls, backend_type: str) -> str:
        """Resolve 'auto' device to a specific device based on backend and hardware"""
        hardware_info = HardwareDetector.detect_platform()
        
        if backend_type == BackendType.MLX.value:
            # MLX only works on MPS
            return "mps"
        
        elif backend_type == BackendType.VLLM.value:
            # vLLM only works on CUDA
            if hardware_info['gpu_available'] and hardware_info['hardware_type'].value == "nvidia_gpu":
                return "cuda"
            else:
                raise RuntimeError("vLLM backend requires NVIDIA GPU")
        
        elif backend_type == BackendType.TRANSFORMERS.value:
            # Transformers supports multiple devices, choose best available
            if hardware_info['gpu_available']:
                if hardware_info['hardware_type'].value == "nvidia_gpu":
                    return "cuda"
                elif hardware_info['hardware_type'].value == "apple_silicon":
                    return "mps"
            return "cpu"
        
        # Default to CPU for unknown backends
        return "cpu"
    
    @classmethod
    def _auto_select_backend(cls, model_id: str, device: str = "auto") -> str:
        """
        Auto-select the best backend based on hardware, model, and device preference
        
        Args:
            model_id: Model identifier
            device: Device preference
            
        Returns:
            Selected backend type
        """
        # Detect hardware
        hardware_info = HardwareDetector.detect_platform()
        logger.info(f"Detected hardware: {hardware_info['hardware_type'].value}")
        
        # If specific device requested, filter backends by device support
        if device != "auto":
            compatible_backends = []
            for backend, supported_devices in cls._backend_device_support.items():
                if device in supported_devices and backend in cls._backends:
                    compatible_backends.append(backend)
            
            if not compatible_backends:
                raise ValueError(f"No backend available for device {device}")
            
            # Prefer in order: vLLM > MLX > Transformers
            if BackendType.VLLM.value in compatible_backends:
                return BackendType.VLLM.value
            elif BackendType.MLX.value in compatible_backends:
                return BackendType.MLX.value
            else:
                return compatible_backends[0]
        
        # Auto device selection based on hardware
        
        # Special handling for Apple Silicon
        if (hardware_info['hardware_type'].value == "apple_silicon" and 
            BackendType.MLX.value in cls._backends):
            # Check if model is compatible with MLX
            mlx_compatible_models = [
                "gpt2", "distilgpt2", "phi-2", "phi-3", "phi-4", "mistral", 
                "llama", "codellama", "gemma", "stable-lm", "qwen"
            ]
            if any(model in model_id.lower() for model in mlx_compatible_models):
                logger.info(f"Auto-selected MLX backend for Apple Silicon")
                return BackendType.MLX.value
        
        # Check for NVIDIA GPU and vLLM
        if (hardware_info['hardware_type'].value == "nvidia_gpu" and
            BackendType.VLLM.value in cls._backends):
            # vLLM is generally faster for larger models on NVIDIA GPUs
            large_model_indicators = ["7b", "13b", "30b", "65b", "70b", "llama", "falcon", "mpt"]
            if any(indicator in model_id.lower() for indicator in large_model_indicators):
                logger.info(f"Auto-selected vLLM backend for large model on NVIDIA GPU")
                return BackendType.VLLM.value
        
        # Get recommended backends from hardware detector
        recommendations = hardware_info['recommended_backends']
        
        # Check which recommended backends are available
        for backend in recommendations:
            if backend.value in cls._backends:
                logger.info(f"Auto-selected backend: {backend.value}")
                return backend.value
        
        # Fallback to transformers
        logger.info("Falling back to transformers backend")
        return BackendType.TRANSFORMERS.value
    
    @classmethod
    def register_backend(
        cls, 
        backend_type: str, 
        backend_class: Type[LLMBackend],
        supported_devices: List[str] = None
    ):
        """Register a new backend type"""
        cls._backends[backend_type] = backend_class
        
        if supported_devices:
            cls._backend_device_support[backend_type] = supported_devices
        else:
            cls._backend_device_support[backend_type] = ['auto']
        
        logger.info(f"Registered backend: {backend_type} with device support: {supported_devices}")
    
    @classmethod
    def list_backends(cls) -> Dict[str, Dict[str, Any]]:
        """List all available backends with their capabilities"""
        backends_info = {}
        
        for backend_type, backend_class in cls._backends.items():
            backends_info[backend_type] = {
                'class': backend_class.__name__,
                'available': True,
                'supported_devices': cls._backend_device_support.get(backend_type, [])
            }
        
        # Add info about unavailable backends
        if not MLX_AVAILABLE:
            backends_info['mlx'] = {
                'class': 'MLXBackend',
                'available': False,
                'supported_devices': ['mps'],
                'install_command': 'pip install mlx mlx-lm'
            }
        
        if not VLLM_AVAILABLE:
            backends_info['vllm'] = {
                'class': 'VLLMBackend',
                'available': False,
                'supported_devices': ['cuda'],
                'install_command': 'pip install vllm'
            }
        
        return backends_info
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get information about available devices"""
        hardware_info = HardwareDetector.detect_platform()
        
        device_info = {
            'available_devices': ['cpu'],  # CPU is always available
            'recommended_device': 'cpu',
            'hardware_type': hardware_info['hardware_type'].value
        }
        
        if hardware_info['gpu_available']:
            if hardware_info['hardware_type'].value == "nvidia_gpu":
                device_info['available_devices'].append('cuda')
                device_info['recommended_device'] = 'cuda'
                
                # Add GPU details
                import torch
                if torch.cuda.is_available():
                    device_info['cuda_devices'] = []
                    for i in range(torch.cuda.device_count()):
                        device_info['cuda_devices'].append({
                            'index': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        })
                        
            elif hardware_info['hardware_type'].value == "apple_silicon":
                device_info['available_devices'].append('mps')
                device_info['recommended_device'] = 'mps'
        
        return device_info