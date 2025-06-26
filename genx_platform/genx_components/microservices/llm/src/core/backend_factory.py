"""
genx_platform/genx_components/microservices/llm/src/core/backend_factory.py
Backend Factory for LLM Service
Manages creation and selection of appropriate backends
"""
import logging
from typing import Dict, Type, Optional, Any
from enum import Enum

from genx_components.common.hardware_detector import HardwareDetector, BackendType
from ..backends.base import LLMBackend
from ..backends.transformers_backend import TransformersBackend

# Conditional import for MLX
try:
    from ..backends.mlx_backend import MLXBackend
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX backend not available")

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating LLM backends"""
    
    # Registry of available backends
    _backends: Dict[str, Type[LLMBackend]] = {
        BackendType.TRANSFORMERS.value: TransformersBackend,
        # We'll add more backends here later:
        # BackendType.VLLM.value: VLLMBackend,
        BackendType.MLX.value: MLXBackend,
        # BackendType.ONNX.value: ONNXBackend,
        # BackendType.CLOUD.value: CloudBackend,
    }
    
    @classmethod
    def create_backend(
        cls,
        backend_type: Optional[str],
        model_id: str,
        auto_select: bool = True,
        **kwargs
    ) -> LLMBackend:
        """
        Create an LLM backend
        
        Args:
            backend_type: Specific backend type to use (optional)
            model_id: Model identifier
            auto_select: Auto-select best backend if type not specified
            **kwargs: Additional backend-specific arguments
            
        Returns:
            LLMBackend instance
        """
        # Determine backend type
        if backend_type:
            selected_backend = backend_type
        elif auto_select:
            selected_backend = cls._auto_select_backend(model_id)
        else:
            selected_backend = BackendType.TRANSFORMERS.value
        
        # Validate backend type
        if selected_backend not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend type: {selected_backend}. "
                f"Available backends: {available}"
            )
        
        # Create backend instance
        backend_class = cls._backends[selected_backend]
        logger.info(f"Creating {selected_backend} backend for model {model_id}")
        
        return backend_class(model_id=model_id, **kwargs)
    
    @classmethod
    def _auto_select_backend(cls, model_id: str) -> str:
        """
        Auto-select the best backend based on hardware and model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Selected backend type
        """
        # Detect hardware
        hardware_info = HardwareDetector.detect_platform()
        logger.info(f"Detected hardware: {hardware_info['hardware_type'].value}")

        # Special handling for Apple Silicon
        if (hardware_info['hardware_type'].value == "apple_silicon" and 
            BackendType.MLX.value in cls._backends):
            # Check if model is compatible with MLX
            mlx_compatible_models = [
                "gpt2", "distilgpt2", "phi-2", "mistral", "llama", 
                "codellama", "gemma", "stable-lm"
            ]
            if any(model in model_id.lower() for model in mlx_compatible_models):
                logger.info(f"Auto-selected MLX backend for Apple Silicon")
                return BackendType.MLX.value
        
        # Get recommended backends
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
    def register_backend(cls, backend_type: str, backend_class: Type[LLMBackend]):
        """Register a new backend type"""
        cls._backends[backend_type] = backend_class
        logger.info(f"Registered backend: {backend_type}")
    
    @classmethod
    def list_backends(cls) -> Dict[str, Type[LLMBackend]]:
        """List all available backends"""
        return cls._backends.copy()