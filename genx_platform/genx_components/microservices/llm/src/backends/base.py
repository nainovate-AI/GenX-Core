"""
genx_platform/genx_components/microservices/llm/src/backends/base.py
Base Backend Interface for LLM Service
All LLM backends must implement this interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio

from genx_components.microservices.grpc import llm_service_pb2


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = None
    num_return_sequences: int = 1
    seed: Optional[int] = None
    response_format: str = "text"
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.extra_params is None:
            self.extra_params = {}

    @classmethod
    def from_proto(cls, proto_config: llm_service_pb2.GenerationConfig) -> 'GenerationConfig':
        """Create from protobuf message"""
        return cls(
            max_tokens=proto_config.max_tokens or 512,
            temperature=proto_config.temperature or 0.7,
            top_p=proto_config.top_p or 0.95,
            top_k=proto_config.top_k or 50,
            repetition_penalty=proto_config.repetition_penalty or 1.0,
            stop_sequences=list(proto_config.stop_sequences),
            num_return_sequences=proto_config.num_return_sequences or 1,
            seed=proto_config.seed if proto_config.HasField('seed') else None,
            response_format=proto_config.response_format or "text",
            extra_params=dict(proto_config.extra_params) if proto_config.extra_params else {}
        )


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_id: str
    provider: str
    version: str
    capabilities: List[str]
    hardware_requirements: Dict[str, Any]
    is_loaded: bool = False
    is_available: bool = False
    current_load: float = 0.0


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.model_info: Optional[ModelInfo] = None
        self._is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the backend and load the model
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            system_prompt: Optional system prompt
            messages: Optional conversation history
            
        Returns:
            Dictionary containing:
                - text: Generated text
                - tokens_used: Token usage information
                - finish_reason: Why generation stopped
                - model_id: Model that was used
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation
        
        Yields:
            Text chunks as they are generated
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate prompt for safety and token limits
        
        Returns:
            Dictionary with validation results
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check backend health
        
        Returns:
            Dictionary with health status
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized"""
        return self._is_initialized
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get model information"""
        return self.model_info