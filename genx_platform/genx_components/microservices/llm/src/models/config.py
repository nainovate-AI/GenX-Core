"""
genx_platform/genx_components/microservices/llm/src/models/config.py
LLM Service Configuration
"""
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings

from genx_components.common.config import BaseServiceConfig
from genx_components.common.hardware_detector import BackendType


class LLMServiceConfig(BaseServiceConfig):
    """Configuration for LLM microservice"""
    
    # Add this to fix the Pydantic warnings
    model_config = {'protected_namespaces': ()}
    
    # Service name override
    service_name: str = Field("llm-service", env='SERVICE_NAME')
    
    # Model configuration
    default_model_id: str = Field("gpt2", env='DEFAULT_MODEL_ID')
    model_cache_dir: str = Field("~/.cache/opea/models", env='MODEL_CACHE_DIR')
    
    # Backend configuration
    backend_type: Optional[str] = Field(None, env='BACKEND_TYPE')
    auto_select_backend: bool = Field(True, env='AUTO_SELECT_BACKEND')
    
    # Model loading configuration
    trust_remote_code: bool = Field(False, env='TRUST_REMOTE_CODE')
    load_in_8bit: bool = Field(False, env='LOAD_IN_8BIT')
    load_in_4bit: bool = Field(False, env='LOAD_IN_4BIT')
    device_map: str = Field("auto", env='DEVICE_MAP')
    
    # Request limits
    max_batch_size: int = Field(32, env='MAX_BATCH_SIZE')
    max_concurrent_requests: int = Field(10, env='MAX_CONCURRENT_REQUESTS')
    request_timeout_seconds: int = Field(300, env='REQUEST_TIMEOUT_SECONDS')
    
    # Model-specific configs
    model_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('backend_type', pre=True)
    def validate_backend_type(cls, v):
        # Handle empty string as None
        if v == '' or v is None:
            return None
            
        # Validate non-empty values
        valid_backends = [b.value for b in BackendType]
        if v not in valid_backends:
            raise ValueError(f"Invalid backend type. Must be one of: {valid_backends}")
        return v