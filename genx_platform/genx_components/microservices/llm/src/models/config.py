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
    
    # Service name and port override
    service_name: str = Field("llm-service", env='SERVICE_NAME')
    service_port: int = Field(50053, env='SERVICE_PORT')
    
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

    preload_models: bool = Field(False, env='PRELOAD_MODELS')
    models_to_preload: List[str] = Field(default_factory=list, env='MODELS_TO_PRELOAD')
    
    # Request limits
    max_batch_size: int = Field(32, env='MAX_BATCH_SIZE')
    max_concurrent_requests: int = Field(10, env='MAX_CONCURRENT_REQUESTS')
    request_timeout_seconds: int = Field(300, env='REQUEST_TIMEOUT_SECONDS')

    # TGI (Text Generation Inference) specific settings
    tgi_server_url: Optional[str] = Field(None, env='TGI_SERVER_URL')
    tgi_external_server: bool = Field(True, env='TGI_EXTERNAL_SERVER')
    tgi_quantize: Optional[str] = Field(None, env='TGI_QUANTIZE')  # bitsandbytes, gptq, awq
    tgi_num_shard: int = Field(1, env='TGI_NUM_SHARD')
    tgi_max_input_length: int = Field(4096, env='TGI_MAX_INPUT_LENGTH')
    tgi_max_total_tokens: int = Field(8192, env='TGI_MAX_TOTAL_TOKENS')
    
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