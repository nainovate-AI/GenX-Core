"""
genx_platform/genx_components/common/config.py
OPEA Platform Configuration Management
Handles environment-based configuration for all microservices
"""
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings  # Changed from pydantic import BaseSettings
from pydantic import Field  # Field stays in pydantic
import os


class BaseServiceConfig(BaseSettings):
    """Base configuration for all OPEA microservices"""
    
    # Service identification
    service_name: str = Field(..., env='SERVICE_NAME')
    service_version: str = Field(default='0.1.0', env='SERVICE_VERSION')
    service_port: int = Field(default=50051, env='SERVICE_PORT')
    
    # Environment
    environment: str = Field(default='development', env='ENVIRONMENT')
    debug: bool = Field(default=False, env='DEBUG')
    
    # gRPC Configuration
    grpc_max_workers: int = Field(default=10, env='GRPC_MAX_WORKERS')
    grpc_max_message_length: int = Field(default=100 * 1024 * 1024, env='GRPC_MAX_MESSAGE_LENGTH')  # 100MB
    
    # Observability
    telemetry_enabled: bool = Field(default=True, env='TELEMETRY_ENABLED')
    telemetry_endpoint: str = Field(default='http://localhost:4317', env='TELEMETRY_ENDPOINT')
    metrics_port: int = Field(default=9090, env='METRICS_PORT')
    
    # Service Discovery
    registry_enabled: bool = Field(default=False, env='REGISTRY_ENABLED')
    registry_endpoint: str = Field(default='http://localhost:8500', env='REGISTRY_ENDPOINT')
    
    class Config:
        # Allow .env file to be loaded
        env_file = '.env'
        # Case insensitive environment variables
        case_sensitive = False
        # Allow extra fields from environment without validation errors
        extra = 'ignore'
        # Validate default values
        validate_default = True
        # Use enum values (if any enums are used)
        use_enum_values = True


class LLMServiceConfig(BaseServiceConfig):
    """Configuration for LLM Service"""
    service_name: str = Field(default='llm-service', env='SERVICE_NAME')
    service_port: int = Field(default=50052, env='SERVICE_PORT')
    
    # Model configuration
    default_model: str = Field(default='gpt2', env='DEFAULT_MODEL')
    model_cache_dir: str = Field(default='/models', env='MODEL_CACHE_DIR')
    
    # Inference settings
    max_tokens: int = Field(default=512, env='MAX_TOKENS')
    temperature: float = Field(default=0.7, env='TEMPERATURE')
    

class EmbeddingServiceConfig(BaseServiceConfig):
    """Configuration for Embedding Service"""
    service_name: str = Field(default='embedding-service', env='SERVICE_NAME')
    service_port: int = Field(default=50053, env='SERVICE_PORT')
    
    # Model configuration
    default_model: str = Field(default='sentence-transformers/all-MiniLM-L6-v2', env='DEFAULT_MODEL')
    batch_size: int = Field(default=32, env='BATCH_SIZE')


# Helper function to get config based on service type
def get_service_config(service_type: str = 'base') -> BaseServiceConfig:
    """Factory function to get appropriate service configuration"""
    configs = {
        'base': BaseServiceConfig,
        'llm': LLMServiceConfig,
        'embedding': EmbeddingServiceConfig,
    }
    
    config_class = configs.get(service_type, BaseServiceConfig)
    return config_class()