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
    service_version: str = Field('0.1.0', env='SERVICE_VERSION')
    service_port: int = Field(50051, env='SERVICE_PORT')
    
    # Environment
    environment: str = Field('development', env='ENVIRONMENT')
    debug: bool = Field(False, env='DEBUG')
    
    # gRPC Configuration
    grpc_max_workers: int = Field(10, env='GRPC_MAX_WORKERS')
    grpc_max_message_length: int = Field(100 * 1024 * 1024, env='GRPC_MAX_MESSAGE_LENGTH')  # 100MB
    
    # Observability
    telemetry_enabled: bool = Field(True, env='TELEMETRY_ENABLED')
    telemetry_endpoint: str = Field('http://localhost:4317', env='TELEMETRY_ENDPOINT')
    metrics_port: int = Field(9090, env='METRICS_PORT')
    
    # Service Discovery
    registry_enabled: bool = Field(False, env='REGISTRY_ENABLED')
    registry_endpoint: str = Field('http://localhost:8500', env='REGISTRY_ENDPOINT')
    
    class Config:
        env_file = '.env'
        case_sensitive = False