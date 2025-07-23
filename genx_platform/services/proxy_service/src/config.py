# services/proxy_service/src/config.py

from genx_components.common.config import BaseServiceConfig
from pydantic import Field

class ProxyServiceConfig(BaseServiceConfig):
    """Configuration for Proxy Service"""
    
    # Override service name
    service_name: str = Field("proxy-service", env='SERVICE_NAME')
    service_port: int = Field(8080, env='SERVICE_PORT')  # Different port
    
    # LLM Service client config
    llm_service_endpoint: str = Field("localhost:50053", env='LLM_SERVICE_ENDPOINT')
    llm_service_timeout: int = Field(30, env='LLM_SERVICE_TIMEOUT')