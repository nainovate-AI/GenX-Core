# services/proxy_service/src/main.py (Updated)

import logging
from typing import Dict, Any

from genx_components.common.base_service import GenxMicroservice
from services.proxy_service.src.grpc import proxy_service_pb2_grpc
from .config import ProxyServiceConfig
from .api.grpc_handlers import ProxyServicer
from .clients.llm_client import LLMServiceClient

logger = logging.getLogger(__name__)


class ProxyService(GenxMicroservice):
    """
    Central Proxy Service - Orchestrates all operations
    """
    
    def __init__(self):
        # Load proxy service config
        config = ProxyServiceConfig()
        super().__init__(config)
        
        # Initialize service components
        self.grpc_servicer = None
        self.llm_client = None
        
    async def initialize(self):
        """Initialize proxy service components"""
        logger.info("Initializing Proxy Service components")
        
        # Initialize LLM client
        self.llm_client = LLMServiceClient(
            endpoint=self.config.llm_service_endpoint,
            timeout=self.config.llm_service_timeout
        )
        
        try:
            await self.llm_client.connect()
        except Exception as e:
            logger.warning(f"Could not connect to LLM service: {e}")
            # Continue anyway - we'll handle errors in requests
        
        # Create gRPC servicer with clients
        self.grpc_servicer = ProxyServicer(llm_client=self.llm_client)
        
        logger.info("Proxy Service initialized successfully")
        
    async def add_grpc_services(self, server):
        """Add gRPC services to the server"""
        logger.info("Adding Proxy Service gRPC handlers")
        
        # Add the proxy service
        proxy_service_pb2_grpc.add_ProxyServiceServicer_to_server(
            self.grpc_servicer, server
        )
        
        logger.info("Proxy Service gRPC handlers added")
        
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up Proxy Service")
        
        # Close client connections
        if self.llm_client:
            await self.llm_client.disconnect()
        
    async def check_health(self) -> Dict[str, Any]:
        """Service-specific health checks"""
        health_status = {
            "proxy_status": "healthy",
            "llm_service_endpoint": self.config.llm_service_endpoint,
            "grpc_servicer": "initialized" if self.grpc_servicer else "not_initialized"
        }
        
        # Check LLM service connection
        if self.llm_client and self.llm_client.stub:
            try:
                # Quick health check
                await self.llm_client.list_models()
                health_status["llm_service_connection"] = "connected"
            except:
                health_status["llm_service_connection"] = "disconnected"
        else:
            health_status["llm_service_connection"] = "not_initialized"
        
        return health_status