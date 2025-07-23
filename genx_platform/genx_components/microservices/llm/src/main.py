"""
genx_platform/genx_components/microservices/llm/src/main.py
LLM Microservice Main Entry Point
"""
import asyncio
import logging
import os
from pathlib import Path

from genx_components.common.base_service import GenxMicroservice
from genx_components.microservices.grpc import llm_service_pb2_grpc

from .models.config import LLMServiceConfig
from .core.model_manager import ModelManager
from .service.llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMMicroservice(GenxMicroservice):
    """LLM Microservice implementation"""
    
    def __init__(self):
        # Load configuration
        config = LLMServiceConfig()
        super().__init__(config)
        
        # Service components
        self.model_manager = None
        self.llm_service = None
        
    async def initialize(self):
        """Initialize service-specific components"""
        try:
            logger.info("Initializing LLM microservice components")
            
            # Expand paths
            self.config.model_cache_dir = os.path.expanduser(self.config.model_cache_dir)
            Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize model manager
            self.model_manager = ModelManager({
                'default_model_id': self.config.default_model_id,
                'backend_type': self.config.backend_type,
                'auto_select_backend': self.config.auto_select_backend,
                'device_map': self.config.device_map,
                'trust_remote_code': self.config.trust_remote_code,
                'load_in_8bit': self.config.load_in_8bit,
                'load_in_4bit': self.config.load_in_4bit,
                'model_configs': self.config.model_configs,
                'max_loaded_models': 3
            })

            if self.config.preload_models and self.config.models_to_preload:
                try:
                    logger.info(f"Preloading models: {self.config.models_to_preload}")
                    await self.model_manager.preload_models(self.config.models_to_preload)
                    logger.info("Model preloading completed")
                except Exception as e:
                    logger.error(f"Failed to preload models: {e}")
                    logger.warning("Continuing without preloaded models")
            else:
                logger.info("Model preloading disabled - models will be loaded on demand")
            
            
            # Create service - this should not be None
            self.llm_service = LLMService(
                model_manager=self.model_manager,
                telemetry=self.telemetry
            )
            
            if self.llm_service is None:
                raise RuntimeError("Failed to create LLM service instance")
            
            logger.info("LLM microservice initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM microservice: {e}", exc_info=True)
            raise
        
    async def add_grpc_services(self, server):
        """Add gRPC services to the server"""
        if self.llm_service is None:
            raise RuntimeError("LLM service not initialized")
            
        llm_service_pb2_grpc.add_LLMServiceServicer_to_server(
            self.llm_service, server
        )
        logger.info("Added LLM gRPC service")
        
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up LLM microservice")
        
        if self.model_manager:
            await self.model_manager.cleanup_all()
            
    async def check_health(self) -> dict:
        """Service-specific health checks"""
        health_status = {
            "models_loaded": 0,
            "default_model_available": False
        }
        
        try:
            if self.model_manager:
                loaded_models = await self.model_manager.list_loaded_models()
                health_status["models_loaded"] = len(loaded_models)
                
                # Check if default model is available
                try:
                    default_backend = await self.model_manager.get_model()
                    if default_backend and default_backend.is_initialized:
                        health_status["default_model_available"] = True
                except Exception as e:
                    logger.warning(f"Default model not available: {e}")
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status["error"] = str(e)
            
        return health_status


async def main():
    """Main entry point"""
    service = LLMMicroservice()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())