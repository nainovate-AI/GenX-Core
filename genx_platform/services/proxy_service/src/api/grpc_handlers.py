# services/proxy_service/src/api/grpc_handlers.py (Updated)

import logging
from typing import Any, Optional
import grpc
import uuid

from genx_components.common.grpc import common_pb2
from services.proxy_service.src.grpc import (
    proxy_service_pb2,
    proxy_service_pb2_grpc
)

logger = logging.getLogger(__name__)


class ProxyServicer(proxy_service_pb2_grpc.ProxyServiceServicer):
    """gRPC service implementation for Proxy Service"""
    
    def __init__(self, llm_client=None):
        self.service_name = "proxy-service"
        self.llm_client = llm_client
        
    async def GenerateText(
        self,
        request: proxy_service_pb2.GenerateTextRequest,
        context: grpc.aio.ServicerContext
    ) -> proxy_service_pb2.GenerateTextResponse:
        """Generate text - now using real LLM service"""
        logger.info(f"GenerateText request for model: {request.model_name}")
        
        # Use default values if not provided
        max_tokens = request.max_tokens if request.max_tokens > 0 else 100
        temperature = request.temperature if request.temperature > 0 else 0.7
        
        try:
            # Call LLM service
            if self.llm_client and self.llm_client.stub:
                result = await self.llm_client.generate_text(
                    prompt=request.prompt,
                    model_id=request.model_name if request.model_name else None,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                response = proxy_service_pb2.GenerateTextResponse(
                    metadata=self._create_response_metadata(request.metadata.request_id),
                    generated_text=result["text"],
                    model_used=result["model_id"],
                    token_usage=result["tokens"]
                )
            else:
                # Fallback if LLM service not available
                response = proxy_service_pb2.GenerateTextResponse(
                    metadata=self._create_response_metadata(request.metadata.request_id),
                    generated_text=f"[LLM Service Unavailable] Echo: {request.prompt}",
                    model_used="fallback"
                )
                response.token_usage.CopyFrom(common_pb2.TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ))
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            await context.set_code(grpc.StatusCode.INTERNAL)
            await context.set_details(str(e))
            raise
        
        return response
    
    async def ListModels(
        self,
        request: proxy_service_pb2.ListModelsRequest,
        context: grpc.aio.ServicerContext
    ) -> proxy_service_pb2.ListModelsResponse:
        """List available models from LLM service"""
        logger.info("ListModels request")
        
        models = []
        
        try:
            if self.llm_client and self.llm_client.stub:
                # Get models from LLM service
                llm_models = await self.llm_client.list_models()
                
                for model in llm_models:
                    # Convert from LLM service format to proxy format
                    model_name = model.model_id.split('_')[0] if '_' in model.model_id else model.model_id

                    status = "loaded"  # Simplified - could check actual status
                    backend = model.provider if hasattr(model, 'provider') else "unknown"
                    
                    models.append(proxy_service_pb2.ModelInfo(
                        name=model_name,
                        status=status,
                        backend=backend
                    ))
            else:
                # Return empty list if LLM service not available
                logger.warning("LLM service not available for listing models")
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            # Return empty list instead of failing
            
        return proxy_service_pb2.ListModelsResponse(
            metadata=self._create_response_metadata(
                request.metadata.request_id if request.metadata else str(uuid.uuid4())
            ),
            models=models
        )
    
    async def HealthCheck(
        self,
        request: proxy_service_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> proxy_service_pb2.HealthCheckResponse:
        """Health check"""
        services = {
            "proxy_service": "healthy"
        }
        
        # Check LLM service
        if self.llm_client and self.llm_client.stub:
            try:
                await self.llm_client.list_models()
                services["llm_service"] = "healthy"
            except:
                services["llm_service"] = "unhealthy"
        else:
            services["llm_service"] = "not_connected"
            
        return proxy_service_pb2.HealthCheckResponse(
            status="healthy",
            services=services
        )
    
    def _create_response_metadata(self, request_id: str) -> common_pb2.ResponseMetadata:
        """Create response metadata"""
        import time
        metadata = common_pb2.ResponseMetadata(
            request_id=request_id or str(uuid.uuid4()),
            service_name=self.service_name,
            service_version="0.1.0"
        )
        return metadata