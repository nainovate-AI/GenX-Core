# services/proxy_service/src/clients/llm_client.py

import logging
import grpc
from typing import Optional, Dict, Any

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc
)

logger = logging.getLogger(__name__)


class LLMServiceClient:
    """Client for communicating with LLM Service"""
    
    def __init__(self, endpoint: str, timeout: int = 30):
        self.endpoint = endpoint
        self.timeout = timeout
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[llm_service_pb2_grpc.LLMServiceStub] = None
        
    async def connect(self):
        """Establish connection to LLM service"""
        logger.info(f"Connecting to LLM service at {self.endpoint}")
        self.channel = grpc.aio.insecure_channel(self.endpoint)
        self.stub = llm_service_pb2_grpc.LLMServiceStub(self.channel)
        
        # Test connection with a list models request
        try:
            request = llm_service_pb2.ListModelsRequest(
                metadata=common_pb2.RequestMetadata(
                    request_id="connection-test"
                )
            )
            await self.stub.ListModels(request, timeout=5)
            logger.info("Successfully connected to LLM service")
        except Exception as e:
            logger.error(f"Failed to connect to LLM service: {e}")
            raise
    
    async def disconnect(self):
        """Close connection"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from LLM service")
    
    async def generate_text(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Call LLM service to generate text"""
        
        if not self.stub:
            raise RuntimeError("LLM service not connected")
        
        request = llm_service_pb2.GenerateRequest(
            metadata=common_pb2.RequestMetadata(
                request_id=f"proxy-gen-{id(self)}",
                user_id="proxy-service"  # Add user_id
            ),
            prompt=prompt,
            config=llm_service_pb2.GenerationConfig(
                max_tokens=max_tokens,
                temperature=temperature
            )
        )
        
        if model_id:
            request.model_id = model_id
        
        try:
            response = await self.stub.Generate(request, timeout=self.timeout)
            
            return {
                "text": response.text,
                "tokens": response.usage,
                "model_id": response.model_id,
                "error": response.error if response.HasField("error") else None
            }
        except grpc.RpcError as e:
            logger.error(f"LLM service call failed: {e}")
            raise

    async def list_models(self) -> list:
        """Get list of available models from LLM service"""
        if not self.stub:
            raise RuntimeError("LLM service not connected")
            
        request = llm_service_pb2.ListModelsRequest(
            metadata=common_pb2.RequestMetadata(
                request_id=f"proxy-list-{id(self)}",
                user_id="proxy-service"  # Add user_id
            )
        )
        
        try:
            response = await self.stub.ListModels(request, timeout=self.timeout)
            return response.models
        except grpc.RpcError as e:
            logger.error(f"LLM service list models failed: {e}")
            raise