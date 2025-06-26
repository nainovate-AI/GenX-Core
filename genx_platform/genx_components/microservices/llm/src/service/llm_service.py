"""
genx_platform/genx_components/microservices/llm/src/service/llm_service.py
LLM gRPC Service Implementation
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any
import grpc
from google.protobuf import timestamp_pb2

from genx_components.common.grpc import common_pb2
from genx_components.microservices.grpc import (
    llm_service_pb2,
    llm_service_pb2_grpc,
)
from genx_components.common.telemetry import GenxTelemetry

from ..backends.base import GenerationConfig
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LLMService(llm_service_pb2_grpc.LLMServiceServicer):
    """gRPC service implementation for LLM operations"""
    
    def __init__(self, model_manager: ModelManager, telemetry: Optional[GenxTelemetry] = None):
        self.model_manager = model_manager
        self.telemetry = telemetry
        self.service_name = "llm-service"
        self.service_version = "0.1.0"
        
    async def Generate(
        self,
        request: llm_service_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.GenerateResponse:
        """Generate text completion"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        # Start tracing
        with self._trace_operation("generate", request_id, request.model_id or "default") as span:
            try:
                # Validate request
                if not request.prompt:
                    await self._set_error(context, "INVALID_ARGUMENT", "Prompt is required")
                    return self._create_error_response(request_id, "Prompt is required")
                
                # Get model backend
                backend = await self.model_manager.get_model(request.model_id)
                
                # Convert proto config to internal config
                config = GenerationConfig.from_proto(request.config)
                
                # Prepare messages if provided
                messages = None
                if request.messages:
                    messages = [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ]
                
                # Generate
                result = await backend.generate(
                    prompt=request.prompt,
                    config=config,
                    system_prompt=request.system_prompt if request.HasField('system_prompt') else None,
                    messages=messages
                )
                
                # Update token usage
                self.model_manager.update_token_usage(
                    result['model_id'],
                    result['tokens_used']['total_tokens']
                )
                
                # Create response
                response = llm_service_pb2.GenerateResponse(
                    metadata=self._create_response_metadata(request_id, start_time),
                    text=result['text'],
                    model_id=result['model_id'],
                    finish_reason=result.get('finish_reason', 'stop'),
                    usage=common_pb2.TokenUsage(
                        prompt_tokens=result['tokens_used']['prompt_tokens'],
                        completion_tokens=result['tokens_used']['completion_tokens'],
                        total_tokens=result['tokens_used']['total_tokens']
                    )
                )
                
                # Add alternatives if any
                if 'alternatives' in result:
                    response.alternatives.extend(result['alternatives'])
                
                # Record metrics
                if self.telemetry:
                    self.telemetry.record_metric(
                        "llm_tokens_generated",
                        result['tokens_used']['completion_tokens'],
                        {"model": result['model_id']}
                    )
                
                return response
                
            except Exception as e:
                logger.error(f"Generate failed: {e}")
                await self._set_error(context, "INTERNAL", str(e))
                return self._create_error_response(request_id, str(e))
    
    async def StreamGenerate(
        self,
        request: llm_service_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.StreamGenerateResponse:
        """Stream text generation"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        # Start tracing
        with self._trace_operation("stream_generate", request_id, request.model_id or "default") as span:
            try:
                # Validate request
                if not request.prompt:
                    await self._set_error(context, "INVALID_ARGUMENT", "Prompt is required")
                    yield self._create_stream_error_response("Prompt is required")
                    return
                
                # Get model backend
                backend = await self.model_manager.get_model(request.model_id)
                
                # Send initial metadata
                yield llm_service_pb2.StreamGenerateResponse(
                    metadata=self._create_response_metadata(request_id, start_time)
                )
                
                # Convert config
                config = GenerationConfig.from_proto(request.config)
                
                # Prepare messages
                messages = None
                if request.messages:
                    messages = [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ]
                
                # Stream generation
                cumulative_text = ""
                token_count = 0
                
                async for chunk in backend.stream_generate(
                    prompt=request.prompt,
                    config=config,
                    system_prompt=request.system_prompt if request.HasField('system_prompt') else None,
                    messages=messages
                ):
                    cumulative_text += chunk
                    token_count += 1
                    
                    yield llm_service_pb2.StreamGenerateResponse(
                        delta=chunk,
                        cumulative_text=cumulative_text,
                        is_final=False
                    )
                
                # Send final message with usage
                prompt_tokens = await backend.count_tokens(request.prompt)
                yield llm_service_pb2.StreamGenerateResponse(
                    is_final=True,
                    finish_reason="stop",
                    usage=common_pb2.TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=token_count,
                        total_tokens=prompt_tokens + token_count
                    )
                )
                
            except Exception as e:
                logger.error(f"StreamGenerate failed: {e}")
                yield self._create_stream_error_response(str(e))
    
    async def ListModels(
        self,
        request: llm_service_pb2.ListModelsRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.ListModelsResponse:
        """List available models"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        try:
            # Get loaded models
            loaded_models = await self.model_manager.list_loaded_models()
            
            # Create model info list
            models = []
            for model_data in loaded_models:
                model_info = common_pb2.ModelInfo(
                    model_id=model_data['model_id'],
                    provider=model_data['provider'],
                    metadata={
                        'loaded_at': str(model_data['loaded_at']),
                        'request_count': str(model_data['request_count'])
                    }
                )
                models.append(model_info)
            
            # Add available but not loaded models
            # This would come from a model registry in production
            available_models = [
                {"model_id": "gpt2", "provider": "transformers"},
                {"model_id": "distilgpt2", "provider": "transformers"},
                # Add more models as needed
            ]
            
            for model in available_models:
                if not any(m.model_id == model['model_id'] for m in models):
                    models.append(common_pb2.ModelInfo(
                        model_id=model['model_id'],
                        provider=model['provider']
                    ))
            
            return llm_service_pb2.ListModelsResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                models=models,
                default_model_id=self.model_manager.default_model_id
            )
            
        except Exception as e:
            logger.error(f"ListModels failed: {e}")
            await self._set_error(context, "INTERNAL", str(e))
            return llm_service_pb2.ListModelsResponse(
                metadata=self._create_response_metadata(request_id, start_time)
            )
    
    async def GetModel(
        self,
        request: llm_service_pb2.GetModelRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.GetModelResponse:
        """Get model information"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        try:
            # Get model backend
            backend = await self.model_manager.get_model(request.model_id)
            model_info = backend.get_model_info()
            
            if not model_info:
                await self._set_error(context, "NOT_FOUND", f"Model {request.model_id} not found")
                return llm_service_pb2.GetModelResponse(
                    metadata=self._create_response_metadata(request_id, start_time)
                )
            
            # Create response
            return llm_service_pb2.GetModelResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                model=common_pb2.ModelInfo(
                    model_id=model_info.model_id,
                    provider=model_info.provider,
                    version=model_info.version,
                    capabilities=model_info.capabilities,
                    hardware_requirements=common_pb2.HardwareRequirements(
                        min_gpu_memory_gb=model_info.hardware_requirements.get('min_gpu_memory_gb', 0),
                        min_ram_gb=model_info.hardware_requirements.get('min_ram_gb', 8),
                        gpu_required=model_info.hardware_requirements.get('gpu_required', False)
                    )
                ),
                status=llm_service_pb2.ModelStatus(
                    is_loaded=model_info.is_loaded,
                    is_available=model_info.is_available,
                    current_load=model_info.current_load
                )
            )
            
        except Exception as e:
            logger.error(f"GetModel failed: {e}")
            await self._set_error(context, "INTERNAL", str(e))
            return llm_service_pb2.GetModelResponse(
                metadata=self._create_response_metadata(request_id, start_time)
            )
    
    async def ValidatePrompt(
        self,
        request: llm_service_pb2.ValidatePromptRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.ValidatePromptResponse:
        """Validate a prompt"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        try:
            # Get model backend
            backend = await self.model_manager.get_model(request.model_id)
            
            # Validate prompt
            validation_result = await backend.validate_prompt(request.prompt)
            
            # Create response
            response = llm_service_pb2.ValidatePromptResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                is_valid=validation_result['is_valid']
            )
            
            # Add issues
            for issue in validation_result.get('issues', []):
                response.issues.append(
                    llm_service_pb2.ValidationIssue(
                        type=issue['type'],
                        severity=issue['severity'],
                        message=issue['message']
                    )
                )
            
            # Add token count if requested
            if request.count_tokens:
                response.token_count = validation_result.get('token_count', 0)
            
            return response
            
        except Exception as e:
            logger.error(f"ValidatePrompt failed: {e}")
            await self._set_error(context, "INTERNAL", str(e))
            return llm_service_pb2.ValidatePromptResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                is_valid=False
            )
    
    def _trace_operation(self, operation: str, request_id: str, model_id: str):
        """Create a trace context for an operation"""
        if self.telemetry:
            return self.telemetry.trace_operation(
                operation,
                {
                    "request_id": request_id,
                    "model_id": model_id,
                    "service": self.service_name
                }
            )
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def _create_response_metadata(
        self,
        request_id: str,
        start_time: float
    ) -> common_pb2.ResponseMetadata:
        """Create response metadata"""
        duration_ms = int((time.time() - start_time) * 1000)
        
        metadata = common_pb2.ResponseMetadata(
            request_id=request_id,
            duration_ms=duration_ms,
            service_name=self.service_name,
            service_version=self.service_version
        )
        
        # Set timestamp
        metadata.timestamp.CopyFrom(
            timestamp_pb2.Timestamp(seconds=int(time.time()))
        )
        
        return metadata
    
    def _create_error_response(
        self,
        request_id: str,
        error_message: str,
        error_code: str = "INTERNAL"
    ) -> llm_service_pb2.GenerateResponse:
        """Create an error response"""
        return llm_service_pb2.GenerateResponse(
            metadata=self._create_response_metadata(request_id, time.time()),
            error=common_pb2.ErrorDetail(
                code=error_code,
                message=error_message
            )
        )
    
    def _create_stream_error_response(
        self,
        error_message: str,
        error_code: str = "INTERNAL"
    ) -> llm_service_pb2.StreamGenerateResponse:
        """Create a streaming error response"""
        return llm_service_pb2.StreamGenerateResponse(
            error=common_pb2.ErrorDetail(
                code=error_code,
                message=error_message
            ),
            is_final=True
        )
    
    async def _set_error(
        self,
        context: grpc.aio.ServicerContext,
        code: str,
        message: str
    ):
        """Set gRPC error status"""
        if code == "INVALID_ARGUMENT":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, message)
        elif code == "NOT_FOUND":
            await context.abort(grpc.StatusCode.NOT_FOUND, message)
        else:
            await context.abort(grpc.StatusCode.INTERNAL, message)