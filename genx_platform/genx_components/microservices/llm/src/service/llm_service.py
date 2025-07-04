"""
Enhanced LLM gRPC Service Implementation with Dynamic Model Management
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any
import grpc
from google.protobuf import timestamp_pb2
from google.protobuf import struct_pb2
from datetime import datetime 

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
    """Enhanced gRPC service implementation for LLM operations"""
    
    def __init__(self, model_manager: ModelManager, telemetry: Optional[GenxTelemetry] = None):
        self.model_manager = model_manager
        self.telemetry = telemetry
        self.service_name = "llm-service"
        self.service_version = "0.2.0"
    
    # ===================== Model Management APIs =====================
    
    async def LoadModel(
        self,
        request: llm_service_pb2.LoadModelRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.LoadModelResponse:
        """Load a model dynamically with specified backend and device"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        with self._trace_operation("load_model", request_id, request.model_name) as span:
            try:
                # Validate request
                if not request.model_name:
                    await self._set_error(context, "INVALID_ARGUMENT", "model_name is required")
                    return self._create_load_model_error_response(request_id, "model_name is required")
                
                if not request.backend:
                    await self._set_error(context, "INVALID_ARGUMENT", "backend is required")
                    return self._create_load_model_error_response(request_id, "backend is required")
                
                # Prepare backend configuration
                backend_config = {}
                if request.HasField('backend_config'):
                    backend_config = self._struct_to_dict(request.backend_config)
                
                # Add options from request
                if request.HasField('options'):
                    options = request.options
                    backend_config.update({
                        'load_in_8bit': options.load_in_8bit,
                        'load_in_4bit': options.load_in_4bit,
                        'device_map': options.device_map or 'auto',
                        'trust_remote_code': options.trust_remote_code,
                    })
                    
                    if options.gpu_memory_fraction > 0:
                        backend_config['gpu_memory_utilization'] = options.gpu_memory_fraction
                    
                    if options.tensor_parallel_size > 0:
                        backend_config['tensor_parallel_size'] = options.tensor_parallel_size
                    
                    if options.use_flash_attention:
                        backend_config['use_flash_attention'] = True
                
                # Load the model
                model_id, loaded_model = await self.model_manager.load_model(
                    model_name=request.model_name,
                    backend=request.backend,
                    device=request.device or 'auto',
                    backend_config=backend_config
                )
                
                if not model_id or not loaded_model:
                    return self._create_load_model_error_response(
                        request_id, 
                        f"Failed to load model {request.model_name}"
                    )
                
                # Create response
                load_time_ms = int((time.time() - start_time) * 1000)
                
                response = llm_service_pb2.LoadModelResponse(
                    metadata=self._create_response_metadata(request_id, start_time),
                    success=True,
                    model_id=model_id,
                    model_info=await self._create_loaded_model_info(model_id, loaded_model)
                )
                
                # Log telemetry
                if self.telemetry:
                    self.telemetry.record_metric(
                        "model_loaded",
                        1,
                        {
                            "model": request.model_name,
                            "backend": request.backend,
                            "device": request.device,
                            "load_time_ms": load_time_ms
                        }
                    )
                
                return response
                
            except Exception as e:
                logger.error(f"LoadModel failed: {e}")
                await self._set_error(context, "INTERNAL", str(e))
                return self._create_load_model_error_response(request_id, str(e))
    
    async def UnloadModel(
        self,
        request: llm_service_pb2.UnloadModelRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.UnloadModelResponse:
        """Unload a specific model"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        try:
            success = await self.model_manager.unload_model(request.model_id)
            
            response = llm_service_pb2.UnloadModelResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                success=success
            )
            
            if not success:
                response.error.CopyFrom(common_pb2.ErrorDetail(
                    code="NOT_FOUND",
                    message=f"Model {request.model_id} not found"
                ))
            
            return response
            
        except Exception as e:
            logger.error(f"UnloadModel failed: {e}")
            await self._set_error(context, "INTERNAL", str(e))
            return llm_service_pb2.UnloadModelResponse(
                metadata=self._create_response_metadata(request_id, start_time),
                success=False,
                error=common_pb2.ErrorDetail(code="INTERNAL", message=str(e))
            )
    
    async def GetLoadedModels(
        self,
        request: llm_service_pb2.GetLoadedModelsRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.GetLoadedModelsResponse:
        """Get information about loaded models"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        
        try:
            # Get models info
            info = await self.model_manager.get_loaded_models_info(
                backend_filter=request.backend if request.HasField('backend') else None,
                device_filter=request.device if request.HasField('device') else None,
                include_stats=request.include_stats
            )
            
            # Create response
            response = llm_service_pb2.GetLoadedModelsResponse(
                metadata=self._create_response_metadata(request_id, start_time)
            )
            
            # Add loaded models
            for model_info in info['models']:
                loaded_model_info = llm_service_pb2.LoadedModelInfo(
                    model_id=model_info['model_id'],
                    model_name=model_info['model_name'],
                    backend=model_info['backend'],
                    device=model_info['device']
                )
                
                # Set loaded_at timestamp
                loaded_at = timestamp_pb2.Timestamp()
                loaded_at.FromDatetime(
                    datetime.fromisoformat(model_info['loaded_at'])
                )
                loaded_model_info.loaded_at.CopyFrom(loaded_at)
                
                # Add stats if requested
                if request.include_stats and 'stats' in model_info:
                    stats = model_info['stats']
                    loaded_model_info.stats.CopyFrom(
                        llm_service_pb2.ModelUsageStats(
                            total_requests=stats['total_requests'],
                            total_tokens_generated=stats['total_tokens_generated'],
                            avg_response_time_ms=stats['avg_response_time_ms']
                        )
                    )
                    
                    # Set last_used timestamp
                    last_used = timestamp_pb2.Timestamp()
                    last_used.FromDatetime(
                        datetime.fromisoformat(stats['last_used'])
                    )
                    loaded_model_info.stats.last_used.CopyFrom(last_used)
                
                # Add capabilities
                if 'capabilities' in model_info:
                    caps = model_info['capabilities']
                    loaded_model_info.capabilities.CopyFrom(
                        llm_service_pb2.ModelCapabilities(
                            max_context_length=caps.get('max_context_length', 2048),
                            features=caps.get('features', []),
                            model_type=caps.get('model_type', 'causal_lm')
                        )
                    )
                
                # Add status
                status = model_info.get('status', {})
                loaded_model_info.status.CopyFrom(
                    llm_service_pb2.ModelStatus(
                        is_loaded=status.get('is_loaded', False),
                        is_available=status.get('is_available', False),
                        current_load=status.get('current_load', 0.0)
                    )
                )
                
                # Add memory usage if available
                if 'memory_usage' in model_info:
                    mem = model_info['memory_usage']
                    loaded_model_info.memory_usage.CopyFrom(
                        llm_service_pb2.MemoryUsage(
                            gpu_memory_mb=int(mem.get('gpu_memory_mb', 0)),
                            ram_mb=int(mem.get('ram_mb', 0))
                        )
                    )
                
                response.models.append(loaded_model_info)
            
            # Add system info
            system_info = info['system_info']
            response.system_info.CopyFrom(
                self._create_system_resource_info(system_info)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"GetLoadedModels failed: {e}")
            await self._set_error(context, "INTERNAL", str(e))
            return llm_service_pb2.GetLoadedModelsResponse(
                metadata=self._create_response_metadata(request_id, start_time)
            )
    
    # ===================== Text Generation APIs =====================
    
    async def GenerateText(
        self,
        request: llm_service_pb2.GenerateTextRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.GenerateTextResponse:
        """Generate text with user tracking and history support"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        generation_start = None
        
        with self._trace_operation("generate_text", request_id, request.user_id) as span:
            try:
                # Validate request
                if not request.prompt:
                    await self._set_error(context, "INVALID_ARGUMENT", "prompt is required")
                    return self._create_generate_text_error_response(request_id, "prompt is required")
                
                if not request.user_id:
                    await self._set_error(context, "INVALID_ARGUMENT", "user_id is required")
                    return self._create_generate_text_error_response(request_id, "user_id is required")
                
                # Get model
                model_id = request.model_id if request.HasField('model_id') else None
                backend = await self.model_manager.get_model(model_id)
                
                if not backend:
                    return self._create_generate_text_error_response(
                        request_id,
                        f"Model {model_id or 'default'} not available"
                    )
                
                # Get actual model ID if default was used
                if not model_id:
                    for mid, loaded in self.model_manager.models.items():
                        if loaded.backend == backend:
                            model_id = mid
                            break
                
                # Build full prompt with config
                full_prompt = self._build_prompt(request.prompt, request.prompt_config)
                
                # Add conversation history if requested
                messages = None
                if request.use_history:
                    history = self.model_manager.get_user_history(request.user_id)
                    if history:
                        messages = history.copy()
                
                # Convert generation config
                config = self._convert_generation_config(request.generation_config)
                
                # Track when generation starts
                generation_start = time.time()
                
                # Generate response
                if request.streaming:
                    # For streaming, we'll handle it in StreamGenerateText
                    await self._set_error(context, "INVALID_ARGUMENT", 
                                        "Use StreamGenerateText for streaming")
                    return self._create_generate_text_error_response(
                        request_id, 
                        "Use StreamGenerateText for streaming"
                    )
                
                # Generate text
                result = await backend.generate(
                    prompt=request.prompt,
                    config=config,
                    system_prompt=request.prompt_config.system_prompt if request.HasField('prompt_config') else None,
                    messages=messages
                )
                
                generation_time = time.time() - generation_start
                
                # Update statistics
                self.model_manager.update_stats(
                    model_id,
                    result['tokens_used']['total_tokens'],
                    generation_time
                )
                
                # Add to history if requested
                if request.use_history:
                    self.model_manager.add_to_history(request.user_id, 'user', request.prompt)
                    self.model_manager.add_to_history(request.user_id, 'assistant', result['text'])
                
                # Create response
                response = llm_service_pb2.GenerateTextResponse(
                    metadata=self._create_response_metadata(request_id, start_time),
                    generated_text=result['text'],
                    model_id=model_id,
                    usage=common_pb2.TokenUsage(
                        prompt_tokens=result['tokens_used']['prompt_tokens'],
                        completion_tokens=result['tokens_used']['completion_tokens'],
                        total_tokens=result['tokens_used']['total_tokens']
                    ),
                    finish_reason=result.get('finish_reason', 'stop')
                )
                
                # Add generation stats
                time_to_first_token = int((generation_start - start_time) * 1000)
                total_time = int((time.time() - start_time) * 1000)
                tokens_per_second = (
                    result['tokens_used']['completion_tokens'] / generation_time
                    if generation_time > 0 else 0
                )
                
                response.stats.CopyFrom(
                    llm_service_pb2.GenerationStats(
                        time_to_first_token_ms=time_to_first_token,
                        total_time_ms=total_time,
                        tokens_per_second=tokens_per_second
                    )
                )
                
                # Record metrics
                if self.telemetry:
                    self.telemetry.record_metric(
                        "text_generated",
                        result['tokens_used']['completion_tokens'],
                        {
                            "model": model_id,
                            "user": request.user_id,
                            "with_history": request.use_history
                        }
                    )
                
                return response
                
            except Exception as e:
                logger.error(f"GenerateText failed: {e}")
                await self._set_error(context, "INTERNAL", str(e))
                return self._create_generate_text_error_response(request_id, str(e))
    
    async def StreamGenerateText(
        self,
        request: llm_service_pb2.GenerateTextRequest,
        context: grpc.aio.ServicerContext
    ) -> llm_service_pb2.StreamGenerateTextResponse:
        """Stream text generation with user tracking"""
        start_time = time.time()
        request_id = request.metadata.request_id or str(uuid.uuid4())
        generation_start = None
        
        with self._trace_operation("stream_generate_text", request_id, request.user_id) as span:
            try:
                # Validate request
                if not request.prompt or not request.user_id:
                    yield self._create_stream_generate_text_error_response(
                        "prompt and user_id are required"
                    )
                    return
                
                # Get model
                model_id = request.model_id if request.HasField('model_id') else None
                backend = await self.model_manager.get_model(model_id)
                
                if not backend:
                    yield self._create_stream_generate_text_error_response(
                        f"Model {model_id or 'default'} not available"
                    )
                    return
                
                # Send initial metadata
                yield llm_service_pb2.StreamGenerateTextResponse(
                    metadata=self._create_response_metadata(request_id, start_time)
                )
                
                # Build prompt and get history
                full_prompt = self._build_prompt(request.prompt, request.prompt_config)
                messages = None
                if request.use_history:
                    messages = self.model_manager.get_user_history(request.user_id)
                
                # Convert config
                config = self._convert_generation_config(request.generation_config)
                
                # Start generation
                generation_start = time.time()
                cumulative_text = ""
                token_count = 0
                
                # Stream generation
                async for chunk in backend.stream_generate(
                    prompt=request.prompt,
                    config=config,
                    system_prompt=request.prompt_config.system_prompt if request.HasField('prompt_config') else None,
                    messages=messages
                ):
                    cumulative_text += chunk
                    token_count += 1
                    
                    yield llm_service_pb2.StreamGenerateTextResponse(
                        text_chunk=chunk,
                        cumulative_text=cumulative_text,
                        is_final=False
                    )
                
                # Calculate final stats
                generation_time = time.time() - generation_start
                prompt_tokens = await backend.count_tokens(full_prompt)
                
                # Add to history if requested
                if request.use_history:
                    self.model_manager.add_to_history(request.user_id, 'user', request.prompt)
                    self.model_manager.add_to_history(request.user_id, 'assistant', cumulative_text)
                
                # Update stats
                if model_id:
                    self.model_manager.update_stats(model_id, token_count, generation_time)
                
                # Send final message
                final_response = llm_service_pb2.StreamGenerateTextResponse(
                    is_final=True,
                    cumulative_text=cumulative_text,
                    finish_reason="stop"
                )
                
                # Add usage
                final_response.usage.CopyFrom(
                    common_pb2.TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=token_count,
                        total_tokens=prompt_tokens + token_count
                    )
                )
                
                # Add stats
                final_response.stats.CopyFrom(
                    llm_service_pb2.GenerationStats(
                        time_to_first_token_ms=int((generation_start - start_time) * 1000),
                        total_time_ms=int((time.time() - start_time) * 1000),
                        tokens_per_second=token_count / generation_time if generation_time > 0 else 0
                    )
                )
                
                yield final_response
                
            except Exception as e:
                logger.error(f"StreamGenerateText failed: {e}")
                yield self._create_stream_generate_text_error_response(str(e))
    
    # ===================== Helper Methods =====================
    
    def _build_prompt(
        self,
        prompt: str,
        prompt_config: Optional[llm_service_pb2.PromptConfig]
    ) -> str:
        """Build full prompt from prompt and config"""
        if not prompt_config:
            return prompt
        
        parts = []
        
        # Add system prompt
        if prompt_config.HasField('system_prompt'):
            parts.append(f"System: {prompt_config.system_prompt}")
        
        # Add context
        if prompt_config.HasField('context'):
            parts.append(f"Context: {prompt_config.context}")
        
        # Add examples
        if prompt_config.examples:
            parts.append("Examples:")
            for example in prompt_config.examples:
                parts.append(f"Input: {example.input}")
                parts.append(f"Output: {example.output}")
        
        # Add format instructions
        if prompt_config.HasField('format_instructions'):
            parts.append(f"Format: {prompt_config.format_instructions}")
        
        # Add the actual prompt
        if parts:
            parts.append("")  # Empty line
            parts.append(f"Input: {prompt}")
            return "\n".join(parts)
        
        return prompt
    
    def _convert_generation_config(
        self,
        config: llm_service_pb2.TextGenerationConfig
    ) -> GenerationConfig:
        """Convert proto generation config to internal config"""
        return GenerationConfig(
            max_tokens=config.max_tokens or 512,
            temperature=config.temperature or 0.7,
            top_p=config.top_p or 0.95,
            top_k=config.top_k or 50,
            repetition_penalty=config.repetition_penalty or 1.0,
            stop_sequences=list(config.stop_sequences) if config.stop_sequences else None,
            seed=config.seed if config.HasField('seed') else None
        )
    
    def _struct_to_dict(self, struct: struct_pb2.Struct) -> Dict[str, Any]:
        """Convert protobuf Struct to Python dict"""
        return dict(struct)
    
    async def _create_loaded_model_info(
        self,
        model_id: str,
        loaded_model
    ) -> llm_service_pb2.LoadedModelInfo:
        """Create LoadedModelInfo from loaded model"""
        info = llm_service_pb2.LoadedModelInfo(
            model_id=model_id,
            model_name=loaded_model.model_name,
            backend=loaded_model.backend_type,
            device=loaded_model.device
        )
        
        # Set timestamp
        loaded_at = timestamp_pb2.Timestamp()
        loaded_at.FromSeconds(int(loaded_model.loaded_at))
        info.loaded_at.CopyFrom(loaded_at)
        
        # Add basic stats
        info.stats.CopyFrom(
            llm_service_pb2.ModelUsageStats(
                total_requests=0,
                total_tokens_generated=0,
                avg_response_time_ms=0
            )
        )
        
        # Set last_used to now
        last_used = timestamp_pb2.Timestamp()
        last_used.GetCurrentTime()
        info.stats.last_used.CopyFrom(last_used)
        
        # Add capabilities from backend
        backend_info = loaded_model.backend.get_model_info()
        if backend_info:
            info.capabilities.CopyFrom(
                llm_service_pb2.ModelCapabilities(
                    max_context_length=getattr(backend_info, 'max_context_length', 2048),
                    features=backend_info.capabilities,
                    model_type='causal_lm'
                )
            )
            
            info.status.CopyFrom(
                llm_service_pb2.ModelStatus(
                    is_loaded=backend_info.is_loaded,
                    is_available=backend_info.is_available,
                    current_load=0.0
                )
            )
        
        return info
    
    def _create_system_resource_info(
        self,
        system_info: Dict[str, Any]
    ) -> llm_service_pb2.SystemResourceInfo:
        """Create SystemResourceInfo from dict"""
        info = llm_service_pb2.SystemResourceInfo()
        
        # Add GPUs
        for gpu in system_info.get('gpus', []):
            gpu_info = llm_service_pb2.GPUInfo(
                index=gpu['index'],
                name=gpu['name'],
                total_memory_mb=gpu['total_memory_mb'],
                used_memory_mb=gpu['used_memory_mb'],
                utilization_percent=gpu.get('utilization_percent', 0)
            )
            info.gpus.append(gpu_info)
        
        # Add CPU info
        cpu = system_info.get('cpu', {})
        info.cpu.CopyFrom(
            llm_service_pb2.CPUInfo(
                cores=cpu.get('cores', 0),
                utilization_percent=cpu.get('utilization_percent', 0)
            )
        )
        
        # Add memory info
        memory = system_info.get('memory', {})
        info.memory.CopyFrom(
            llm_service_pb2.SystemMemoryInfo(
                total_ram_mb=memory.get('total_ram_mb', 0),
                used_ram_mb=memory.get('used_ram_mb', 0),
                available_ram_mb=memory.get('available_ram_mb', 0)
            )
        )
        
        return info
    
    def _create_load_model_error_response(
        self,
        request_id: str,
        error_message: str
    ) -> llm_service_pb2.LoadModelResponse:
        """Create error response for LoadModel"""
        return llm_service_pb2.LoadModelResponse(
            metadata=self._create_response_metadata(request_id, time.time()),
            success=False,
            error=common_pb2.ErrorDetail(
                code="INTERNAL",
                message=error_message
            )
        )
    
    def _create_generate_text_error_response(
        self,
        request_id: str,
        error_message: str
    ) -> llm_service_pb2.GenerateTextResponse:
        """Create error response for GenerateText"""
        return llm_service_pb2.GenerateTextResponse(
            metadata=self._create_response_metadata(request_id, time.time()),
            error=common_pb2.ErrorDetail(
                code="INTERNAL",
                message=error_message
            )
        )
    
    def _create_stream_generate_text_error_response(
        self,
        error_message: str
    ) -> llm_service_pb2.StreamGenerateTextResponse:
        """Create error response for StreamGenerateText"""
        return llm_service_pb2.StreamGenerateTextResponse(
            error=common_pb2.ErrorDetail(
                code="INTERNAL",
                message=error_message
            ),
            is_final=True
        )
    
    # ===================== Existing Methods (kept for compatibility) =====================
    # [Keep all existing methods like Generate, StreamGenerate, etc.]
    # ... (previous implementation remains the same)