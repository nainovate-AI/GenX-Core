"""
MLX Backend for Apple Silicon
Optimized for M1/M2/M3 chips with unified memory architecture
"""
import asyncio
import logging
import platform
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
import importlib.metadata
import time

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx.utils import tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX not available. Install with: pip install mlx mlx-lm")

from .base import LLMBackend, GenerationConfig, ModelInfo

logger = logging.getLogger(__name__)


class MLXBackend(LLMBackend):
    """MLX backend for Apple Silicon optimization"""
    
    # MLX-optimized models available in the community
    MLX_MODEL_MAP = {
        # Direct MLX community models
        "gpt2": "mlx-community/gpt2-mlx",
        "distilgpt2": "mlx-community/distilgpt2-mlx",
        "phi-4": "mlx-community/Phi-4-reasoning-4bit",
        "mistral-small":"mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
        "stable-lm-3b": "mlx-community/stable-lm-2-zephyr-1_6b-mlx",
        "mistral-7b": "mlx-community/Mistral-7B-v0.1-mlx",
        "llama-2-7b": "mlx-community/llama-2-7b-chat",
        "codellama-7b": "mlx-community/CodeLlama-7b-Python-mlx",
        "gemma-3b": "mlx-community/gemma-3-12b-it-qat-4bit",
        "falcon-0.5b": "mlx-community/Falcon-H1-0.5B-Instruct-bf16",
        
        # Map common names to MLX versions
        "microsoft/phi-2": "mlx-community/phi-2-mlx",
        "stabilityai/stablelm-3b": "mlx-community/stable-lm-2-zephyr-1_6b-mlx",
    }
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Run: pip install mlx mlx-lm")
            
        # Check if running on Apple Silicon
        if not self._is_apple_silicon():
            logger.warning("MLX backend is optimized for Apple Silicon. Performance may be suboptimal.")
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        return (
            platform.system() == "Darwin" and 
            platform.processor() == "arm" and
            platform.machine() == "arm64"
        )
    
    def _get_mlx_model_id(self, model_id: str) -> str:
        """Get MLX-compatible model ID"""
        # Check if it's already an MLX model
        if "mlx" in model_id.lower():
            return model_id
            
        # Check our mapping
        if model_id in self.MLX_MODEL_MAP:
            mlx_id = self.MLX_MODEL_MAP[model_id]
            logger.info(f"Mapped {model_id} to MLX model: {mlx_id}")
            return mlx_id
            
        # Try adding -mlx suffix
        mlx_id = f"mlx-community/{model_id.split('/')[-1]}-mlx"
        logger.info(f"Trying MLX model: {mlx_id}")
        return mlx_id
    
    async def initialize(self) -> bool:
        """Initialize the backend and load the model"""
        try:
            logger.info(f"Initializing MLX backend for {self.model_id}")
            
            # Show MLX device info
            logger.info(f"MLX default device: {mx.default_device()}")
            
            # Get MLX-compatible model ID
            mlx_model_id = self._get_mlx_model_id(self.model_id)
            
            # Load model and tokenizer
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._load_model_sync, mlx_model_id
                )
            except Exception as e:
                if "not supported" in str(e):
                    logger.error(f"Model architecture not supported by MLX: {e}")
                    logger.info("Try one of these models instead: gpt2, distilgpt2, opt-125m, pythia-70m")
                    return False
                raise
            
            # Try to get model size (with error handling)
            param_count = 0
            try:
                # MLX models might have different parameter structure
                if hasattr(self.model, 'parameters'):
                    params = self.model.parameters()
                    if params:
                        # Try different ways to count parameters
                        try:
                            # Method 1: If parameters are MLX arrays
                            param_count = sum(p.size for p in params.values() if hasattr(p, 'size')) / 1e9
                        except:
                            try:
                                # Method 2: If parameters are in a different structure
                                param_count = sum(p.size for p in params if hasattr(p, 'size')) / 1e9
                            except:
                                # Method 3: Just estimate based on model name
                                param_count = self._estimate_model_size(mlx_model_id)
            except Exception as e:
                logger.debug(f"Could not count parameters: {e}")
                param_count = self._estimate_model_size(mlx_model_id)
            
            # Set model info
            self.model_info = ModelInfo(
                model_id=self.model_id,
                provider="mlx",
                version="latest",
                capabilities=["text-generation", "streaming"],
                hardware_requirements={
                    "min_ram_gb": max(8, int(param_count * 2)),  # Rough estimate
                    "min_gpu_memory_gb": 0,  # Uses unified memory
                    "gpu_required": False,
                    "apple_silicon_required": True
                },
                is_loaded=True,
                is_available=True
            )
            
            self._is_initialized = True
            logger.info(f"Successfully initialized {self.model_id} with MLX ({param_count:.1f}B parameters)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLX model: {e}")
            # Fallback suggestion
            logger.info("Try one of these MLX models: " + ", ".join(self.MLX_MODEL_MAP.keys()))
            return False
    
    def _load_model_sync(self, model_id: str):
        """Load model synchronously"""
        logger.info(f"Loading MLX model: {model_id}")
        start_time = time.time()
        
        try:
            # Load model with MLX
            self.model, self.tokenizer = load(model_id)
            
            # Store model config if available
            if hasattr(self.model, 'config'):
                self.model_config = self.model.config
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            # Try without -mlx suffix
            if model_id.endswith("-mlx"):
                alt_id = model_id[:-4]
                logger.info(f"Trying alternative ID: {alt_id}")
                self.model, self.tokenizer = load(alt_id)
            else:
                raise
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate text completion using MLX"""
        try:
            # Prepare input
            input_text = self._prepare_input(prompt, system_prompt, messages)
            
            # Set random seed if provided
            if config.seed is not None:
                mx.random.seed(config.seed)
            
            # sampler = self.build_sampler_from_config(config)

            # Prepare generation kwargs
            generate_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": input_text,
                "max_tokens": config.max_tokens,
                # "sampler": sampler,
                # "temperature": config.temperature,
                # "top_p": config.top_p,
                # "repetition_penalty": config.repetition_penalty,
                # "repetition_context_size": 20,
            }
            
            # Generate in thread pool to avoid blocking
            start_time = time.time()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate(**generate_kwargs)
            )
            
            generation_time = time.time() - start_time
            
            # Extract just the generated text (remove the prompt)
            if response.startswith(input_text):
                generated_text = response[len(input_text):]
            else:
                generated_text = response
            
            # Count tokens
            prompt_tokens = len(self.tokenizer.encode(input_text))
            completion_tokens = len(self.tokenizer.encode(generated_text))
            
            logger.debug(f"Generated {completion_tokens} tokens in {generation_time:.2f}s "
                        f"({completion_tokens/generation_time:.1f} tokens/s)")
            
            return {
                "text": generated_text.strip(),
                "tokens_used": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "finish_reason": "stop",
                "model_id": self.model_id
            }
            
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using MLX"""
        try:
            # For now, MLX doesn't have built-in streaming support
            # We'll simulate it by generating the full response and yielding in chunks
            result = await self.generate(prompt, config, system_prompt, messages)
            text = result["text"]
            
            # Yield text in small chunks to simulate streaming
            chunk_size = 5  # Characters per chunk
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate streaming
                
        except Exception as e:
            logger.error(f"MLX streaming failed: {e}")
            raise
    
    def _prepare_input(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Prepare input text from prompt and messages"""
        if messages:
            # Format as conversation
            formatted_parts = []
            
            if system_prompt:
                formatted_parts.append(f"System: {system_prompt}")
            
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                formatted_parts.append(f"{role}: {content}")
            
            formatted_parts.append(f"Human: {prompt}")
            formatted_parts.append("Assistant:")
            
            return "\n".join(formatted_parts)
        else:
            if system_prompt:
                return f"{system_prompt}\n\nHuman: {prompt}\nAssistant:"
            return prompt
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate prompt for token limits"""
        token_count = await self.count_tokens(prompt)
        
        # Get max length from model config or use default
        max_length = 2048
        if self.model_config and hasattr(self.model_config, 'max_position_embeddings'):
            max_length = self.model_config.max_position_embeddings
        
        issues = []
        if token_count > max_length * 0.9:  # Warn at 90% capacity
            issues.append({
                "type": "TOO_LONG",
                "severity": "WARNING" if token_count < max_length else "ERROR",
                "message": f"Prompt uses {token_count}/{max_length} tokens"
            })
        
        return {
            "is_valid": token_count <= max_length,
            "issues": issues,
            "token_count": token_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        health_info = {
            "status": "healthy" if self._is_initialized else "unhealthy",
            "backend": "mlx",
            "model_loaded": self.model is not None,
            "device": "apple_silicon",
            "mlx_version": importlib.metadata.version("mlx") if MLX_AVAILABLE else "not_installed",
            "metal_device": str(mx.default_device()),
            "is_apple_silicon": self._is_apple_silicon()
        }
        
        # Add memory info if available
        if self.model is not None:
            flat_params = tree_flatten(self.model)
            param_count = sum(p[1].size for p in flat_params)  # p = (name, tensor)
            health_info["model_parameters"] = param_count
            health_info["model_size_gb"] = param_count * 4 / 1e9  # Assuming float32

        return health_info
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up MLX backend")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # MLX uses unified memory, no need for explicit cache clearing
        self._is_initialized = False