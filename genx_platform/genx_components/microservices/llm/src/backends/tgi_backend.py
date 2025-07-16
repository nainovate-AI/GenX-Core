"""
TGI (Text Generation Inference) Backend with Multi-Instance Support
"""
import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator
import aiohttp
import json
from dataclasses import dataclass
import time

from .base import LLMBackend, GenerationConfig, ModelInfo
from .tgi_instance_manager import get_tgi_instance_manager, TGIInstance

logger = logging.getLogger(__name__)


class TGIBackend(LLMBackend):
    """TGI backend with support for multiple model instances"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.instance_manager = get_tgi_instance_manager()
        self.instance: Optional[TGIInstance] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._server_info: Optional[Dict[str, Any]] = None
        
        # Configuration from kwargs
        self.quantize = kwargs.get('quantize', None)
        self.num_shard = kwargs.get('num_shard', 1)
        self.max_input_length = kwargs.get('max_input_length', 2048)
        self.max_total_tokens = kwargs.get('max_total_tokens', 4096)
        self.gpu_memory_fraction = kwargs.get('gpu_memory_fraction', 0.9)
        
    async def initialize(self) -> bool:
        """Initialize the TGI backend by getting or creating an instance"""
        try:
            logger.info(f"Initializing TGI backend for {self.model_id}")
            
            # Get or create TGI instance for this model
            self.instance = await self.instance_manager.get_or_create_instance(
                model_id=self.model_id,
                quantize=self.quantize,
                max_input_length=self.max_input_length,
                max_total_tokens=self.max_total_tokens,
                gpu_memory_fraction=self.gpu_memory_fraction,
                num_shard=self.num_shard
            )
            
            # Create HTTP session for this instance
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Get server info
            self._server_info = await self._get_server_info()
            
            # Verify the correct model is loaded
            if self._server_info:
                loaded_model = self._server_info.get('model_id', '')
                if loaded_model != self.model_id:
                    logger.warning(
                        f"Model mismatch: requested {self.model_id}, "
                        f"but server has {loaded_model}"
                    )
            
            # Set model info
            self.model_info = ModelInfo(
                model_id=self.model_id,
                provider="tgi",
                version=self._server_info.get('version', 'unknown') if self._server_info else 'unknown',
                capabilities=["text-generation", "streaming", "continuous-batching"],
                hardware_requirements={
                    "min_gpu_memory_gb": self._estimate_gpu_memory(),
                    "gpu_required": True,
                    "recommended_gpu": "NVIDIA A100/A6000/V100"
                },
                is_loaded=True,
                is_available=True
            )
            
            self._is_initialized = True
            logger.info(
                f"Successfully initialized TGI backend for {self.model_id} "
                f"on port {self.instance.port}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TGI backend: {e}")
            if self.session:
                await self.session.close()
            return False
    
    async def _get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get TGI server information"""
        if not self.instance:
            return None
            
        try:
            async with self.session.get(f"{self.instance.server_url}/info") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
        return None
    
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory requirements based on model"""
        model_lower = self.model_id.lower()
        
        if "70b" in model_lower:
            return 140.0
        elif "40b" in model_lower:
            return 80.0
        elif "30b" in model_lower:
            return 60.0
        elif "20b" in model_lower:
            return 40.0
        elif "13b" in model_lower:
            return 26.0
        elif "7b" in model_lower:
            return 14.0
        elif "3b" in model_lower:
            return 6.0
        else:
            return 8.0  # Default for smaller models
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate text using TGI server"""
        if not self.instance or not self.session:
            raise RuntimeError("TGI backend not initialized")
        
        # Create a new session for each request if none exists
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
        try:
            # Prepare input
            input_text = self._prepare_input(prompt, system_prompt, messages)
            
            # Build request payload
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": config.repetition_penalty,
                    "do_sample": config.temperature > 0,
                    "return_full_text": False,
                    "details": True
                }
            }
            
            if config.stop_sequences:
                payload["parameters"]["stop"] = config.stop_sequences
            
            if config.seed is not None:
                payload["parameters"]["seed"] = config.seed
            
            # Make request
            async with self.session.post(
                f"{self.instance.server_url}/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    generated_text = result.get("generated_text", "")
                    details = result.get("details", {})
                    
                    prompt_tokens = len(details.get("prefill", []))
                    completion_tokens = len(details.get("tokens", []))
                    
                    return {
                        "text": generated_text,
                        "tokens_used": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        },
                        "finish_reason": details.get("finish_reason", "stop"),
                        "model_id": self.model_id
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"TGI generation failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"TGI generation failed: {e}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using TGI server"""
        if not self.instance or not self.session:
            raise RuntimeError("TGI backend not initialized")
        
        # Create a new session for each request if none exists
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
        try:
            input_text = self._prepare_input(prompt, system_prompt, messages)
            
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": config.repetition_penalty,
                    "do_sample": config.temperature > 0,
                    "return_full_text": False
                },
                "stream": True
            }
            
            if config.stop_sequences:
                payload["parameters"]["stop"] = config.stop_sequences
            
            async with self.session.post(
                f"{self.instance.server_url}/generate_stream",
                json=payload
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "token" in chunk:
                                    yield chunk["token"]["text"]
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    raise Exception(f"TGI streaming failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"TGI streaming failed: {e}")
            raise
    
    def _prepare_input(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Prepare input text from prompt and messages"""
        if messages:
            formatted_parts = []
            
            if system_prompt:
                formatted_parts.append(f"System: {system_prompt}")
            
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                if role == "System":
                    formatted_parts.append(f"System: {content}")
                elif role == "User":
                    formatted_parts.append(f"User: {content}")
                elif role == "Assistant":
                    formatted_parts.append(f"Assistant: {content}")
            
            formatted_parts.append(f"User: {prompt}")
            formatted_parts.append("Assistant:")
            
            return "\n".join(formatted_parts)
        else:
            if system_prompt:
                return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            return prompt
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using estimation"""
        # TGI doesn't expose tokenization endpoint, use estimation
        return int(len(text.split()) * 1.3)
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate prompt for token limits"""
        token_count = await self.count_tokens(prompt)
        max_length = self.max_input_length
        
        issues = []
        if token_count > max_length:
            issues.append({
                "type": "TOO_LONG",
                "severity": "ERROR",
                "message": f"Prompt exceeds maximum length ({token_count} > {max_length})"
            })
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "token_count": token_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check TGI backend health"""
        try:
            health_info = {
                "status": "unhealthy",
                "backend": "tgi",
                "model_loaded": False,
                "device": "cuda"
            }
            
            if self.instance:
                health_info["server_url"] = self.instance.server_url
                health_info["port"] = self.instance.port
                health_info["container_name"] = self.instance.container_name
                
                if self.session:
                    try:
                        async with self.session.get(f"{self.instance.server_url}/health") as response:
                            if response.status == 200:
                                health_info["status"] = "healthy"
                                health_info["model_loaded"] = True
                                
                                if self._server_info:
                                    health_info["model_info"] = {
                                        "model_id": self._server_info.get("model_id"),
                                        "max_concurrent_requests": self._server_info.get("max_concurrent_requests"),
                                        "max_input_length": self._server_info.get("max_input_length"),
                                        "max_total_tokens": self._server_info.get("max_total_tokens")
                                    }
                    except Exception as e:
                        health_info["error"] = str(e)
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Update the cleanup method to properly close sessions
    async def cleanup(self):
        """Clean up resources (but keep TGI instance running for reuse)"""
        logger.info(f"Cleaning up TGI backend for {self.model_id}")
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
            # Wait a bit for proper cleanup
            await asyncio.sleep(0.1)
        self.session = None
        
        self._is_initialized = False
    
    @staticmethod
    async def get_all_instances_info() -> Dict[str, Dict]:
        """Get information about all running TGI instances"""
        manager = get_tgi_instance_manager()
        return manager.get_instance_info()