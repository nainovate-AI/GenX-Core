# genx_platform/genx_components/microservices/llm/src/backends/tgi_backend.py
"""
TGI (Text Generation Inference) Backend
High-performance inference server by Hugging Face
Optimized for NVIDIA GPUs with continuous batching
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

logger = logging.getLogger(__name__)


@dataclass
class TGIConfig:
    """TGI server configuration"""
    server_url: str = "http://localhost:8080"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30


class TGIBackend(LLMBackend):
    """TGI backend for high-performance inference"""
    
    # TGI-optimized models
    TGI_RECOMMENDED_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
        "codellama/CodeLlama-13b-Instruct-hf",
        "tiiuae/falcon-7b-instruct",
        "tiiuae/falcon-40b-instruct",
        "bigscience/bloom",
        "google/flan-t5-xxl",
        "EleutherAI/gpt-neox-20b",
        "databricks/dolly-v2-12b",
    ]
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.config = TGIConfig(
            server_url=kwargs.get('server_url', os.environ.get('TGI_SERVER_URL', 'http://localhost:8080')),
            timeout=kwargs.get('timeout', 300),
            max_retries=kwargs.get('max_retries', 3),
            retry_delay=kwargs.get('retry_delay', 1.0)
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._server_info: Optional[Dict[str, Any]] = None
        self.is_external_server = kwargs.get('external_server', True)
        self.container_name = kwargs.get('container_name', f'tgi-{model_id.replace("/", "-")}')
        self.docker_image = kwargs.get('docker_image', 'ghcr.io/huggingface/text-generation-inference:latest')
        self.sharded = kwargs.get('sharded', False)
        self.quantize = kwargs.get('quantize', None)  # bitsandbytes, gptq, awq
        self.num_shard = kwargs.get('num_shard', 1)
        self.max_input_length = kwargs.get('max_input_length', 4096)
        self.max_total_tokens = kwargs.get('max_total_tokens', 8192)
        self.max_batch_prefill_tokens = kwargs.get('max_batch_prefill_tokens', 4096)
        
    async def initialize(self) -> bool:
        """Initialize the TGI backend"""
        try:
            logger.info(f"Initializing TGI backend for {self.model_id}")
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Check if we need to start TGI server
            if not self.is_external_server:
                logger.info("Starting TGI server in Docker container")
                await self._start_tgi_server()
            else:
                logger.info("Using external TGI server at " + self.config.server_url)
            
            # Wait for server to be ready
            if not await self._wait_for_server():
                logger.error("TGI server failed to start or is not accessible")
                return False
            
            # Get server info
            self._server_info = await self._get_server_info()
            
            # Validate model matches
            if self._server_info and self._server_info.get('model_id') != self.model_id:
                logger.warning(
                    f"Server is running model {self._server_info.get('model_id')}, "
                    f"but requested {self.model_id}"
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
            logger.info(f"Successfully initialized TGI backend for {self.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TGI backend: {e}")
            if self.session:
                await self.session.close()
            return False
    
    async def _start_tgi_server(self):
        """Start TGI server using Docker"""
        try:
            print("Starting TGI server...")
            import subprocess
            
            # Check if container already exists
            check_cmd = f"docker ps -a -q -f name={self.container_name}"
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Container exists, remove it
                logger.info(f"Removing existing container {self.container_name}")
                subprocess.run(f"docker rm -f {self.container_name}", shell=True)
            
            # Build docker command
            docker_cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--gpus", "all",
                "-p", f"{self.config.server_url.split(':')[-1]}:80",
                "-v", f"{os.path.expanduser('~/.cache/huggingface')}:/data",
                "--shm-size", "1g",
                self.docker_image,
                "--model-id", self.model_id,
                "--max-input-length", str(self.max_input_length),
                "--max-total-tokens", str(self.max_total_tokens),
                "--max-batch-prefill-tokens", str(self.max_batch_prefill_tokens)
            ]
            
            # Add optional parameters
            if self.quantize:
                docker_cmd.extend(["--quantize", self.quantize])
            
            if self.sharded or self.num_shard > 1:
                docker_cmd.extend(["--num-shard", str(self.num_shard)])
            
            logger.info(f"Starting TGI server with command: {' '.join(docker_cmd)}")
            subprocess.run(docker_cmd, check=True)
            
            logger.info(f"TGI server container {self.container_name} started")
            
        except Exception as e:
            logger.error(f"Failed to start TGI server: {e}")
            raise
    
    async def _wait_for_server(self, max_wait: int = 120) -> bool:
        """Wait for TGI server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                async with self.session.get(f"{self.config.server_url}/health") as response:
                    if response.status == 200:
                        logger.info("TGI server is ready")
                        return True
            except:
                pass
            
            await asyncio.sleep(2)
        
        return False
    
    async def _get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get TGI server information"""
        try:
            async with self.session.get(f"{self.config.server_url}/info") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
        return None
    
    def _estimate_gpu_memory(self) -> float:
        """Estimate GPU memory requirements based on model"""
        # Rough estimates based on model size
        model_lower = self.model_id.lower()
        
        if "70b" in model_lower:
            return 140.0  # 2x A100 80GB
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
            return 16.0  # Default
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate text using TGI server"""
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
            
            # Add stop sequences if provided
            if config.stop_sequences:
                payload["parameters"]["stop"] = config.stop_sequences
            
            # Add seed if provided
            if config.seed is not None:
                payload["parameters"]["seed"] = config.seed
            
            # Make request with retries
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(
                        f"{self.config.server_url}/generate",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Extract generated text and details
                            generated_text = result.get("generated_text", "")
                            details = result.get("details", {})
                            
                            # Calculate token usage
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
                            logger.error(f"TGI generation failed: {response.status} - {error_text}")
                            
                except aiohttp.ClientError as e:
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise
            
            raise Exception("Failed to generate after all retries")
            
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
                    "return_full_text": False
                },
                "stream": True
            }
            
            # Add stop sequences if provided
            if config.stop_sequences:
                payload["parameters"]["stop"] = config.stop_sequences
            
            # Make streaming request
            async with self.session.post(
                f"{self.config.server_url}/generate_stream",
                json=payload
            ) as response:
                if response.status == 200:
                    # Read server-sent events
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
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
        # TGI supports various chat templates, let's use a simple format
        if messages:
            # Format as conversation
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
        """Count tokens using TGI tokenizer endpoint"""
        try:
            # TGI doesn't have a direct token counting endpoint
            # We'll estimate based on the model
            # For now, use a rough estimate
            return len(text.split()) * 1.3  # Rough approximation
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split())
    
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
        """Check TGI server health"""
        try:
            health_info = {
                "status": "unhealthy",
                "backend": "tgi",
                "model_loaded": False,
                "server_url": self.config.server_url,
                "device": "cuda"  # TGI primarily supports NVIDIA GPUs
            }
            
            if self.session:
                # Check health endpoint
                try:
                    async with self.session.get(f"{self.config.server_url}/health") as response:
                        if response.status == 200:
                            health_info["status"] = "healthy"
                            health_info["model_loaded"] = True
                            
                            # Get additional info if available
                            if self._server_info:
                                health_info["model_info"] = {
                                    "model_id": self._server_info.get("model_id"),
                                    "model_type": self._server_info.get("model_type"),
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
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up TGI backend")
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Stop TGI server if we started it
        if not self.is_external_server:
            try:
                import subprocess
                logger.info(f"Stopping TGI container {self.container_name}")
                subprocess.run(f"docker stop {self.container_name}", shell=True)
                subprocess.run(f"docker rm {self.container_name}", shell=True)
            except Exception as e:
                logger.error(f"Failed to stop TGI container: {e}")
        
        self._is_initialized = False