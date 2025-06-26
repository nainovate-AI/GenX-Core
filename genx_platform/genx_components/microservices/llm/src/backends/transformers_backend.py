"""
genx_platform/genx_components/microservices/llm/src/backends/transformers_backend.py
Transformers Backend - Universal fallback that works on all platforms
Supports CPU, CUDA, MPS (Apple Silicon)
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig as HFGenerationConfig
)
from threading import Thread
import gc

from .base import LLMBackend, GenerationConfig, ModelInfo

logger = logging.getLogger(__name__)


class TransformersBackend(LLMBackend):
    """Transformers backend for universal model support"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.device_map = kwargs.get('device_map', 'auto')
        self.torch_dtype = kwargs.get('torch_dtype', 'auto')
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.load_in_8bit = kwargs.get('load_in_8bit', False)
        self.load_in_4bit = kwargs.get('load_in_4bit', False)
        
    async def initialize(self) -> bool:
        """Initialize the backend and load the model"""
        try:
            logger.info(f"Initializing Transformers backend for {self.model_id}")
            
            # Detect best device
            self._detect_device()
            
            # Load model and tokenizer in thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_model_sync
            )
            
            # Set model info
            self.model_info = ModelInfo(
                model_id=self.model_id,
                provider="transformers",
                version="latest",
                capabilities=["text-generation", "chat"],
                hardware_requirements={
                    "min_ram_gb": 8,
                    "min_gpu_memory_gb": 0,
                    "gpu_required": False
                },
                is_loaded=True,
                is_available=True
            )
            
            self._is_initialized = True
            logger.info(f"Successfully initialized {self.model_id} on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def _detect_device(self):
        """Detect the best available device"""
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Apple Silicon detected. Using MPS acceleration")
        else:
            self.device = "cpu"
            logger.info("Using CPU")
    
    def _load_model_sync(self):
        """Load model and tokenizer synchronously"""
        logger.info(f"Loading tokenizer for {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model {self.model_id}")
        
        # Determine torch dtype
        if self.torch_dtype == 'auto':
            if self.device == "cuda":
                torch_dtype = torch.float16
            elif self.device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = getattr(torch, self.torch_dtype)
        
        # Model loading arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        
        # Handle device mapping
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = self.device_map
        elif self.device != "cpu":
            model_kwargs["device_map"] = self.device
        
        # Quantization
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs and self.device != "cpu":
            self.model = self.model.to(self.device)
        
        # Set to eval mode
        self.model.eval()
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate text completion"""
        try:
            # Prepare input
            input_text = self._prepare_input(prompt, system_prompt, messages)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create generation config
            gen_config = HFGenerationConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.temperature > 0,
                num_return_sequences=config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Add stop sequences if provided
            if config.stop_sequences:
                stop_token_ids = []
                for stop_seq in config.stop_sequences:
                    tokens = self.tokenizer(stop_seq, add_special_tokens=False).input_ids
                    if tokens:
                        stop_token_ids.extend(tokens)
                if stop_token_ids:
                    gen_config.eos_token_id = stop_token_ids
            
            # Generate in thread pool to avoid blocking
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = output[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate token usage
            prompt_tokens = input_length
            completion_tokens = len(generated_tokens)
            
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
            logger.error(f"Generation failed: {e}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        try:
            # Prepare input
            input_text = self._prepare_input(prompt, system_prompt, messages)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Create generation config
            gen_config = HFGenerationConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Start generation in separate thread
            generation_kwargs = {
                **inputs,
                "generation_config": gen_config,
                "streamer": streamer
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            for token in streamer:
                yield token
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def _prepare_input(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Prepare input text from prompt and messages"""
        if messages:
            # Format as chat if messages provided
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            formatted_messages.extend(messages)
            formatted_messages.append({"role": "user", "content": prompt})
            
            # Try to use chat template if available
            try:
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    return self.tokenizer.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            except Exception as e:
                logger.debug(f"Chat template not available or failed: {e}")
            
            # Fallback to simple format
            text = ""
            for msg in formatted_messages:
                role = msg['role'].capitalize()
                content = msg['content']
                if role == "System":
                    text += f"System: {content}\n\n"
                elif role == "User":
                    text += f"Human: {content}\n"
                elif role == "Assistant":
                    text += f"Assistant: {content}\n"
            text += "Assistant: "
            return text
        else:
            # Simple prompt
            if system_prompt:
                return f"{system_prompt}\n\nHuman: {prompt}\nAssistant: "
            return prompt
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        tokens = self.tokenizer(text, add_special_tokens=True)
        return len(tokens.input_ids)
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate prompt"""
        token_count = await self.count_tokens(prompt)
        max_length = self.tokenizer.model_max_length
        
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
        """Check backend health"""
        return {
            "status": "healthy" if self._is_initialized else "unhealthy",
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "available_memory": self._get_available_memory()
        }
    
    def _get_available_memory(self) -> Dict[str, float]:
        """Get available memory"""
        memory_info = {}
        
        if self.device == "cuda":
            memory_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        elif self.device == "mps":
            # MPS doesn't provide memory info yet
            memory_info["device"] = "mps"
        
        return memory_info
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up Transformers backend")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self._is_initialized = False