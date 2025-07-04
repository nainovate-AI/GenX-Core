"""
Model Manager for LLM Service with Dynamic Loading
Manages model lifecycle, loading, and resource allocation
"""
import asyncio
import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import time
import uuid
import psutil
import torch
from datetime import datetime

from ..backends.base import LLMBackend, ModelInfo
from .backend_factory import BackendFactory

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Information about a loaded model"""
    model_id: str  # Unique ID for this instance
    model_name: str  # Model name (e.g., "gpt2")
    backend_type: str  # Backend used (transformers, vllm, etc.)
    device: str  # Device loaded on (cuda:0, cpu, mps, etc.)
    backend: LLMBackend
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    backend_config: Dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Enhanced Model Manager with dynamic loading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self.default_model_id = config.get('default_model_id', 'gpt2')
        self.max_loaded_models = config.get('max_loaded_models', 5)
        self._user_history: Dict[str, List[Dict[str, str]]] = {}  # Simple in-memory history
        
    async def load_model(
        self,
        model_name: str,
        backend: str,
        device: str = "auto",
        backend_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[LoadedModel]]:
        """
        Dynamically load a model with specified backend and device
        
        Returns:
            Tuple of (model_id, LoadedModel) if successful, (None, None) if failed
        """
        async with self._lock:
            # Check if model already loaded with same config
            for mid, loaded in self.models.items():
                if (loaded.model_name == model_name and 
                    loaded.backend_type == backend and 
                    loaded.device == device):
                    logger.info(f"Model {model_name} already loaded as {mid}")
                    return mid, loaded
            
            # Check capacity
            if len(self.models) >= self.max_loaded_models:
                await self._unload_least_recently_used()
            
            # Generate unique model ID
            model_id = f"{model_name}_{backend}_{device}_{uuid.uuid4().hex[:8]}"
            
            try:
                # Prepare backend configuration
                full_config = {
                    'device': device,
                    'device_map': backend_config.get('device_map', 'auto') if backend_config else 'auto',
                    'trust_remote_code': backend_config.get('trust_remote_code', False) if backend_config else False,
                    'load_in_8bit': backend_config.get('load_in_8bit', False) if backend_config else False,
                    'load_in_4bit': backend_config.get('load_in_4bit', False) if backend_config else False,
                }
                
                if backend_config:
                    full_config.update(backend_config)
                
                # Create backend
                logger.info(f"Loading model {model_name} with {backend} backend on {device}")
                backend_instance = BackendFactory.create_backend(
                    backend_type=backend,
                    model_id=model_name,
                    auto_select=False,  # Use specified backend
                    **full_config
                )
                
                # Initialize backend
                success = await backend_instance.initialize()
                if not success:
                    logger.error(f"Failed to initialize {model_name} with {backend}")
                    return None, None
                
                # Store loaded model
                loaded_model = LoadedModel(
                    model_id=model_id,
                    model_name=model_name,
                    backend_type=backend,
                    device=self._get_actual_device(backend_instance),
                    backend=backend_instance,
                    backend_config=full_config
                )
                
                self.models[model_id] = loaded_model
                logger.info(f"Successfully loaded {model_name} as {model_id}")
                
                return model_id, loaded_model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None, None
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a specific model"""
        async with self._lock:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found")
                return False
            
            try:
                loaded_model = self.models[model_id]
                await loaded_model.backend.cleanup()
                del self.models[model_id]
                logger.info(f"Unloaded model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {e}")
                return False
    
    async def get_model(self, model_id: Optional[str] = None) -> Optional[LLMBackend]:
        """
        Get a model backend by ID or return default
        """
        async with self._lock:
            if model_id:
                if model_id in self.models:
                    loaded_model = self.models[model_id]
                    loaded_model.last_used = time.time()
                    loaded_model.request_count += 1
                    return loaded_model.backend
                else:
                    logger.warning(f"Model {model_id} not found")
                    return None
            
            # Return default model or first available
            if self.models:
                # Try to find default model
                for mid, loaded in self.models.items():
                    if loaded.model_name == self.default_model_id:
                        loaded.last_used = time.time()
                        loaded.request_count += 1
                        return loaded.backend
                
                # Return first available model
                first_model = list(self.models.values())[0]
                first_model.last_used = time.time()
                first_model.request_count += 1
                return first_model.backend
            
            # No models loaded, try to load default
            logger.info("No models loaded, attempting to load default model")
            model_id, loaded = await self.load_model(
                self.default_model_id,
                self.config.get('backend_type', 'transformers'),
                'auto'
            )
            return loaded.backend if loaded else None
    
    def _get_actual_device(self, backend: LLMBackend) -> str:
        """Get the actual device the model is loaded on"""
        if hasattr(backend, 'device'):
            return str(backend.device)
        elif hasattr(backend, 'model') and hasattr(backend.model, 'device'):
            return str(backend.model.device)
        return "unknown"
    
    async def _unload_least_recently_used(self):
        """Unload the least recently used model"""
        if not self.models:
            return
        
        lru_model_id = min(
            self.models.keys(),
            key=lambda k: self.models[k].last_used
        )
        
        await self.unload_model(lru_model_id)
    
    async def get_loaded_models_info(
        self,
        backend_filter: Optional[str] = None,
        device_filter: Optional[str] = None,
        include_stats: bool = True
    ) -> Dict[str, Any]:
        """Get information about loaded models"""
        models_info = []
        
        for model_id, loaded_model in self.models.items():
            # Apply filters
            if backend_filter and loaded_model.backend_type != backend_filter:
                continue
            if device_filter and loaded_model.device != device_filter:
                continue
            
            # Get model info from backend
            backend_info = loaded_model.backend.get_model_info()
            
            info = {
                'model_id': model_id,
                'model_name': loaded_model.model_name,
                'backend': loaded_model.backend_type,
                'device': loaded_model.device,
                'loaded_at': datetime.fromtimestamp(loaded_model.loaded_at).isoformat(),
                'status': {
                    'is_loaded': backend_info.is_loaded if backend_info else False,
                    'is_available': backend_info.is_available if backend_info else False,
                    'current_load': backend_info.current_load if backend_info else 0.0
                }
            }
            
            if include_stats:
                info['stats'] = {
                    'total_requests': loaded_model.request_count,
                    'total_tokens_generated': loaded_model.total_tokens,
                    'avg_response_time_ms': (
                        loaded_model.total_response_time / loaded_model.request_count * 1000
                        if loaded_model.request_count > 0 else 0
                    ),
                    'last_used': datetime.fromtimestamp(loaded_model.last_used).isoformat()
                }
            
            # Add capabilities
            if backend_info:
                info['capabilities'] = {
                    'max_context_length': getattr(backend_info, 'max_context_length', 2048),
                    'features': backend_info.capabilities,
                    'model_type': 'causal_lm'  # Could be detected from model
                }
            
            # Memory usage (if available)
            try:
                health = await loaded_model.backend.health_check()
                info['memory_usage'] = {
                    'gpu_memory_mb': health.get('gpu_memory_used_gb', 0) * 1024,
                    'ram_mb': 0  # Would need to track this
                }
            except:
                pass
            
            models_info.append(info)
        
        # Get system info
        system_info = self._get_system_info()
        
        return {
            'models': models_info,
            'system_info': system_info
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        system_info = {
            'gpus': [],
            'cpu': {
                'cores': psutil.cpu_count(),
                'utilization_percent': psutil.cpu_percent(interval=0.1)
            },
            'memory': {
                'total_ram_mb': psutil.virtual_memory().total // (1024 * 1024),
                'used_ram_mb': psutil.virtual_memory().used // (1024 * 1024),
                'available_ram_mb': psutil.virtual_memory().available // (1024 * 1024)
            }
        }
        
        # Get GPU info if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                system_info['gpus'].append({
                    'index': i,
                    'name': props.name,
                    'total_memory_mb': props.total_memory // (1024 * 1024),
                    'used_memory_mb': torch.cuda.memory_allocated(i) // (1024 * 1024),
                    'utilization_percent': 0  # Would need nvidia-ml-py for this
                })
        
        return system_info
    
    def update_stats(self, model_id: str, tokens: int, response_time: float):
        """Update usage statistics for a model"""
        if model_id in self.models:
            self.models[model_id].total_tokens += tokens
            self.models[model_id].total_response_time += response_time
    
    def get_user_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a user"""
        return self._user_history.get(user_id, [])
    
    def add_to_history(self, user_id: str, role: str, content: str):
        """Add a message to user's history"""
        if user_id not in self._user_history:
            self._user_history[user_id] = []
        
        self._user_history[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages per user
        if len(self._user_history[user_id]) > 20:
            self._user_history[user_id] = self._user_history[user_id][-20:]
    
    def clear_user_history(self, user_id: str):
        """Clear history for a user"""
        if user_id in self._user_history:
            del self._user_history[user_id]
    
    async def cleanup_all(self):
        """Cleanup all loaded models"""
        logger.info("Cleaning up all models")
        
        model_ids = list(self.models.keys())
        for model_id in model_ids:
            await self.unload_model(model_id)
        
        self._user_history.clear()