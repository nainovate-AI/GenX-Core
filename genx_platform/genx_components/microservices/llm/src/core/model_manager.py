"""
genx_platform/genx_components/microservices/llm/src/core/model_manager.py
Model Manager for LLM Service
Manages model lifecycle and resource allocation
"""
import asyncio
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import time

from ..backends.base import LLMBackend, ModelInfo
from .backend_factory import BackendFactory

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Information about a loaded model"""
    model_id: str
    backend: LLMBackend
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    total_tokens: int = 0


class ModelManager:
    """Manages loaded models and their lifecycle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self.default_model_id = config.get('default_model_id', 'gpt2')
        self.max_loaded_models = config.get('max_loaded_models', 3)
        
    async def get_model(self, model_id: Optional[str] = None) -> LLMBackend:
        """
        Get a model backend, loading it if necessary
        
        Args:
            model_id: Model to get (uses default if None)
            
        Returns:
            LLMBackend instance
        """
        model_id = model_id or self.default_model_id
        
        async with self._lock:
            # Check if model is already loaded
            if model_id in self.models:
                loaded_model = self.models[model_id]
                loaded_model.last_used = time.time()
                loaded_model.request_count += 1
                return loaded_model.backend
            
            # Check if we need to unload a model
            if len(self.models) >= self.max_loaded_models:
                await self._unload_least_recently_used()
            
            # Load the model
            backend = await self._load_model(model_id)
            
            # Store loaded model info
            self.models[model_id] = LoadedModel(
                model_id=model_id,
                backend=backend
            )
            
            return backend
    
    async def _load_model(self, model_id: str) -> LLMBackend:
        """Load a model"""
        logger.info(f"Loading model: {model_id}")
        
        # Get model-specific config
        model_config = self.config.get('model_configs', {}).get(model_id, {})
        
        # Merge with global config
        backend_config = {
            'device_map': self.config.get('device_map', 'auto'),
            'trust_remote_code': self.config.get('trust_remote_code', False),
            'load_in_8bit': self.config.get('load_in_8bit', False),
            'load_in_4bit': self.config.get('load_in_4bit', False),
            **model_config
        }
        
        # Create backend
        backend = BackendFactory.create_backend(
            backend_type=self.config.get('backend_type'),
            model_id=model_id,
            auto_select=self.config.get('auto_select_backend', True),
            **backend_config
        )
        
        # Initialize backend
        success = await backend.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize model: {model_id}")
        
        logger.info(f"Successfully loaded model: {model_id}")
        return backend
    
    async def _unload_least_recently_used(self):
        """Unload the least recently used model"""
        if not self.models:
            return
        
        # Find LRU model
        lru_model_id = min(
            self.models.keys(),
            key=lambda k: self.models[k].last_used
        )
        
        logger.info(f"Unloading LRU model: {lru_model_id}")
        
        # Cleanup backend
        loaded_model = self.models[lru_model_id]
        await loaded_model.backend.cleanup()
        
        # Remove from cache
        del self.models[lru_model_id]
    
    async def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all loaded models"""
        models_info = []
        
        for model_id, loaded_model in self.models.items():
            model_info = loaded_model.backend.get_model_info()
            if model_info:
                models_info.append({
                    'model_id': model_id,
                    'provider': model_info.provider,
                    'loaded_at': loaded_model.loaded_at,
                    'last_used': loaded_model.last_used,
                    'request_count': loaded_model.request_count,
                    'total_tokens': loaded_model.total_tokens,
                    'is_available': model_info.is_available
                })
        
        return models_info
    
    async def preload_models(self, model_ids: List[str]):
        """Preload a list of models"""
        logger.info(f"Preloading models: {model_ids}")
        
        tasks = []
        for model_id in model_ids:
            tasks.append(self.get_model(model_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for model_id, result in zip(model_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to preload {model_id}: {result}")
            else:
                logger.info(f"Successfully preloaded {model_id}")
    
    async def cleanup_all(self):
        """Cleanup all loaded models"""
        logger.info("Cleaning up all models")
        
        for loaded_model in self.models.values():
            try:
                await loaded_model.backend.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {loaded_model.model_id}: {e}")
        
        self.models.clear()
    
    def update_token_usage(self, model_id: str, tokens: int):
        """Update token usage statistics"""
        if model_id in self.models:
            self.models[model_id].total_tokens += tokens