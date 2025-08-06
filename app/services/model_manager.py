import asyncio
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict
import mlx.core as mx
from mlx_lm import load, generate
import psutil
import threading
import time

from app.core.config import get_settings
from app.core.logging import logger


class ModelManager:
    def __init__(self):
        self.settings = get_settings()
        self._models_cache: OrderedDict[str, Tuple[Any, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._last_access: Dict[str, float] = {}
        self._load_times: Dict[str, float] = {}
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    def _should_evict_model(self) -> bool:
        """Check if we need to evict a model from cache"""
        memory = self._get_memory_usage()
        return (
            len(self._models_cache) >= self.settings.max_model_cache_size or
            memory["available_gb"] < 2.0  # Keep at least 2GB free
        )
    
    def _evict_least_recently_used(self):
        """Evict the least recently used model from cache"""
        if not self._models_cache:
            return
            
        # Find LRU model
        lru_model = min(self._last_access.items(), key=lambda x: x[1])[0]
        
        with self._cache_lock:
            if lru_model in self._models_cache:
                del self._models_cache[lru_model]
                del self._last_access[lru_model]
                del self._load_times[lru_model]
                logger.info(f"Evicted model {lru_model} from cache")
    
    async def get_model(self, model_name: str) -> Tuple[Any, Any]:
        """Get a model from cache or load it"""
        # Check cache first
        with self._cache_lock:
            if model_name in self._models_cache:
                self._last_access[model_name] = time.time()
                # Move to end (most recently used)
                self._models_cache.move_to_end(model_name)
                logger.debug(f"Model {model_name} retrieved from cache")
                return self._models_cache[model_name]
        
        # Get or create loading lock for this model
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = threading.Lock()
        
        # Load model with lock to prevent duplicate loading
        with self._loading_locks[model_name]:
            # Double-check cache after acquiring lock
            with self._cache_lock:
                if model_name in self._models_cache:
                    self._last_access[model_name] = time.time()
                    return self._models_cache[model_name]
            
            # Load model
            logger.info(f"Loading model {model_name}")
            start_time = time.time()
            
            try:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                model, tokenizer = await loop.run_in_executor(
                    None, 
                    load, 
                    model_name
                )
                
                load_time = time.time() - start_time
                self._load_times[model_name] = load_time
                logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
                
                # Check if we need to evict before adding
                while self._should_evict_model():
                    self._evict_least_recently_used()
                
                # Add to cache
                with self._cache_lock:
                    self._models_cache[model_name] = (model, tokenizer)
                    self._last_access[model_name] = time.time()
                
                return model, tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
    
    async def generate_response(
        self, 
        model_name: str, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """Generate response using the specified model"""
        model, tokenizer = await self.get_model(model_name)
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            generate,
            model,
            tokenizer,
            prompt,
            {"temp": temperature, "max_tokens": max_tokens, "top_p": top_p}
        )
        
        return response
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache"""
        with self._cache_lock:
            return {
                "cached_models": list(self._models_cache.keys()),
                "cache_size": len(self._models_cache),
                "max_cache_size": self.settings.max_model_cache_size,
                "load_times": self._load_times.copy(),
                "memory_usage": self._get_memory_usage()
            }
    
    def clear_cache(self):
        """Clear all models from cache"""
        with self._cache_lock:
            self._models_cache.clear()
            self._last_access.clear()
            self._load_times.clear()
            logger.info("Model cache cleared")


# Singleton instance
model_manager = ModelManager()