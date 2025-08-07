import asyncio
import os
import platform
from typing import Dict, Optional, Tuple, Any, AsyncGenerator
from collections import OrderedDict
import psutil
import threading
import time

from app.core.config import get_settings
from app.core.logging import logger

# Backend detection and imports
def detect_backend():
    """Detect which backend to use based on platform and available libraries"""
    backend = os.getenv("INFERENCE_BACKEND", "auto").lower()
    
    if backend != "auto":
        return backend
    
    # Check if we're on Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        try:
            import mlx.core as mx
            return "mlx"
        except ImportError:
            logger.warning("MLX not available on Apple Silicon, falling back to vLLM")
    
    # Try vLLM for other platforms
    try:
        import vllm
        return "vllm"
    except ImportError:
        logger.warning("vLLM not available, using CPU fallback")
        return "cpu"

BACKEND = detect_backend()
logger.info(f"Using inference backend: {BACKEND}")

# Import appropriate modules based on backend
if BACKEND == "mlx":
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        logger.info("MLX backend loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import MLX: {e}")
        BACKEND = "cpu"

elif BACKEND == "vllm":
    try:
        from vllm import LLM, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.utils import random_uuid
        logger.info("vLLM backend loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import vLLM: {e}")
        BACKEND = "cpu"

# CPU fallback imports
if BACKEND == "cpu":
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        logger.info("Using Transformers CPU backend")
    except ImportError as e:
        logger.error(f"Failed to import Transformers: {e}")
        # Keep MLX as fallback if available
        if platform.system() == "Darwin":
            try:
                import mlx.core as mx
                from mlx_lm import load, generate
                BACKEND = "mlx"
                logger.info("Falling back to MLX")
            except ImportError:
                raise RuntimeError("No suitable inference backend available")


class ModelManager:
    def __init__(self):
        self.settings = get_settings()
        self.backend = BACKEND
        self._models_cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_lock = threading.Lock()  # Use threading lock
        self._last_access: Dict[str, float] = {}
        self._load_times: Dict[str, float] = {}
        # Global lock for MLX operations to prevent Metal GPU conflicts (lazy init)
        self._mlx_lock = None
        
        # Backend-specific initialization
        if self.backend == "vllm":
            self._vllm_engines: Dict[str, Any] = {}
        
        logger.info(f"ModelManager initialized with {self.backend} backend")
        
    def _get_mlx_lock(self):
        """Get MLX lock with lazy initialization"""
        if self.backend == "mlx" and self._mlx_lock is None:
            self._mlx_lock = asyncio.Lock()
        return self._mlx_lock
        
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
                
                # Record cache eviction
                from app.utils.metrics import metrics_collector
                metrics_collector.record_cache_operation("eviction")
    
    async def get_model(self, model_name: str) -> Tuple[Any, Any]:
        """Get a model from cache or load it"""
        # Simple cache check - no complex locking
        with self._cache_lock:
            if model_name in self._models_cache:
                self._last_access[model_name] = time.time()
                # Move to end (most recently used)
                self._models_cache.move_to_end(model_name)
                logger.debug(f"Model {model_name} retrieved from cache")
                
                # Record cache hit
                from app.utils.metrics import metrics_collector
                metrics_collector.record_cache_operation("hit")
                
                return self._models_cache[model_name]
        
        # Simple approach: just load the model (allow concurrent loads for now)
        logger.info(f"Loading model {model_name}")
        from app.utils.metrics import metrics_collector
        metrics_collector.record_cache_operation("miss")
        metrics_collector.record_cache_operation("load")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # Load model based on backend (with MLX lock if needed)
        mlx_lock = self._get_mlx_lock()
        if mlx_lock:
            async with mlx_lock:
                model_data = await self._load_model_backend_specific(model_name)
        else:
            model_data = await self._load_model_backend_specific(model_name)
        
        load_time = time.time() - start_time
        self._load_times[model_name] = load_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s using {self.backend} backend")
        
        # Check if we need to evict before adding
        while self._should_evict_model():
            self._evict_least_recently_used()
        
        # Add to cache
        with self._cache_lock:
            self._models_cache[model_name] = model_data
            self._last_access[model_name] = time.time()
        
        # Estimate memory usage (rough calculation)
        memory_after = psutil.virtual_memory().used
        estimated_model_memory = max(0, memory_after - start_memory)
        
        # Record model memory footprint
        from app.utils.metrics import model_memory_usage_bytes
        model_memory_usage_bytes.labels(model_name=model_name).set(estimated_model_memory)
        
        # Record that model was loaded
        metrics_collector.record_model_loaded(model_name)
        
        return model_data
    
    def _clean_response(self, response: str) -> str:
        """Clean and post-process the generated response"""
        # Stop at common conversation boundaries
        stop_sequences = [
            "<|user|>", "<|system|>", "<|assistant|>",
            "\nUser:", "\nAssistant:", "\nSystem:",
            "User:", "Assistant:", "System:"
        ]
        
        # Find the earliest stop sequence
        min_pos = len(response)
        for stop_seq in stop_sequences:
            pos = response.find(stop_seq)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        # Truncate at stop sequence
        if min_pos < len(response):
            response = response[:min_pos]
        
        # Clean up trailing whitespace and incomplete sentences
        response = response.strip()
        
        # Remove incomplete trailing sentences if they end abruptly
        if response and not response[-1] in '.!?':
            # Find last complete sentence
            last_complete = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?')
            )
            if last_complete > len(response) * 0.5:  # Only if we have substantial content
                response = response[:last_complete + 1]
        
        return response

    async def _load_model_backend_specific(self, model_name: str) -> Any:
        """Backend-specific model loading"""
        
        if self.backend == "mlx":
            # MLX loading (synchronous, run in thread pool)
            def load_mlx_model():
                return load(model_name)
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, load_mlx_model)
        
        elif self.backend == "vllm":
            # vLLM loading
            engine_args = AsyncEngineArgs(
                model=model_name,
                tensor_parallel_size=1,
                dtype="auto",
                max_model_len=getattr(self.settings, 'max_model_len', 2048),
                gpu_memory_utilization=getattr(self.settings, 'gpu_memory_fraction', 0.8),
                disable_log_stats=True
            )
            
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._vllm_engines[model_name] = engine
            return engine
        
        elif self.backend == "cpu":
            # CPU loading with Transformers
            def load_cpu_model():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                return {"tokenizer": tokenizer, "model": model}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, load_cpu_model)
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

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
        model_data = await self.get_model(model_name)
        
        if self.backend == "mlx":
            # MLX generation with lock to prevent Metal GPU conflicts
            model, tokenizer = model_data
            loop = asyncio.get_event_loop()
            
            mlx_lock = self._get_mlx_lock()
            if mlx_lock:
                async with mlx_lock:
                    response = await loop.run_in_executor(
                        None,
                        generate,
                        model,
                        tokenizer,
                        prompt,
                        {"temp": temperature, "max_tokens": max_tokens, "top_p": top_p}
                    )
            else:
                response = await loop.run_in_executor(
                    None,
                    generate,
                    model,
                    tokenizer,
                    prompt,
                    {"temp": temperature, "max_tokens": max_tokens, "top_p": top_p}
                )
        
        elif self.backend == "vllm":
            # vLLM generation
            engine = model_data
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=["User:", "Human:"]
            )
            
            request_id = random_uuid()
            results = []
            async for result in engine.generate(prompt, sampling_params, request_id):
                results.append(result)
            
            if results:
                response = results[-1].outputs[0].text
            else:
                response = ""
        
        elif self.backend == "cpu":
            # CPU generation with Transformers
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            
            def cpu_generate():
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = generator(prompt, return_full_text=False)
                return result[0]["generated_text"]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, cpu_generate)
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Clean and post-process the response
        cleaned_response = self._clean_response(response)
        return cleaned_response
    
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