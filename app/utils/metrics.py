from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import mlx.core as mx
from typing import Dict, Any
import time

from app.core.config import get_settings


# Metrics
request_count = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

model_load_duration = Histogram(
    'model_load_duration_seconds',
    'Model loading time',
    ['model_name']
)

inference_duration = Histogram(
    'inference_duration_seconds',
    'Model inference time',
    ['model_name']
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)

model_cache_size = Gauge(
    'model_cache_size',
    'Number of models in cache'
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type']
)

gpu_memory_usage_bytes = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes'
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

# Token and inference performance metrics
token_generation_rate = Histogram(
    'token_generation_rate_tokens_per_second',
    'Token generation rate in tokens per second',
    ['model_name'],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, float('inf')]
)

time_to_first_token = Histogram(
    'time_to_first_token_seconds',
    'Time to generate first token',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
)

concurrent_inferences = Gauge(
    'concurrent_inferences',
    'Number of concurrent inference requests'
)

inference_queue_depth = Gauge(
    'inference_queue_depth',
    'Number of requests waiting in queue'
)

# Token usage metrics
tokens_processed_total = Counter(
    'tokens_processed_total',
    'Total tokens processed',
    ['model_name', 'type']  # type: prompt, completion
)

tokens_per_request = Histogram(
    'tokens_per_request',
    'Token count per request',
    ['model_name', 'type'],  # type: prompt, completion
    buckets=[10, 50, 100, 250, 500, 1000, 2000, 4000, float('inf')]
)

# Enhanced error metrics
errors_total = Counter(
    'errors_total',
    'Total errors by type and model',
    ['error_type', 'model_name', 'endpoint']
)

# Model cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Model cache operations',
    ['operation']  # hit, miss, eviction, load
)

# Per-model memory footprint
model_memory_usage_bytes = Gauge(
    'model_memory_usage_bytes',
    'Memory usage per loaded model (estimated)',
    ['model_name']
)

# API key usage metrics
api_key_token_usage = Counter(
    'api_key_token_usage_total',
    'Total tokens used per API key',
    ['api_key_prefix', 'model_name', 'type']  # type: prompt, completion
)

api_key_requests_total = Counter(
    'api_key_requests_total',
    'Total requests per API key',
    ['api_key_prefix', 'model_name', 'status']
)


class MetricsCollector:
    def __init__(self):
        self.settings = get_settings()
        self._start_time = time.time()
        self._request_count = 0
        self._failed_requests = 0
        self._total_latency = 0.0
        self._active_inferences = 0
        self._queue_depth = 0
        
    def update_memory_metrics(self):
        """Update memory usage metrics"""
        memory = psutil.virtual_memory()
        memory_usage_bytes.labels(type='total').set(memory.total)
        memory_usage_bytes.labels(type='used').set(memory.used)
        memory_usage_bytes.labels(type='available').set(memory.available)
        
        # Update CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking call
        cpu_usage_percent.set(cpu_percent)
        
        # MLX GPU memory tracking
        try:
            import mlx.core as mx
            # Try to get MLX memory info
            if mx.default_device().type.name == "gpu":
                # MLX doesn't directly expose GPU memory, but we can try system tools
                try:
                    # For Apple Silicon, we can use system_profiler or Activity Monitor APIs
                    # This is a basic implementation - could be enhanced
                    import subprocess
                    _ = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                     capture_output=True, text=True, timeout=2)
                    # Parse memory info from output if available
                    # For now, we'll set a placeholder value
                    gpu_memory_usage_bytes.set(0)  # Will be enhanced later
                except Exception:
                    gpu_memory_usage_bytes.set(0)
            else:
                gpu_memory_usage_bytes.set(0)
        except Exception:
            gpu_memory_usage_bytes.set(0)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics"""
        request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        self._request_count += 1
        self._total_latency += duration
        if status >= 400:
            self._failed_requests += 1

    def record_inference_start(self):
        """Record start of inference"""
        self._active_inferences += 1
        concurrent_inferences.set(self._active_inferences)

    def record_inference_end(self):
        """Record end of inference"""
        self._active_inferences = max(0, self._active_inferences - 1)
        concurrent_inferences.set(self._active_inferences)

    def record_token_metrics(self, model_name: str, prompt_tokens: int, completion_tokens: int, 
                           generation_time: float, first_token_time: float = None, api_key_prefix: str = None):
        """Record token-related metrics"""
        # Token counts
        tokens_processed_total.labels(model_name=model_name, type='prompt').inc(prompt_tokens)
        tokens_processed_total.labels(model_name=model_name, type='completion').inc(completion_tokens)
        
        # Token distributions
        tokens_per_request.labels(model_name=model_name, type='prompt').observe(prompt_tokens)
        tokens_per_request.labels(model_name=model_name, type='completion').observe(completion_tokens)
        
        # API key token usage
        if api_key_prefix:
            api_key_token_usage.labels(api_key_prefix=api_key_prefix, model_name=model_name, type='prompt').inc(prompt_tokens)
            api_key_token_usage.labels(api_key_prefix=api_key_prefix, model_name=model_name, type='completion').inc(completion_tokens)
        
        # Token generation rate (tokens per second)
        if generation_time > 0 and completion_tokens > 0:
            rate = completion_tokens / generation_time
            token_generation_rate.labels(model_name=model_name).observe(rate)
        
        # Time to first token
        if first_token_time is not None:
            time_to_first_token.labels(model_name=model_name).observe(first_token_time)

    def record_api_key_request(self, api_key_prefix: str, model_name: str, status: str):
        """Record API key request"""
        api_key_requests_total.labels(api_key_prefix=api_key_prefix, model_name=model_name, status=status).inc()

    def record_error(self, error_type: str, model_name: str = "unknown", endpoint: str = "unknown"):
        """Record error with categorization"""
        errors_total.labels(error_type=error_type, model_name=model_name, endpoint=endpoint).inc()

    def record_cache_operation(self, operation: str):
        """Record cache operation (hit, miss, eviction, load)"""
        cache_operations_total.labels(operation=operation).inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics"""
        uptime = time.time() - self._start_time
        avg_latency = self._total_latency / max(1, self._request_count)
        
        memory = psutil.virtual_memory()
        
        return {
            "uptime_seconds": uptime,
            "requests_total": self._request_count,
            "requests_failed": self._failed_requests,
            "average_latency_ms": avg_latency * 1000,
            "memory_usage_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        self.update_memory_metrics()
        return generate_latest().decode('utf-8')


# Singleton instance
metrics_collector = MetricsCollector()