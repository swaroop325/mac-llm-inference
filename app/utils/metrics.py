from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
from typing import Dict, Any
import time
import os

# Conditional MLX import
try:
    if os.getenv("INFERENCE_BACKEND", "auto") == "mlx":
        import mlx.core as mx
    else:
        mx = None
except ImportError:
    mx = None

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
    ['api_key_prefix', 'api_key_name', 'model_name', 'type']  # type: prompt, completion
)

api_key_requests_total = Counter(
    'api_key_requests_total',
    'Total requests per API key',
    ['api_key_prefix', 'api_key_name', 'model_name', 'status']
)

# Enhanced API key metrics
api_key_usage_by_endpoint = Counter(
    'api_key_usage_by_endpoint_total',
    'API key usage by endpoint',
    ['api_key_prefix', 'api_key_name', 'endpoint', 'method']
)

api_key_last_used_timestamp = Gauge(
    'api_key_last_used_timestamp_seconds',
    'Timestamp when API key was last used',
    ['api_key_prefix', 'api_key_name']
)

api_key_rate_limit_hits = Counter(
    'api_key_rate_limit_hits_total',
    'Number of times API key hit rate limits',
    ['api_key_prefix', 'api_key_name']
)

# New useful metrics

# Response quality metrics
response_truncated_total = Counter(
    'response_truncated_total',
    'Responses truncated due to max_tokens limit',
    ['model_name']
)

# Model switching metrics
model_switch_duration = Histogram(
    'model_switch_duration_seconds',
    'Time taken to switch between models',
    ['from_model', 'to_model']
)

# Request size metrics
request_size_bytes = Histogram(
    'request_size_bytes',
    'Size of incoming requests in bytes',
    ['endpoint'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, float('inf')]
)

response_size_bytes = Histogram(
    'response_size_bytes', 
    'Size of responses in bytes',
    ['endpoint'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, float('inf')]
)

# Streaming metrics
streaming_requests_total = Counter(
    'streaming_requests_total',
    'Total streaming requests',
    ['model_name', 'status']
)

streaming_chunk_latency = Histogram(
    'streaming_chunk_latency_seconds',
    'Time between streaming chunks',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
)

# Temperature and sampling metrics
temperature_distribution = Histogram(
    'temperature_distribution',
    'Distribution of temperature values used',
    ['model_name'],
    buckets=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, float('inf')]
)

top_p_distribution = Histogram(
    'top_p_distribution',
    'Distribution of top_p values used',
    ['model_name'],
    buckets=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
)

# Model-specific performance
model_warmup_time = Gauge(
    'model_warmup_time_seconds',
    'Time since model was loaded (warmup indicator)',
    ['model_name']
)

# API Response Time Tracking
api_response_time_min = Gauge(
    'api_response_time_min_seconds',
    'Minimum API response time for chat completions',
    ['model_name']
)

api_response_time_max = Gauge(
    'api_response_time_max_seconds', 
    'Maximum API response time for chat completions',
    ['model_name']
)

# Keep the old queue time metric for backwards compatibility but mark as deprecated
request_queue_time = Histogram(
    'request_queue_time_seconds',
    'Time spent waiting in queue before processing (deprecated)',
    ['model_name'],
    buckets=[0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
)

rejected_requests_total = Counter(
    'rejected_requests_total',
    'Requests rejected due to overload',
    ['reason']  # rate_limit, memory_pressure, queue_full
)

# Context window utilization
context_utilization_ratio = Histogram(
    'context_utilization_ratio',
    'Ratio of tokens used vs context window size',
    ['model_name'],
    buckets=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
)

# GPU utilization (Apple Silicon specific)
metal_memory_usage_bytes = Gauge(
    'metal_memory_usage_bytes',
    'Metal GPU memory usage in bytes'
)

metal_memory_peak_bytes = Gauge(
    'metal_memory_peak_bytes',
    'Peak Metal GPU memory usage in bytes'
)

# System health
disk_usage_bytes = Gauge(
    'disk_usage_bytes',
    'Disk usage for model cache directory',
    ['type']  # total, used, free
)

# Network I/O
network_io_bytes = Counter(
    'network_io_bytes_total',
    'Network I/O in bytes',
    ['direction']  # sent, received
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
        self._last_memory_update = 0
        self._memory_update_interval = 5.0  # Update memory metrics every 5 seconds (reduced frequency)
        self._model_load_times = {}  # Track when each model was loaded
        self._last_active_model = None
        self._model_switch_start = None
        self._last_net_io = {'bytes_sent': 0, 'bytes_recv': 0}  # Track network I/O counters
        self._last_disk_update = 0
        self._disk_update_interval = 30.0  # Update disk usage every 30 seconds
        # Track min/max response times per model
        self._response_times: Dict[str, Dict[str, float]] = {}  # {model_name: {"min": float, "max": float}}
        
    def update_memory_metrics(self, force=False):
        """Update memory usage metrics with rate limiting"""
        current_time = time.time()
        
        # Rate limit memory updates unless forced
        if not force and (current_time - self._last_memory_update) < self._memory_update_interval:
            return
        
        self._last_memory_update = current_time
        
        memory = psutil.virtual_memory()
        memory_usage_bytes.labels(type='total').set(memory.total)
        memory_usage_bytes.labels(type='used').set(memory.used)
        memory_usage_bytes.labels(type='available').set(memory.available)
        
        # Update CPU usage with non-blocking call
        cpu_percent = psutil.cpu_percent(interval=0)  # Immediate non-blocking
        cpu_usage_percent.set(cpu_percent)
        
        # MLX GPU memory tracking
        try:
            if mx:
                # MLX provides memory info directly
                # This is much faster than calling system_profiler
                import mlx.core as mx
                # Get current and peak memory usage from MLX (using new API)
                current_memory = mx.get_active_memory()
                peak_memory = mx.get_peak_memory()
                
                # Set both GPU and Metal-specific metrics
                gpu_memory_usage_bytes.set(current_memory)
                metal_memory_usage_bytes.set(current_memory)
                metal_memory_peak_bytes.set(peak_memory)
        except Exception:
            # If MLX memory tracking fails, set to 0
            gpu_memory_usage_bytes.set(0)
            metal_memory_usage_bytes.set(0)
            metal_memory_peak_bytes.set(0)
        
        # Update disk usage for model cache (rate limited)
        if force or (current_time - self._last_disk_update) >= self._disk_update_interval:
            try:
                import shutil
                cache_dir = self.settings.model_cache_dir
                if cache_dir:
                    # Create cache directory if it doesn't exist
                    os.makedirs(cache_dir, exist_ok=True)
                    disk_usage = shutil.disk_usage(cache_dir)
                    disk_usage_bytes.labels(type='total').set(disk_usage.total)
                    disk_usage_bytes.labels(type='used').set(disk_usage.used)
                    disk_usage_bytes.labels(type='free').set(disk_usage.free)
                    self._last_disk_update = current_time
            except Exception as e:
                # Fallback to current directory
                try:
                    import shutil
                    disk_usage = shutil.disk_usage('.')
                    disk_usage_bytes.labels(type='total').set(disk_usage.total)
                    disk_usage_bytes.labels(type='used').set(disk_usage.used)
                    disk_usage_bytes.labels(type='free').set(disk_usage.free)
                    self._last_disk_update = current_time
                except Exception:
                    pass
        
        # Update network I/O stats (incremental)
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                # Calculate incremental values since last update
                sent_delta = net_io.bytes_sent - self._last_net_io['bytes_sent']
                recv_delta = net_io.bytes_recv - self._last_net_io['bytes_recv']
                
                # Only increment if positive (avoid negative values on counter resets)
                if sent_delta > 0:
                    network_io_bytes.labels(direction='sent').inc(sent_delta)
                if recv_delta > 0:
                    network_io_bytes.labels(direction='received').inc(recv_delta)
                
                # Update last known values
                self._last_net_io['bytes_sent'] = net_io.bytes_sent
                self._last_net_io['bytes_recv'] = net_io.bytes_recv
        except Exception:
            pass
    
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
        # Removed update_memory_metrics() call - was causing server hangs

    def record_inference_end(self):
        """Record end of inference"""
        self._active_inferences = max(0, self._active_inferences - 1)
        concurrent_inferences.set(self._active_inferences)
        # Removed update_memory_metrics() call - was causing server hangs

    def record_token_metrics(self, model_name: str, prompt_tokens: int, completion_tokens: int, 
                           generation_time: float, first_token_time: float = None, api_key_prefix: str = None,
                           api_key_name: str = None, max_tokens: int = None, actual_tokens: int = None, context_window: int = 4096):
        """Record token-related metrics"""
        # Token counts
        tokens_processed_total.labels(model_name=model_name, type='prompt').inc(prompt_tokens)
        tokens_processed_total.labels(model_name=model_name, type='completion').inc(completion_tokens)
        
        # Token distributions
        tokens_per_request.labels(model_name=model_name, type='prompt').observe(prompt_tokens)
        tokens_per_request.labels(model_name=model_name, type='completion').observe(completion_tokens)
        
        # API key token usage
        if api_key_prefix:
            api_key_token_usage.labels(
                api_key_prefix=api_key_prefix,
                api_key_name=api_key_name or "unknown",
                model_name=model_name,
                type='prompt'
            ).inc(prompt_tokens)
            api_key_token_usage.labels(
                api_key_prefix=api_key_prefix,
                api_key_name=api_key_name or "unknown", 
                model_name=model_name,
                type='completion'
            ).inc(completion_tokens)
        
        # Token generation rate (tokens per second)
        if generation_time > 0 and completion_tokens > 0:
            rate = completion_tokens / generation_time
            token_generation_rate.labels(model_name=model_name).observe(rate)
        
        # Time to first token
        if first_token_time is not None:
            time_to_first_token.labels(model_name=model_name).observe(first_token_time)
        
        # Context window utilization
        total_tokens = prompt_tokens + completion_tokens
        if context_window > 0:
            utilization = total_tokens / context_window
            context_utilization_ratio.labels(model_name=model_name).observe(utilization)
        
        # Check if response was truncated
        if max_tokens and actual_tokens and actual_tokens >= max_tokens:
            response_truncated_total.labels(model_name=model_name).inc()

    def record_api_key_request(self, api_key_prefix: str, model_name: str, status: str, api_key_name: str = None):
        """Record API key request"""
        api_key_requests_total.labels(
            api_key_prefix=api_key_prefix,
            api_key_name=api_key_name or "unknown",
            model_name=model_name,
            status=status
        ).inc()
        
        # Update last used timestamp
        api_key_last_used_timestamp.labels(
            api_key_prefix=api_key_prefix,
            api_key_name=api_key_name or "unknown"
        ).set(time.time())
    
    def record_api_key_endpoint_usage(self, api_key_prefix: str, api_key_name: str, endpoint: str, method: str):
        """Record API key usage by endpoint"""
        api_key_usage_by_endpoint.labels(
            api_key_prefix=api_key_prefix,
            api_key_name=api_key_name or "unknown",
            endpoint=endpoint,
            method=method
        ).inc()
    
    def record_api_key_rate_limit_hit(self, api_key_prefix: str, api_key_name: str = None):
        """Record when an API key hits rate limits"""
        api_key_rate_limit_hits.labels(
            api_key_prefix=api_key_prefix,
            api_key_name=api_key_name or "unknown"
        ).inc()

    def record_error(self, error_type: str, model_name: str = "unknown", endpoint: str = "unknown"):
        """Record error with categorization"""
        errors_total.labels(error_type=error_type, model_name=model_name, endpoint=endpoint).inc()

    def record_model_loaded(self, model_name: str):
        """Record when a model is loaded"""
        self._model_load_times[model_name] = time.time()
        
        # Track model switch if applicable
        if self._last_active_model and self._last_active_model != model_name:
            if self._model_switch_start:
                switch_duration = time.time() - self._model_switch_start
                model_switch_duration.labels(
                    from_model=self._last_active_model,
                    to_model=model_name
                ).observe(switch_duration)
        
        self._last_active_model = model_name
        self._model_switch_start = time.time()
    
    def record_model_warmup(self, model_name: str):
        """Update model warmup time"""
        if model_name in self._model_load_times:
            warmup_time = time.time() - self._model_load_times[model_name]
            model_warmup_time.labels(model_name=model_name).set(warmup_time)
    
    def record_sampling_params(self, model_name: str, temperature: float, top_p: float):
        """Record sampling parameters used"""
        temperature_distribution.labels(model_name=model_name).observe(temperature)
        top_p_distribution.labels(model_name=model_name).observe(top_p)
    
    def record_request_size(self, endpoint: str, request_bytes: int, response_bytes: int):
        """Record request and response sizes"""
        request_size_bytes.labels(endpoint=endpoint).observe(request_bytes)
        response_size_bytes.labels(endpoint=endpoint).observe(response_bytes)
    
    def record_streaming_metrics(self, model_name: str, chunk_latency: float = None, status: str = "success"):
        """Record streaming-related metrics"""
        streaming_requests_total.labels(model_name=model_name, status=status).inc()
        if chunk_latency is not None:
            streaming_chunk_latency.labels(model_name=model_name).observe(chunk_latency)
    
    def record_queue_time(self, model_name: str, queue_time: float):
        """Record time spent in queue (deprecated)"""
        request_queue_time.labels(model_name=model_name).observe(queue_time)
    
    def record_api_response_time(self, model_name: str, response_time: float):
        """Record API response time and update min/max metrics"""
        if model_name not in self._response_times:
            self._response_times[model_name] = {"min": response_time, "max": response_time}
        else:
            self._response_times[model_name]["min"] = min(self._response_times[model_name]["min"], response_time)
            self._response_times[model_name]["max"] = max(self._response_times[model_name]["max"], response_time)
        
        # Update Prometheus metrics
        api_response_time_min.labels(model_name=model_name).set(self._response_times[model_name]["min"])
        api_response_time_max.labels(model_name=model_name).set(self._response_times[model_name]["max"])
    
    def record_rejected_request(self, reason: str):
        """Record rejected request"""
        rejected_requests_total.labels(reason=reason).inc()
    
    def record_cache_operation(self, operation: str):
        """Record cache operation (hit, miss, eviction, load)"""
        cache_operations_total.labels(operation=operation).inc()
        
        # Update cache size metric in real-time
        if operation in ['load', 'eviction']:
            from app.services.model_manager import model_manager
            cache_info = model_manager.get_cache_info()
            model_cache_size.set(cache_info['cache_size'])
            
            # Update model warmup times
            for model_name in cache_info.get('cached_models', []):
                self.record_model_warmup(model_name)
    
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
        # Always update memory metrics when Prometheus scrapes
        self.update_memory_metrics(force=True)
        
        # Update model cache size
        from app.services.model_manager import model_manager
        cache_info = model_manager.get_cache_info()
        model_cache_size.set(cache_info['cache_size'])
        
        return generate_latest().decode('utf-8')


# Singleton instance
metrics_collector = MetricsCollector()