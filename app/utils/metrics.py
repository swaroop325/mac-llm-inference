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


class MetricsCollector:
    def __init__(self):
        self.settings = get_settings()
        self._start_time = time.time()
        self._request_count = 0
        self._failed_requests = 0
        self._total_latency = 0.0
        
    def update_memory_metrics(self):
        """Update memory usage metrics"""
        memory = psutil.virtual_memory()
        memory_usage_bytes.labels(type='total').set(memory.total)
        memory_usage_bytes.labels(type='used').set(memory.used)
        memory_usage_bytes.labels(type='available').set(memory.available)
        
        # MLX GPU memory if available
        try:
            # This is a placeholder - MLX doesn't directly expose GPU memory
            # In production, you might use system-specific tools
            pass
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