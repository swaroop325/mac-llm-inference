from fastapi import APIRouter, Response
import mlx.core as mx
import psutil
from datetime import datetime

from app.models.schemas import HealthResponse, MetricsResponse
from app.services.model_manager import model_manager
from app.utils.metrics import metrics_collector
from app.core.config import get_settings
from app.core.logging import logger


router = APIRouter()
settings = get_settings()


@router.get(
    "/health", 
    response_model=HealthResponse,
    summary="Health check",
    description="""
    Check the health status of the Activate inference server.
    
    Returns information about:
    - Server status (healthy/unhealthy)
    - GPU availability and type
    - Model cache status
    - System resources
    - Service version
    """
)
async def health_check():
    """
    Perform a comprehensive health check of the server.
    
    This endpoint checks GPU access, model loading capability,
    and system resources to determine overall server health.
    """
    try:
        # Check if Activate is available and GPU is accessible
        gpu_available = mx.default_device().type.name == "gpu"
        
        # Get model cache info
        cache_info = model_manager.get_cache_info()
        model_loaded = len(cache_info["cached_models"]) > 0
        
        # Get system info
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        details = {
            "gpu_device": mx.default_device().type.name,
            "cached_models": cache_info["cached_models"],
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
            "uptime_seconds": psutil.boot_time()
        }
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            details=details
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            model_loaded=False,
            gpu_available=False,
            details={"error": str(e)}
        )


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # Check if at least one model is loaded or can be loaded
    cache_info = model_manager.get_cache_info()
    
    if len(cache_info["cached_models"]) > 0:
        return {"status": "ready", "models": cache_info["cached_models"]}
    
    # If no models loaded, system is still ready but cold
    return {"status": "ready", "models": [], "note": "No models loaded, first request will be slower"}


@router.get(
    "/metrics", 
    response_model=MetricsResponse,
    summary="Get metrics (JSON)",
    description="Get application metrics in JSON format for monitoring dashboards"
)
async def get_metrics():
    """
    Retrieve application metrics in JSON format.
    
    Returns performance metrics, request statistics,
    and system resource usage information.
    """
    metrics_summary = metrics_collector.get_metrics_summary()
    cache_info = model_manager.get_cache_info()
    
    return MetricsResponse(
        requests_total=metrics_summary["requests_total"],
        requests_failed=metrics_summary["requests_failed"],
        average_latency_ms=metrics_summary["average_latency_ms"],
        model_cache_size=len(cache_info["cached_models"]),
        memory_usage_mb=metrics_summary["memory_usage_mb"],
        gpu_memory_usage_mb=None  # Activate doesn't expose this directly
    )


@router.get(
    "/metrics/prometheus", 
    response_class=Response,
    summary="Get metrics (Prometheus)",
    description="""
    Get metrics in Prometheus exposition format.
    
    This endpoint returns metrics in the standard Prometheus format
    that can be scraped by Prometheus servers for monitoring and alerting.
    """
)
async def get_prometheus_metrics():
    """
    Retrieve metrics in Prometheus exposition format.
    
    Returns all application metrics in a format suitable
    for Prometheus scraping and monitoring.
    """
    if not settings.enable_metrics:
        return Response(
            content="Metrics disabled",
            status_code=404
        )
    
    metrics_data = metrics_collector.get_prometheus_metrics()
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4"
    )


@router.get("/models")
async def list_models():
    """List available and cached models"""
    cache_info = model_manager.get_cache_info()
    
    return {
        "cached_models": cache_info["cached_models"],
        "cache_size": cache_info["cache_size"],
        "max_cache_size": cache_info["max_cache_size"],
        "memory_usage": cache_info["memory_usage"],
        "default_model": settings.model_path
    }


@router.post("/models/clear-cache")
async def clear_model_cache():
    """Clear the model cache (admin endpoint)"""
    model_manager.clear_cache()
    return {"status": "success", "message": "Model cache cleared"}