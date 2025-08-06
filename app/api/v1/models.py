from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
import time
import asyncio
from typing import Dict, Any

from app.models.schemas import (
    ModelPreloadRequest,
    ModelPreloadResponse,
    ModelCacheResponse,
    ErrorResponse
)
from app.services.model_manager import model_manager
from app.utils.metrics import model_load_duration, model_cache_size
from app.core.logging import logger
from app.core.config import get_settings


router = APIRouter()
settings = get_settings()


@router.post(
    "/preload",
    response_model=ModelPreloadResponse,
    summary="Preload model into cache",
    description="Preload a model into cache for faster inference. Model will be downloaded and loaded into memory.",
    responses={
        200: {"description": "Model preloaded successfully or already cached"},
        400: {"description": "Invalid model identifier"},
        413: {"description": "Model too large or cache full"},
        500: {"description": "Model loading failed"},
    },
    tags=["Model Management"]
)
async def preload_model(
    request: ModelPreloadRequest,
    background_tasks: BackgroundTasks,
    req: Request = None
):
    """
    Preload a model into the cache for faster inference.
    
    This endpoint allows you to load models into memory before making inference requests,
    reducing the latency of the first request to that model.
    """
    request_id = req.state.request_id if req else "preload"
    
    logger.info(
        f"Model preload request",
        extra={
            "request_id": request_id,
            "model": request.model
        }
    )
    
    try:
        # Check if already cached
        cache_info = model_manager.get_cache_info()
        if request.model in cache_info["cached_models"]:
            logger.info(f"Model {request.model} already cached")
            return ModelPreloadResponse(
                model=request.model,
                status="already_cached",
                message=f"Model {request.model} is already cached",
                cache_info=cache_info
            )
        
        # Check cache capacity
        if len(cache_info["cached_models"]) >= cache_info["max_cache_size"]:
            memory_info = cache_info["memory_usage"]
            if memory_info["available_gb"] < 2.0:
                raise HTTPException(
                    status_code=413,
                    detail="Cache is full and insufficient memory available"
                )
        
        # Load model with timeout
        start_time = time.time()
        
        try:
            with model_load_duration.labels(model_name=request.model).time():
                model, tokenizer = await asyncio.wait_for(
                    model_manager.get_model(request.model),
                    timeout=settings.model_load_timeout_seconds
                )
            
            load_time = time.time() - start_time
            
            # Update cache size metric
            updated_cache_info = model_manager.get_cache_info()
            model_cache_size.set(updated_cache_info["cache_size"])
            
            logger.info(
                f"Model preload successful",
                extra={
                    "request_id": request_id,
                    "model": request.model,
                    "load_time": load_time
                }
            )
            
            return ModelPreloadResponse(
                model=request.model,
                status="success",
                load_time_seconds=load_time,
                message=f"Model {request.model} loaded successfully",
                cache_info=updated_cache_info
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Model loading timeout for {request.model}")
            raise HTTPException(
                status_code=504,
                detail=f"Model loading timeout for {request.model}"
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load model {request.model}: {error_msg}")
            
            if "No such file or directory" in error_msg or "not found" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model} not found or invalid identifier"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model loading failed: {error_msg}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Model preload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/cache",
    response_model=ModelCacheResponse,
    summary="Get model cache information",
    description="Get information about currently cached models and cache statistics.",
    tags=["Model Management"]
)
async def get_model_cache():
    """
    Get information about the model cache.
    
    Returns details about which models are currently cached, cache size,
    load times, and memory usage statistics.
    """
    try:
        cache_info = model_manager.get_cache_info()
        
        return ModelCacheResponse(
            cached_models=cache_info["cached_models"],
            cache_size=cache_info["cache_size"],
            max_cache_size=cache_info["max_cache_size"],
            load_times=cache_info["load_times"],
            memory_usage=cache_info["memory_usage"]
        )
        
    except Exception as e:
        logger.exception(f"Failed to get cache info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache information: {str(e)}"
        )


@router.delete(
    "/cache",
    summary="Clear model cache",
    description="Clear all models from the cache to free up memory.",
    responses={
        200: {"description": "Cache cleared successfully"},
        500: {"description": "Failed to clear cache"},
    },
    tags=["Model Management"]
)
async def clear_model_cache():
    """
    Clear all models from the cache.
    
    This will remove all loaded models from memory, freeing up system resources.
    Subsequent inference requests will need to reload models.
    """
    try:
        model_manager.clear_cache()
        model_cache_size.set(0)
        
        logger.info("Model cache cleared")
        
        return {
            "status": "success",
            "message": "Model cache cleared successfully"
        }
        
    except Exception as e:
        logger.exception(f"Failed to clear cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.delete(
    "/cache/{model_name:path}",
    summary="Remove specific model from cache",
    description="Remove a specific model from the cache to free up memory.",
    responses={
        200: {"description": "Model removed from cache"},
        404: {"description": "Model not found in cache"},
        500: {"description": "Failed to remove model from cache"},
    },
    tags=["Model Management"]
)
async def remove_model_from_cache(model_name: str):
    """
    Remove a specific model from the cache.
    
    This will remove the specified model from memory while keeping other models cached.
    """
    try:
        cache_info = model_manager.get_cache_info()
        
        if model_name not in cache_info["cached_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found in cache"
            )
        
        # Remove from cache (we need to add this method to ModelManager)
        with model_manager._cache_lock:
            if model_name in model_manager._models_cache:
                del model_manager._models_cache[model_name]
                del model_manager._last_access[model_name]
                if model_name in model_manager._load_times:
                    del model_manager._load_times[model_name]
                logger.info(f"Removed model {model_name} from cache")
        
        # Update cache size metric
        updated_cache_info = model_manager.get_cache_info()
        model_cache_size.set(updated_cache_info["cache_size"])
        
        return {
            "status": "success",
            "message": f"Model {model_name} removed from cache",
            "cache_info": updated_cache_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to remove model {model_name} from cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove model from cache: {str(e)}"
        )