from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import mlx.core as mx

from app.core.config import get_settings
from app.core.logging import logger
from app.api.v1 import chat, health
from app.utils.middleware import (
    RequestIdMiddleware, 
    ErrorHandlingMiddleware, 
    setup_cors
)
from app.utils.metrics import metrics_collector, active_requests
from app.models.schemas import ErrorResponse


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"MLX device: {mx.default_device().type.name}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API authentication: {'enabled' if settings.api_keys else 'disabled'}")
    
    # Preload default model if specified
    if settings.model_path and settings.debug:
        try:
            from app.services.model_manager import model_manager
            logger.info(f"Preloading default model: {settings.model_path}")
            await model_manager.get_model(settings.model_path)
        except Exception as e:
            logger.warning(f"Failed to preload default model: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    from app.services.model_manager import model_manager
    model_manager.clear_cache()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade MLX inference server with OpenAI-compatible API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Setup middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
setup_cors(app)

# Include routers
app.include_router(
    chat.router,
    prefix="/v1",
    tags=["chat"]
)

app.include_router(
    health.router,
    prefix="",
    tags=["health"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs" if settings.debug else None
        }
    }


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error={
                "message": f"Path {request.url.path} not found",
                "type": "not_found",
                "code": "resource_not_found"
            }
        ).dict()
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    """Handle 405 errors"""
    return JSONResponse(
        status_code=405,
        content=ErrorResponse(
            error={
                "message": f"Method {request.method} not allowed for {request.url.path}",
                "type": "method_not_allowed",
                "code": "method_not_allowed"
            }
        ).dict()
    )


# Middleware for tracking active requests
@app.middleware("http")
async def track_requests(request: Request, call_next):
    active_requests.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests.dec()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True
    )