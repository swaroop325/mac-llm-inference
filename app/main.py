from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import mlx.core as mx
import time

from app.core.config import get_settings
from app.core.logging import logger
from app.api.v1 import chat, health, auth, models
from app.utils.middleware import (
    RequestIdMiddleware, 
    ErrorHandlingMiddleware,
    AuthenticationMiddleware,
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
    logger.info(f"Activate device: {mx.default_device().type.name}")
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
    description="""
    # Activate LLM Inference Server
    
    Enterprise-grade LLM inference server with OpenAI-compatible API endpoints.
    
    ## Quick Start
    1. Create an API key using `POST /auth/keys`
    2. Click the **🔒 Authorize** button above to enter your API key
    3. Use the chat completion endpoints
    
    ## Available Models
    - `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
    
    ## Monitoring
    - Grafana: http://localhost:3000 (admin/admin)
    - Prometheus: http://localhost:9090
    """,
    contact={
        "name": "Activate LLM Inference Server",
        "url": "https://github.com/ml-explore/mlx",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc", 
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure security for Swagger UI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme for Bearer token
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "Enter your API key (starts with mlx_)"
        }
    }
    
    # Apply security to protected endpoints
    protected_paths = ["/v1/chat/completions", "/v1/models", "/auth/me"]
    for path, methods in openapi_schema["paths"].items():
        # Check if this path needs authentication
        needs_auth = any(path.startswith(protected) for protected in protected_paths)
        if needs_auth:
            for method, operation in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    operation["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Setup middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(AuthenticationMiddleware)
setup_cors(app)

# Include routers
app.include_router(
    chat.router,
    prefix="/v1",
    tags=["Chat Completions"]
)

app.include_router(
    health.router,
    prefix="",
    tags=["Health & Monitoring"]
)

app.include_router(
    auth.router,
    prefix="/auth",
    tags=["API Key Management"]
)

app.include_router(
    models.router,
    prefix="/v1/models",
    tags=["Model Management"]
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
            "models": "/v1/models",
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


# Middleware for tracking active requests and HTTP metrics
@app.middleware("http")
async def track_requests(request: Request, call_next):
    from app.utils.metrics import request_count, request_duration
    
    active_requests.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        
        # Record HTTP metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code)
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)
        
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