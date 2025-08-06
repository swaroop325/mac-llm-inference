from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time
import uuid
from typing import Callable
import asyncio

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import ErrorResponse
from app.core.database import db_manager


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to logger context
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {request.url.path}")
            return JSONResponse(
                status_code=504,
                content=ErrorResponse(
                    error={
                        "message": "Request timeout",
                        "type": "timeout_error",
                        "code": "request_timeout"
                    }
                ).dict()
            )
        except Exception as e:
            logger.exception(f"Unhandled exception: {str(e)}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error={
                        "message": "Internal server error",
                        "type": "internal_error",
                        "code": "internal_error"
                    }
                ).dict()
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API key authentication."""
    
    def __init__(self, app, protected_paths: list = None):
        super().__init__(app)
        self.protected_paths = protected_paths or ["/v1/chat/completions"]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health/monitoring endpoints and docs  
        if request.url.path in ["/health", "/ready", "/metrics", "/metrics/prometheus", 
                               "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)
        
        # Skip auth for static files
        if request.url.path.startswith("/static"):
            return await call_next(request)
        
        # Skip auth for key management endpoints (as per your requirement)
        # Note: This means anyone can manage API keys - consider security implications
        if request.url.path.startswith("/auth"):
            return await call_next(request)
        
        # Check if this is a protected endpoint (chat completions and other API endpoints)
        is_protected = any(request.url.path.startswith(path) for path in self.protected_paths)
        
        if not is_protected:
            return await call_next(request)
        
        # Check for API key in headers
        api_key = None
        
        # Try Authorization header first (Bearer token)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        # Try X-API-Key header as fallback
        if not api_key:
            api_key = request.headers.get("x-api-key")
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error={
                        "message": "API key required",
                        "type": "authentication_error",
                        "code": "missing_api_key"
                    }
                ).dict()
            )
        
        # Verify the API key
        key_info = db_manager.verify_api_key(api_key)
        if not key_info:
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error={
                        "message": "Invalid or expired API key",
                        "type": "authentication_error",
                        "code": "invalid_api_key"
                    }
                ).dict()
            )
        
        # Add key info to request state for use in endpoints
        request.state.api_key_info = key_info
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Log API usage (async, don't block response)
        try:
            db_manager.log_api_usage(
                api_key_id=key_info['id'],
                endpoint=request.url.path,
                method=request.method,
                response_status=response.status_code,
                processing_time_ms=processing_time
            )
        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")
        
        return response


def setup_cors(app) -> None:
    settings = get_settings()
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )