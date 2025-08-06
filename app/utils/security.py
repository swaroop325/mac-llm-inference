from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from app.core.config import get_settings
from app.core.logging import logger


settings = get_settings()
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Verify API key if authentication is enabled"""
    # If no API keys configured, allow all requests
    if not settings.api_keys:
        return "anonymous"
    
    # If API keys are configured but none provided
    if not api_key:
        logger.warning("No API key provided for protected endpoint")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Verify API key
    if api_key not in settings.api_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key