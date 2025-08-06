"""Authentication and API key management endpoints."""

from fastapi import APIRouter, HTTPException, Depends, status, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.core.database import db_manager
from app.core.logging import logger

router = APIRouter()
security = HTTPBearer()

# Pydantic models for request/response
class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Unique name for the API key")
    expires_days: Optional[int] = Field(None, ge=1, le=365, description="Number of days until expiration")
    rate_limit: int = Field(1000, ge=1, le=10000, description="Requests per hour limit")
    metadata: Optional[str] = Field(None, max_length=500, description="Additional metadata")

class APIKeyResponse(BaseModel):
    id: int
    name: str
    prefix: str
    is_active: bool
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    usage_count: int
    rate_limit: int
    created_by: str
    metadata: Optional[str] = None

class CreateAPIKeyResponse(APIKeyResponse):
    api_key: str = Field(..., description="The actual API key (only shown once)")

class UsageStatsResponse(BaseModel):
    id: int
    key_name: str
    key_prefix: str
    request_count: int
    avg_processing_time: Optional[float]
    last_request: Optional[str]

class APIKeyInfo(BaseModel):
    """Information about the authenticated API key."""
    id: int
    name: str
    prefix: str
    usage_count: int
    rate_limit: int

def verify_admin_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify admin API key for management endpoints."""
    api_key = credentials.credentials
    key_info = db_manager.verify_api_key(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )
    
    # For now, all valid keys have admin access
    # In production, you might want to add role-based access
    return key_info

def get_current_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Get current API key information for regular endpoints."""
    api_key = credentials.credentials
    key_info = db_manager.verify_api_key(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )
    
    return key_info

@router.post(
    "/keys",
    response_model=CreateAPIKeyResponse,
    summary="Create API key",
    description="Create a new API key. The key is only shown once - save it securely.",
    tags=["API Key Management"]
)
async def create_api_key(request: CreateAPIKeyRequest):
    """Create a new API key."""
    try:
        key_data = db_manager.create_api_key(
            name=request.name,
            expires_days=request.expires_days,
            rate_limit=request.rate_limit,
            metadata=request.metadata,
            created_by="unauth"
        )
        
        logger.info(f"API key '{request.name}' created without authentication")
        return CreateAPIKeyResponse(**key_data)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get(
    "/keys",
    response_model=List[APIKeyResponse],
    summary="List API keys",
    description="Get all API keys with their details and usage statistics.",
    tags=["API Key Management"]
)
async def list_api_keys(include_inactive: bool = False):
    """List all API keys."""
    keys = db_manager.list_api_keys(include_inactive=include_inactive)
    return [APIKeyResponse(**key) for key in keys]

@router.get(
    "/keys/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API key",
    description="Get details for a specific API key.",
    tags=["API Key Management"]
)
async def get_api_key(key_id: int):
    """Get API key details by ID."""
    key_data = db_manager.get_api_key(key_id)
    
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    
    return APIKeyResponse(**key_data)

@router.patch(
    "/keys/{key_id}/deactivate",
    summary="Deactivate API key",
    description="Deactivate an API key (can be reactivated later). No authentication required.",
    tags=["API Key Management"]
)
async def deactivate_api_key(key_id: int):
    """Deactivate an API key."""
    success = db_manager.deactivate_api_key(key_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    
    logger.info(f"API key ID {key_id} deactivated without authentication")
    return {"message": f"API key {key_id} deactivated successfully"}

@router.delete(
    "/keys/{key_id}",
    summary="Delete API key",
    description="Permanently delete an API key and its usage history.",
    tags=["API Key Management"]
)
async def delete_api_key(key_id: int):
    """Permanently delete an API key."""
    success = db_manager.delete_api_key(key_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    
    logger.info(f"API key ID {key_id} deleted without authentication")
    return {"message": f"API key {key_id} deleted successfully"}

@router.get(
    "/keys/usage/stats",
    response_model=List[UsageStatsResponse],
    summary="Usage statistics",
    description="Get usage statistics for all API keys.",
    tags=["API Key Management"]
)
async def get_usage_stats(days: int = Query(7, ge=1, le=90, description="Number of days to analyze")):
    """Get usage statistics for API keys."""
    stats = db_manager.get_usage_stats(days=days)
    return [UsageStatsResponse(**stat) for stat in stats]

@router.get(
    "/keys/{key_id}/usage/stats",
    response_model=List[UsageStatsResponse],
    summary="Key usage statistics",
    description="Get usage statistics for a specific API key.",
    tags=["API Key Management"]
)
async def get_key_usage_stats(
    key_id: int,
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze")
):
    """Get usage statistics for a specific API key."""
    stats = db_manager.get_usage_stats(key_id=key_id, days=days)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    
    return [UsageStatsResponse(**stat) for stat in stats]

@router.get(
    "/me",
    response_model=APIKeyInfo,
    summary="Get current API key info",
    description="Get information about the currently authenticated API key",
    tags=["Authentication"]
)
async def get_current_key_info(
    current_key: Dict[str, Any] = Depends(get_current_api_key)
):
    """Get information about the current API key."""
    return APIKeyInfo(
        id=current_key['id'],
        name=current_key['key_name'],
        prefix=current_key['key_prefix'],
        usage_count=current_key['usage_count'],
        rate_limit=current_key['rate_limit']
    )

