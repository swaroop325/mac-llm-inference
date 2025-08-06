from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    
    @validator("content")
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        ...,
        description="ID of the model to use",
        example="mlx-community/Llama-3.2-1B-Instruct-bf16"
    )
    messages: List[Message] = Field(
        ...,
        description="List of messages comprising the conversation so far",
        example=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ]
    )
    temperature: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Controls randomness: 0 is deterministic, 2 is very random"
    )
    max_tokens: Optional[int] = Field(
        default=256, 
        ge=1, 
        le=4096,
        description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0, 
        ge=-2.0, 
        le=2.0,
        description="Penalty for frequent tokens"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0, 
        ge=-2.0, 
        le=2.0,
        description="Penalty for tokens that have appeared"
    )
    stop: Optional[List[str]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating tokens"
    )
    stream: Optional[bool] = Field(
        False,
        description="Whether to stream back partial progress"
    )
    
    @validator("messages")
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v
    
    @validator("max_tokens")
    def validate_max_tokens(cls, v, values):
        from app.core.config import get_settings
        settings = get_settings()
        if v > settings.max_allowed_tokens:
            raise ValueError(f"max_tokens cannot exceed {settings.max_allowed_tokens}")
        return v


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]
    

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ErrorResponse(BaseModel):
    error: Dict[str, Any]
    
    
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: Literal["healthy", "unhealthy"]
    version: str
    model_loaded: bool
    gpu_available: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    requests_total: int
    requests_failed: int
    average_latency_ms: float
    model_cache_size: int
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float] = None


class ModelPreloadRequest(BaseModel):
    model: str = Field(
        ...,
        description="Model identifier to preload into cache",
        example="mlx-community/Llama-3.2-1B-Instruct-bf16"
    )


class ModelPreloadResponse(BaseModel):
    model: str
    status: Literal["success", "error", "already_cached"]
    load_time_seconds: Optional[float] = None
    message: str
    cache_info: Optional[Dict[str, Any]] = None


class ModelCacheResponse(BaseModel):
    cached_models: List[str]
    cache_size: int
    max_cache_size: int
    load_times: Dict[str, float]
    memory_usage: Dict[str, float]