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
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    
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
    status: Literal["healthy", "unhealthy"]
    version: str
    model_loaded: bool
    gpu_available: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    requests_total: int
    requests_failed: int
    average_latency_ms: float
    model_cache_size: int
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float] = None