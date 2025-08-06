from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import time
import uuid
import asyncio
from typing import AsyncGenerator

from app.models.schemas import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Message,
    Usage,
    ErrorResponse
)
from app.services.model_manager import model_manager
from app.utils.security import verify_api_key
from app.utils.metrics import metrics_collector, inference_duration
from app.core.logging import logger
from app.core.config import get_settings


router = APIRouter()
settings = get_settings()


def extract_prompt(messages: list[Message]) -> str:
    """Extract prompt from messages list"""
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "\n".join(prompt_parts)


def count_tokens(text: str) -> int:
    """Simple token counting - in production use proper tokenizer"""
    # Rough estimate: 1 token ~= 4 characters
    return len(text) // 4


@router.post(
    "/chat/completions", 
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="Create chat completion using OpenAI-compatible API with Activate models.",
    responses={
        200: {"description": "Chat completion generated successfully"},
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
    tags=["Chat Completions"]
)
async def chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """
    Generate a chat completion response using Activate models.
    
    Compatible with OpenAI's chat completion API format.
    """
    request_id = req.state.request_id if req else str(uuid.uuid4())
    
    logger.info(
        f"Chat completion request",
        extra={
            "request_id": request_id,
            "model": request.model,
            "message_count": len(request.messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
    )
    
    try:
        # Extract prompt from messages
        prompt = extract_prompt(request.messages)
        prompt_tokens = count_tokens(prompt)
        
        # Generate response with timeout
        start_time = time.time()
        
        response_text = await asyncio.wait_for(
            model_manager.generate_response(
                model_name=request.model,
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            ),
            timeout=settings.timeout_seconds
        )
        
        inference_time = time.time() - start_time
        completion_tokens = count_tokens(response_text)
        
        # Record metrics
        with inference_duration.labels(model_name=request.model).time():
            pass  # Metric already recorded
        
        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop" if len(response_text) < request.max_tokens else "length"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        logger.info(
            f"Chat completion successful",
            extra={
                "request_id": request_id,
                "inference_time": inference_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        )
        
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Inference timeout for request {request_id}")
        raise HTTPException(
            status_code=504,
            detail="Inference timeout"
        )
    except Exception as e:
        logger.exception(f"Chat completion error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/chat/completions/stream",
    summary="Stream chat completion", 
    description="Stream chat completion responses (placeholder implementation)",
    tags=["Chat Completions"]
)
async def chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """Streaming chat completion endpoint (placeholder)"""
    # Note: Activate doesn't natively support streaming yet
    # This is a placeholder for future implementation
    
    if not request.stream:
        # If not streaming, redirect to regular endpoint
        return await chat_completion(request, BackgroundTasks(), api_key, req)
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        # Placeholder streaming implementation
        response = await chat_completion(request, BackgroundTasks(), api_key, req)
        
        # Convert to SSE format
        import json
        data = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": 0,
                "delta": {"content": response.choices[0].message.content},
                "finish_reason": response.choices[0].finish_reason
            }]
        }
        
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )