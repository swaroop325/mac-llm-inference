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
from app.utils.metrics import (
    metrics_collector, 
    inference_duration,
    temperature_distribution,
    top_p_distribution,
    request_size_bytes,
    response_size_bytes
)
from app.core.logging import logger
from app.core.config import get_settings


router = APIRouter()
settings = get_settings()


def extract_prompt(messages: list[Message]) -> str:
    """Extract prompt from messages list"""
    prompt_parts = []
    
    # Handle system message if present
    system_msg = None
    conversation = []
    
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
        else:
            conversation.append(msg)
    
    # Build prompt based on model format
    if system_msg:
        prompt_parts.append(f"<|system|>\n{system_msg}")
    
    # Add conversation history
    for msg in conversation:
        if msg.role == "user":
            prompt_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{msg.content}")
    
    # Add assistant prompt for completion
    prompt_parts.append("<|assistant|>")
    
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
    req: Request = None
):
    """
    Generate a chat completion response using Activate models.
    
    Compatible with OpenAI's chat completion API format.
    """
    request_id = req.state.request_id if req else str(uuid.uuid4())
    
    # Get API key info for metrics
    api_key_prefix = "unknown"
    api_key_name = "unknown"
    
    if req and hasattr(req.state, 'api_key_info') and req.state.api_key_info:
        api_key_info = req.state.api_key_info
        api_key_prefix = api_key_info.get('key_prefix', 'unknown')
        api_key_name = api_key_info.get('key_name', 'unknown')
    elif req:
        # Fallback: get from header for prefix only
        api_key = req.headers.get("X-API-Key") or req.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key and len(api_key) > 8:
            api_key_prefix = api_key[:8] + "..."
    
    logger.info(
        f"Chat completion request",
        extra={
            "request_id": request_id,
            "model": request.model,
            "message_count": len(request.messages),
            "max_tokens_raw": request.max_tokens,
            "max_tokens_effective": request.max_tokens or settings.default_max_tokens,
            "default_max_tokens": settings.default_max_tokens,
            "temperature": request.temperature
        }
    )
    
    try:
        # Record request size
        import json
        request_bytes = len(json.dumps(request.dict()))
        
        # Extract prompt from messages
        prompt = extract_prompt(request.messages)
        prompt_tokens = count_tokens(prompt)
        
        # Record sampling parameters
        metrics_collector.record_sampling_params(
            model_name=request.model,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Record inference start
        metrics_collector.record_inference_start()
        
        # Generate response with timeout
        start_time = time.time()
        first_token_start = time.time()
        
        # Use default max_tokens from settings if not provided
        max_tokens = request.max_tokens or settings.default_max_tokens
        
        response_text = await asyncio.wait_for(
            model_manager.generate_response(
                model_name=request.model,
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=max_tokens,
                top_p=request.top_p
            ),
            timeout=settings.timeout_seconds
        )
        
        inference_time = time.time() - start_time
        completion_tokens = count_tokens(response_text)
        
        # Record inference end
        metrics_collector.record_inference_end()
        
        # Record API response time for min/max tracking
        metrics_collector.record_api_response_time(request.model, inference_time)
        
        # Check if response was truncated
        actual_tokens = count_tokens(response_text)
        
        # Record token metrics with context window info
        first_token_time = min(0.1, inference_time)  # Estimate first token time
        metrics_collector.record_token_metrics(
            model_name=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time=inference_time,
            first_token_time=first_token_time,
            api_key_prefix=api_key_prefix,
            api_key_name=api_key_name,
            max_tokens=max_tokens,
            actual_tokens=actual_tokens,
            context_window=4096  # Default context window, could be model-specific
        )
        
        # Record API key request success
        metrics_collector.record_api_key_request(api_key_prefix, request.model, "success", api_key_name)
        
        # Record metrics
        with inference_duration.labels(model_name=request.model).time():
            pass  # Metric already recorded
        
        # Record response size
        response_json = {
            "id": f"chatcmpl-{request_id}",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop" if len(response_text) < max_tokens else "length"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        response_bytes = len(json.dumps(response_json))
        
        metrics_collector.record_request_size(
            endpoint="/v1/chat/completions",
            request_bytes=request_bytes,
            response_bytes=response_bytes
        )
        
        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop" if len(response_text) < max_tokens else "length"
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
        metrics_collector.record_inference_end()  # Ensure we clean up inference counter
        metrics_collector.record_error("timeout", request.model, "chat_completion")
        metrics_collector.record_api_key_request(api_key_prefix, request.model, "timeout", api_key_name)
        logger.error(f"Inference timeout for request {request_id}")
        raise HTTPException(
            status_code=504,
            detail="Inference timeout"
        )
    except Exception as e:
        metrics_collector.record_inference_end()  # Ensure we clean up inference counter
        
        # Categorize error types
        if "404" in str(e) or "Repository Not Found" in str(e):
            error_type = "model_not_found"
        elif "Memory" in str(e) or "OOM" in str(e):
            error_type = "out_of_memory"
        elif "CUDA" in str(e) or "GPU" in str(e):
            error_type = "gpu_error"
        else:
            error_type = "internal_error"
            
        metrics_collector.record_error(error_type, request.model, "chat_completion")
        metrics_collector.record_api_key_request(api_key_prefix, request.model, "error", api_key_name)
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
    req: Request = None
):
    """Streaming chat completion endpoint (placeholder)"""
    # Note: Activate doesn't natively support streaming yet
    # This is a placeholder for future implementation
    
    if not request.stream:
        # If not streaming, redirect to regular endpoint
        return await chat_completion(request, BackgroundTasks(), req)
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        # Placeholder streaming implementation
        response = await chat_completion(request, BackgroundTasks(), req)
        
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