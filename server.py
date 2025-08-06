# server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List, Dict
import mlx.core as mx
from mlx_lm import load, generate

app = FastAPI()

def extract_prompt(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

def build_response(model: str, content: str):
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model_name = body.get("model")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 256)

    prompt = extract_prompt(messages)
    model, tokenizer = load(model_name)
    output = generate(model, tokenizer, prompt, temp=temperature, max_tokens=max_tokens)

    return JSONResponse(build_response(model_name, output))
