#!/bin/bash
# Test OpenAI API compatibility

PORT=${PORT:-8123}
API_KEY=${API_KEY:-"test-key"}

echo "Testing OpenAI API compatibility on port $PORT..."
echo ""

# Test 1: Basic chat completion
echo "1. Testing basic chat completion..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }' | jq .

echo ""
echo "2. Testing with minimal parameters..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "messages": [
      {"role": "user", "content": "Hi"}
    ]
  }' | jq .

echo ""
echo "3. Testing error handling (invalid model)..."
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invalid-model",
    "messages": [
      {"role": "user", "content": "Test"}
    ]
  }' | jq .

echo ""
echo "4. Comparing response format with OpenAI..."
echo "Expected OpenAI response structure:"
cat << 'EOF' | jq .
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
EOF