#!/usr/bin/env python3
"""
Example script to test the Activate LLM Inference Server API
"""

import requests
import json

def test_mlx_api():
    base_url = "http://localhost:8000"
    
    # Test health check
    print("🔍 Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    # Test chat completion
    print("💬 Testing chat completion...")
    payload = {
        "model": "mlx-community/Llama-3.2-1B-Instruct-bf16",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Usage: {result.get('usage', {})}")
    else:
        print(f"Error: {response.text}")
    print()
    
    # Test metrics
    print("📊 Testing metrics...")
    response = requests.get(f"{base_url}/metrics")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        metrics = response.json()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print()

if __name__ == "__main__":
    test_mlx_api()