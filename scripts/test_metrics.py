#!/usr/bin/env python3
"""Test script to generate metrics data for Activate inference server."""

import requests
import time
import json

def test_endpoints():
    """Test various endpoints to generate metrics data."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Activate LLM Inference Server Endpoints...")
    
    # Test 1: Health check (should always work)
    print("\n1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   ✅ Health: {response.status_code} - {response.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    # Test 2: Metrics endpoint
    print("\n2️⃣  Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics/prometheus", timeout=5)
        metrics_data = response.text
        lines = metrics_data.count('\n')
        print(f"   ✅ Metrics: {response.status_code} - {lines} lines of metrics data")
        
        # Show some sample metrics
        for line in metrics_data.split('\n')[:10]:
            if line and not line.startswith('#'):
                print(f"   📊 {line}")
                
    except Exception as e:
        print(f"   ❌ Metrics failed: {e}")
    
    # Test 3: Create an API key
    print("\n3️⃣  Creating test API key...")
    try:
        key_payload = {
            "name": "test-metrics-key",
            "rate_limit": 100,
            "metadata": "For metrics testing"
        }
        response = requests.post(
            f"{base_url}/auth/keys", 
            json=key_payload, 
            timeout=10
        )
        if response.status_code == 200:
            key_data = response.json()
            api_key = key_data['api_key']
            print(f"   ✅ API Key created: {key_data['name']} (ID: {key_data['id']})")
            print(f"   🔑 Key: {api_key[:20]}...")
            
            # Test 4: Make authenticated requests
            print("\n4️⃣  Testing chat completion with API key...")
            chat_payload = {
                "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "messages": [
                    {"role": "user", "content": "Hello! This is a test message for metrics collection."}
                ],
                "max_tokens": 50
            }
            
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Make multiple requests to generate metrics
            for i in range(3):
                print(f"   📝 Request {i+1}/3...")
                try:
                    response = requests.post(
                        f"{base_url}/v1/chat/completions",
                        json=chat_payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content'][:50]
                        print(f"   ✅ Chat: {response.status_code} - '{content}...'")
                    else:
                        print(f"   ⚠️  Chat: {response.status_code} - {response.text[:100]}")
                        
                    time.sleep(1)  # Brief pause between requests
                    
                except Exception as e:
                    print(f"   ❌ Chat request {i+1} failed: {e}")
            
            # Test 5: Check metrics again
            print("\n5️⃣  Checking metrics after API calls...")
            try:
                response = requests.get(f"{base_url}/metrics/prometheus", timeout=5)
                metrics_data = response.text
                
                # Look for specific metrics
                request_metrics = [line for line in metrics_data.split('\n') 
                                 if 'http_requests_total' in line and not line.startswith('#')]
                
                print(f"   ✅ Found {len(request_metrics)} request metrics:")
                for metric in request_metrics[:5]:
                    print(f"   📈 {metric}")
                    
            except Exception as e:
                print(f"   ❌ Final metrics check failed: {e}")
                
        else:
            print(f"   ❌ API Key creation failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ❌ API key test failed: {e}")
    
    print("\n✨ Metrics testing complete!")
    print("\n🔗 You can now check:")
    print("   • Prometheus: http://localhost:9090")
    print("   • Grafana: http://localhost:3000")
    print("   • Metrics: http://localhost:8000/metrics/prometheus")

if __name__ == "__main__":
    test_endpoints()