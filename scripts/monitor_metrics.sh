#!/bin/bash
# Real-time metrics monitoring for Activate LLM Inference Server

PORT=${PORT:-8000}
REFRESH_INTERVAL=${REFRESH_INTERVAL:-5}

echo "Activate LLM Inference Server Metrics Monitor"
echo "Server: http://localhost:$PORT"
echo "Refresh: ${REFRESH_INTERVAL}s"
echo "Press Ctrl+C to exit"
echo "======================================"

while true; do
    clear
    echo "🚀 Activate LLM Inference Server Metrics - $(date)"
    echo "======================================"
    
    # Check if server is running
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Server Status: HEALTHY"
        echo ""
        
        # Get metrics in JSON format
        METRICS=$(curl -s http://localhost:$PORT/metrics 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$METRICS" ]; then
            echo "📊 Performance Metrics:"
            echo "$METRICS" | jq -r '
                "  • Total Requests: \(.requests_total // 0)",
                "  • Failed Requests: \(.requests_failed // 0)",
                "  • Avg Latency: \(.average_latency_ms // 0 | floor)ms",
                "  • Memory Usage: \(.memory_usage_mb // 0 | floor)MB",
                "  • Model Cache Size: \(.model_cache_size // 0)"
            ' 2>/dev/null || echo "  ⚠️  Could not parse metrics"
        else
            echo "  ⚠️  Could not retrieve metrics"
        fi
        
        echo ""
        echo "🧠 Model Information:"
        MODELS=$(curl -s http://localhost:$PORT/models 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$MODELS" ]; then
            echo "$MODELS" | jq -r '
                "  • Cached Models: \(.cached_models | length)",
                "  • Cache Usage: \(.cache_size)/\(.max_cache_size)",
                "  • Memory %: \(.memory_usage.percent // 0 | floor)%"
            ' 2>/dev/null || echo "  ⚠️  Could not parse model info"
        else
            echo "  ⚠️  Could not retrieve model information"
        fi
        
    else
        echo "❌ Server Status: DOWN"
        echo ""
        echo "Server appears to be offline on port $PORT"
        echo "Try starting it with: ./setup.sh"
    fi
    
    echo ""
    echo "======================================"
    echo "Next refresh in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)"
    
    sleep $REFRESH_INTERVAL
done