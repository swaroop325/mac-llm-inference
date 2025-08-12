#!/bin/bash

# Universal Inference Server Comprehensive Load Test
# Tests all metrics including API key usage, model switching, and various scenarios
# Supports MLX, vLLM, and CPU backends with appropriate model selection

set -e

# Configuration
# Read configuration from .env file if available
if [[ -f .env ]]; then
    PORT=$(grep "^PORT=" .env | cut -d'=' -f2)
    PORT=${PORT:-7000}  # Default to 7000 if not found
    BACKEND=$(grep "^INFERENCE_BACKEND=" .env | cut -d'=' -f2)
    BACKEND=${BACKEND:-auto}  # Default to auto if not found
else
    PORT=7000
    BACKEND=auto
fi

BASE_URL="http://localhost:${PORT}"
API_KEY="llm_XQSpsKs_ctT1dPzHYa5bCHPLpC9okrB40FbCuoWfpR8"
CONCURRENT=3        # Concurrent requests per batch
TOTAL=30           # Total requests
DELAY=2            # 2 seconds between batches

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Multiple models to test model switching - dynamically set based on backend
if [[ "$BACKEND" == "mlx" ]]; then
    MODELS=(
        "mlx-community/Llama-3.2-1B-Instruct-bf16"
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        "mlx-community/Llama-3.2-1B-Instruct-bf16"  # Switch back to create switching metrics
    )
    echo -e "${BLUE}Using MLX-optimized models${NC}"
elif [[ "$BACKEND" == "vllm" ]]; then
    MODELS=(
        "microsoft/Phi-3-mini-4k-instruct"
        "meta-llama/Llama-3.2-1B-Instruct"  
        "microsoft/Phi-3-mini-4k-instruct"  # Switch back to create switching metrics
    )
    echo -e "${BLUE}Using vLLM-compatible models${NC}"
else
    # CPU or auto backend
    MODELS=(
        "microsoft/Phi-3-mini-4k-instruct"
        "meta-llama/Llama-3.2-1B-Instruct"
        "microsoft/Phi-3-mini-4k-instruct"  # Switch back to create switching metrics
    )
    echo -e "${BLUE}Using CPU-compatible models (backend: $BACKEND)${NC}"
fi

# Diverse test scenarios to populate all metrics
TEST_SCENARIOS=(
    '{"prompt":"Hello world","temp":0.1,"tokens":5,"model_idx":0}'
    '{"prompt":"What is artificial intelligence?","temp":0.3,"tokens":15,"model_idx":1}'
    '{"prompt":"Write a short poem about code","temp":0.7,"tokens":25,"model_idx":0}'
    '{"prompt":"Explain machine learning briefly","temp":0.5,"tokens":20,"model_idx":1}'
    '{"prompt":"Tell me a programming joke","temp":0.8,"tokens":18,"model_idx":2}'
    '{"prompt":"How does neural network work?","temp":0.4,"tokens":30,"model_idx":0}'
    '{"prompt":"Describe the future of AI","temp":0.9,"tokens":35,"model_idx":1}'
    '{"prompt":"What is deep learning?","temp":0.6,"tokens":12,"model_idx":2}'
    '{"prompt":"Code a hello world in Python","temp":0.2,"tokens":22,"model_idx":0}'
    '{"prompt":"Summarize quantum computing","temp":0.7,"tokens":28,"model_idx":1}'
    '{"prompt":"Write haiku about programming","temp":1.0,"tokens":16,"model_idx":2}'
    '{"prompt":"Explain API design patterns","temp":0.4,"tokens":40,"model_idx":0}'
)

# Function to create additional API keys for testing
create_test_api_keys() {
    echo -e "${BLUE}Creating additional API keys for testing...${NC}"
    
    # Create load-test-user-1 key
    local key1_response=$(curl -s -X POST "$BASE_URL/auth/keys" \
        -H "Content-Type: application/json" \
        -d '{"name": "load-test-user-1", "description": "Load test user 1"}' 2>/dev/null)
    
    if echo "$key1_response" | grep -q "api_key"; then
        LOAD_TEST_KEY_1=$(echo "$key1_response" | jq -r '.api_key')
        echo -e "${GREEN}Created API key 1: load-test-user-1${NC}"
    else
        LOAD_TEST_KEY_1="$API_KEY"
        echo -e "${YELLOW}Using default API key for user 1${NC}"
    fi
    
    # Create load-test-user-2 key  
    local key2_response=$(curl -s -X POST "$BASE_URL/auth/keys" \
        -H "Content-Type: application/json" \
        -d '{"name": "load-test-user-2", "description": "Load test user 2"}' 2>/dev/null)
    
    if echo "$key2_response" | grep -q "api_key"; then
        LOAD_TEST_KEY_2=$(echo "$key2_response" | jq -r '.api_key')
        echo -e "${GREEN}Created API key 2: load-test-user-2${NC}"
    else
        LOAD_TEST_KEY_2="$API_KEY"
        echo -e "${YELLOW}Using default API key for user 2${NC}"
    fi
}

# Function to send a single API request
send_request() {
    local request_id=$1
    local scenario_data=$2
    local api_key=$3
    local key_name=$4
    
    # Parse scenario JSON
    local prompt=$(echo "$scenario_data" | jq -r '.prompt')
    local temp=$(echo "$scenario_data" | jq -r '.temp')
    local tokens=$(echo "$scenario_data" | jq -r '.tokens')
    local model_idx=$(echo "$scenario_data" | jq -r '.model_idx')
    local model_name=${MODELS[$model_idx]}
    
    # Create request payload
    local payload=$(cat <<EOF
{
  "model": "$model_name",
  "messages": [{"role": "user", "content": "$prompt"}],
  "temperature": $temp,
  "max_tokens": $tokens
}
EOF
)
    
    echo -e "${BLUE}[$request_id-$key_name]${NC} Model: $(basename $model_name), Temp: $temp, Tokens: $tokens"
    echo -e "${PURPLE}[$request_id-$key_name]${NC} Prompt: $prompt"
    
    local start_time=$(date +%s.%N)
    local temp_file=$(mktemp)
    
    # Send request with longer timeout for model loading
    local http_code=$(timeout 120s curl -s -w "%{http_code}" -o "$temp_file" \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $api_key" \
        -d "$payload" 2>/dev/null || echo "TIMEOUT")
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
    
    if [[ $http_code == "200" ]]; then
        local response_body=$(cat "$temp_file")
        local content=$(echo "$response_body" | jq -r '.choices[0].message.content // "No content"' 2>/dev/null || echo "Parse error")
        local usage=$(echo "$response_body" | jq -r '.usage // {}' 2>/dev/null)
        
        # Truncate content for display
        local short_content=$(echo "$content" | head -c 100)
        [[ ${#content} -gt 100 ]] && short_content="${short_content}..."
        
        echo -e "${GREEN}[$request_id-$key_name]${NC} Success (${duration}s): $short_content"
        echo -e "${GREEN}[$request_id-$key_name]${NC} Usage: $usage"
    elif [[ $http_code == "TIMEOUT" ]]; then
        echo -e "${RED}[$request_id-$key_name]${NC} TIMEOUT after 120s - Model loading or server overloaded..."
    else
        local response_body=$(cat "$temp_file" 2>/dev/null || echo "No response")
        local error_msg=$(echo "$response_body" | jq -r '.error.message // "Unknown error"' 2>/dev/null || echo "Parse error")
        echo -e "${RED}[$request_id-$key_name]${NC} Failed HTTP $http_code (${duration}s): $error_msg"
    fi
    
    rm -f "$temp_file"
}

# Function to run concurrent batch with different API keys
run_concurrent_batch() {
    local batch_num=$1
    local batch_size=$2
    
    echo ""
    echo -e "${BLUE}=== Starting Batch $batch_num (${batch_size} requests) ===${NC}"
    
    # Start requests in parallel with different API keys
    local pids=()
    for ((i=1; i<=batch_size; i++)); do
        local scenario_index=$(( (batch_num * batch_size + i - 1) % ${#TEST_SCENARIOS[@]} ))
        local scenario=${TEST_SCENARIOS[$scenario_index]}
        local request_id="B${batch_num}R${i}"
        
        # Rotate API keys for testing
        local api_key
        local key_name
        case $((i % 3)) in
            0) api_key="$LOAD_TEST_KEY_1"; key_name="user1" ;;
            1) api_key="$LOAD_TEST_KEY_2"; key_name="user2" ;;
            2) api_key="$API_KEY"; key_name="main" ;;
        esac
        
        send_request "$request_id" "$scenario" "$api_key" "$key_name" &
        pids+=($!)
        
        # Small delay between starting requests to avoid overwhelming
        sleep 0.2
    done
    
    # Wait for all requests in this batch to complete
    echo -e "${YELLOW}Waiting for batch $batch_num to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo -e "${GREEN}✓ Batch $batch_num completed${NC}"
}

# Function to check system status (simplified)
check_system_status() {
    echo -e "${BLUE}Checking inference server process...${NC}"
    
    # Check if server process is running
    if pgrep -f uvicorn >/dev/null; then
        echo -e "${GREEN}✓ Server process is running on port 7000${NC}"
        
        # Try a simple connectivity test with timeout
        echo -e "${BLUE}Testing server connectivity...${NC}"
        if timeout 5s curl -s "$BASE_URL/docs" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is responding${NC}"
        else
            echo -e "${YELLOW}⚠ Server may be starting up or busy - proceeding with load test${NC}"
        fi
    else
        echo -e "${RED}❌ Error: No uvicorn process found${NC}"
        echo -e "${YELLOW}Start the server with: uvicorn app.main:app --host 0.0.0.0 --port 7000${NC}"
        exit 1
    fi
}

# Function to show metrics summary
show_metrics_summary() {
    echo ""
    echo -e "${BLUE}=== FETCHING METRICS SUMMARY ===${NC}"
    
    # Get key metrics with error handling
    local metrics_data=$(curl -s --connect-timeout 10 "$BASE_URL/metrics/prometheus" 2>/dev/null)
    
    if [[ -z "$metrics_data" ]]; then
        echo -e "${RED}Could not fetch metrics data${NC}"
        return
    fi
    
    echo "$metrics_data" > /tmp/metrics_output.txt
    
    # Parse metrics safely
    local api_requests=$(echo "$metrics_data" | grep '^http_requests_total.*chat/completions.*200' | awk '{sum += $2} END {print sum+0}')
    local failed_requests=$(echo "$metrics_data" | grep '^http_requests_total.*chat/completions.*[45][0-9][0-9]' | awk '{sum += $2} END {print sum+0}')
    local active_requests=$(echo "$metrics_data" | grep '^active_requests ' | awk '{print $2}' | head -1)
    local memory_used=$(echo "$metrics_data" | grep '^memory_usage_bytes.*used' | awk '{print $2}' | head -1)
    local api_key_usage=$(echo "$metrics_data" | grep '^api_key_requests_total' | wc -l)
    local token_usage=$(echo "$metrics_data" | grep '^api_key_token_usage_total' | awk '{sum += $2} END {print sum+0}')
    
    # Parse API response time metrics
    local min_response_time=$(echo "$metrics_data" | grep '^api_response_time_min_seconds' | awk '{print $2}' | sort -n | head -1)
    local max_response_time=$(echo "$metrics_data" | grep '^api_response_time_max_seconds' | awk '{print $2}' | sort -n | tail -1)
    
    echo -e "${GREEN}=== FINAL METRICS SUMMARY ===${NC}"
    echo -e "✅ API Requests (200): ${api_requests:-0}"
    echo -e "❌ Failed Requests (4xx/5xx): ${failed_requests:-0}"
    echo -e "🔄 Active Requests: ${active_requests:-0}"
    echo -e "💾 Memory Used: $(echo "scale=2; ${memory_used:-0} / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0") GB"
    echo -e "🔑 API Key Metrics Count: ${api_key_usage:-0}"
    echo -e "🎯 Total Tokens Used: ${token_usage:-0}"
    echo -e "⚡ Min Response Time: ${min_response_time:-N/A}s"
    echo -e "🐌 Max Response Time: ${max_response_time:-N/A}s"
    
    # Show API key breakdown
    echo ""
    echo -e "${BLUE}API Key Usage Breakdown:${NC}"
    echo "$metrics_data" | grep '^api_key_requests_total' | head -5 | while read line; do
        echo -e "${PURPLE}$line${NC}"
    done
    
    rm -f /tmp/metrics_output.txt
}

# Main execution function
main() {
    echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Universal Comprehensive Load Test  ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "🎯 Target: $BASE_URL"
    echo -e "🧠 Backend: $BACKEND"
    echo -e "⚡ Concurrent: $CONCURRENT requests"
    echo -e "📊 Total: $TOTAL requests"
    echo -e "⏱️ Delay: ${DELAY}s between batches"
    echo -e "🔄 Models: ${#MODELS[@]} different models (for switching metrics)"
    echo -e "🔑 API Keys: 3 different keys (for key usage metrics)"
    echo ""
    
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ Error: jq is required but not installed${NC}"
        echo "Install with: brew install jq"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        echo -e "${RED}❌ Error: bc is required but not installed${NC}"
        echo "Install with: brew install bc"
        exit 1
    fi
    
    # System checks
    check_system_status
    
    # Create test API keys
    create_test_api_keys
    
    echo ""
    echo -e "${BLUE}Starting comprehensive load test...${NC}"
    echo -e "${YELLOW}⚠️ First requests may take 1-2 minutes while models download${NC}"
    
    # Record start time
    local test_start=$(date +%s)
    
    # Calculate batches
    local batches=$((TOTAL / CONCURRENT))
    local remaining=$((TOTAL % CONCURRENT))
    
    echo ""
    echo -e "${BLUE}📈 Test Plan: $batches full batches + $remaining remaining requests${NC}"
    
    # Run full batches
    for ((batch=1; batch<=batches; batch++)); do
        run_concurrent_batch $batch $CONCURRENT
        
        # Delay between batches
        if [[ $batch -lt $batches || $remaining -gt 0 ]]; then
            echo -e "${YELLOW}⏳ Waiting ${DELAY}s before next batch...${NC}"
            sleep $DELAY
        fi
    done
    
    # Run remaining requests
    if [[ $remaining -gt 0 ]]; then
        run_concurrent_batch $((batches + 1)) $remaining
    fi
    
    # Calculate total time
    local test_end=$(date +%s)
    local total_time=$((test_end - test_start))
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            LOAD TEST COMPLETED           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
    echo -e "📊 Total Requests Sent: $TOTAL"
    echo -e "⏱️ Total Duration: ${total_time}s"
    echo -e "⚡ Average RPS: $(echo "scale=2; $TOTAL / $total_time" | bc 2>/dev/null || echo "N/A")"
    
    # Show metrics summary
    show_metrics_summary
    
    echo ""
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo -e "1. Check your Grafana dashboard: ${YELLOW}http://localhost:3000${NC}"
    echo -e "2. Look for populated metrics in all sections"
    echo -e "3. API Key Usage Analytics should now show data"
    echo -e "4. Model Switch Duration may show data if models switched"
    echo ""
    echo -e "${GREEN}✨ All metrics should now be populated with real data!${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        -t|--total)
            TOTAL="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -h|--help)
            echo "LLM Comprehensive Load Test"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --concurrent NUM    Number of concurrent requests (default: 3)"
            echo "  -t, --total NUM         Total number of requests (default: 12)"
            echo "  -d, --delay SECONDS     Delay between batches (default: 2)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "This script will:"
            echo "• Test multiple models to populate model switching metrics"
            echo "• Use different API keys to populate key usage metrics"
            echo "• Vary temperatures, token counts, and prompts"
            echo "• Populate ALL dashboard metrics with real data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Run main function
main