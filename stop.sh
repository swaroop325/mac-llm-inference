#!/bin/bash
# Stop script for LLM Inference Server - reverses run.sh/setup.sh

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default ports if not in environment
PORT=${PORT:-7000}
APP_PORT=${APP_PORT:-${PORT}}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
GRAFANA_PORT=${GRAFANA_PORT:-3000}

print_info "Stopping LLM Inference Server and monitoring services..."

# Function to stop a service by port
stop_service_by_port() {
    local port=$1
    local service_name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pids=$(lsof -t -i:$port)
        if [ -n "$pids" ]; then
            print_info "Stopping $service_name on port $port (PIDs: $pids)..."
            kill $pids 2>/dev/null || true
            
            # Wait for process to stop (max 5 seconds)
            local count=0
            while [ $count -lt 5 ] && lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                print_warn "Force killing $service_name..."
                kill -9 $(lsof -t -i:$port) 2>/dev/null || true
            fi
            
            print_info "$service_name stopped ✓"
        fi
    else
        print_info "$service_name not running on port $port"
    fi
}

# Function to stop a service by process name
stop_service_by_name() {
    local process_name=$1
    local service_name=$2
    
    local pids=$(pgrep -f "$process_name" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        print_info "Stopping $service_name (PIDs: $pids)..."
        kill $pids 2>/dev/null || true
        
        # Wait for process to stop (max 5 seconds)
        local count=0
        while [ $count -lt 5 ] && pgrep -f "$process_name" >/dev/null 2>&1; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if pgrep -f "$process_name" >/dev/null 2>&1; then
            print_warn "Force killing $service_name..."
            pkill -9 -f "$process_name" 2>/dev/null || true
        fi
        
        print_info "$service_name stopped ✓"
    else
        print_info "$service_name not running"
    fi
}

# Step 1: Stop the main application server (uvicorn)
print_info "Checking for application server..."
stop_service_by_port $APP_PORT "LLM Inference Server"
# Also check for any uvicorn processes
stop_service_by_name "uvicorn app.main:app" "Uvicorn Server"

# Step 2: Stop Grafana
print_info "Checking for Grafana..."
stop_service_by_port $GRAFANA_PORT "Grafana"
# Also stop grafana-server process if running
stop_service_by_name "grafana server" "Grafana Server"
# Stop any brew services
brew services stop grafana 2>/dev/null || true

# Step 3: Stop Prometheus
print_info "Checking for Prometheus..."
stop_service_by_port $PROMETHEUS_PORT "Prometheus"
# Also stop by process name
stop_service_by_name "prometheus --config.file" "Prometheus"

# Step 4: Clean up any orphaned Python processes from the virtual environment
if [ -d "venv" ]; then
    print_info "Checking for orphaned Python processes..."
    venv_python_pids=$(pgrep -f "$(pwd)/venv/bin/python" 2>/dev/null || true)
    if [ -n "$venv_python_pids" ]; then
        print_warn "Found orphaned Python processes from venv: $venv_python_pids"
        kill $venv_python_pids 2>/dev/null || true
        sleep 2
        # Force kill if still running
        venv_python_pids=$(pgrep -f "$(pwd)/venv/bin/python" 2>/dev/null || true)
        if [ -n "$venv_python_pids" ]; then
            kill -9 $venv_python_pids 2>/dev/null || true
        fi
        print_info "Orphaned processes cleaned up ✓"
    fi
fi

# Step 5: Check for any remaining processes on monitored ports
print_info "Final port check..."
declare -a ports=($APP_PORT $PROMETHEUS_PORT $GRAFANA_PORT)
declare -a port_names=("App Server" "Prometheus" "Grafana")

for i in "${!ports[@]}"; do
    if lsof -Pi :${ports[$i]} -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warn "Port ${ports[$i]} (${port_names[$i]}) is still in use!"
        print_info "You may need to manually kill: kill \$(lsof -t -i:${ports[$i]})"
    fi
done

# Step 6: Optional cleanup of logs (commented out by default)
# Uncomment if you want to clean logs when stopping
# if [ -d "logs" ]; then
#     print_info "Cleaning up log files..."
#     rm -f logs/*.log
#     print_info "Logs cleaned ✓"
# fi

print_info "================================================"
print_info "All services stopped successfully! ✓"
print_info "================================================"
print_info ""
print_info "To restart the services, run:"
print_info "  ./setup.sh          # Full setup with monitoring"
print_info "  ./run.sh            # Quick restart"
print_info ""
print_info "To check if any services are still running:"
print_info "  lsof -i :$APP_PORT     # Check app server"
print_info "  lsof -i :$PROMETHEUS_PORT     # Check Prometheus"
print_info "  lsof -i :$GRAFANA_PORT      # Check Grafana"