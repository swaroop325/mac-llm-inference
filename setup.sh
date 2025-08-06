#!/bin/bash
# Unified setup and run script for MLX Inference Server

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

# Parse command line arguments
MODE="dev"
INSTALL_DEPS="yes"
ENABLE_PROMETHEUS="yes"
ENABLE_GRAFANA="yes"
for arg in "$@"; do
    case $arg in
        --production|--prod)
            MODE="production"
            shift
            ;;
        --no-install)
            INSTALL_DEPS="no"
            shift
            ;;
        --no-prometheus)
            ENABLE_PROMETHEUS="no"
            ENABLE_GRAFANA="no"  # No Grafana without Prometheus
            shift
            ;;
        --no-grafana)
            ENABLE_GRAFANA="no"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --production, --prod    Run in production mode"
            echo "  --no-install           Skip dependency installation"
            echo "  --no-prometheus        Disable Prometheus & Grafana monitoring"
            echo "  --no-grafana           Disable Grafana (keep Prometheus only)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  PORT                   Server port (default: 8000)"
            echo "  HOST                   Server host (default: 0.0.0.0)"
            echo "  LOG_LEVEL             Log level (default: INFO)"
            echo ""
            echo "Examples:"
            echo "  $0                     Run with MLX + Prometheus + Grafana"
            echo "  $0 --production        Production mode with full monitoring"
            echo "  $0 --no-grafana        Run with MLX + Prometheus only"
            echo "  $0 --no-prometheus     Run MLX server only (no monitoring)"
            echo "  PORT=8001 $0          Run on custom port 8001"
            exit 0
            ;;
    esac
done

print_info "Starting MLX Inference Server setup (Mode: $MODE)"

# Step 1: Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
print_info "Python version: $PYTHON_VERSION ✓"

# Step 2: Virtual environment
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
else
    print_info "Virtual environment already exists ✓"
fi

# Step 3: Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Step 4: Install/upgrade dependencies
if [ "$INSTALL_DEPS" = "yes" ]; then
    print_info "Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    print_info "Dependencies installed ✓"
else
    print_info "Skipping dependency installation (--no-install flag)"
fi

# Step 5: Create necessary directories
print_info "Creating required directories..."
mkdir -p logs data/prometheus
print_info "Directories created ✓"

# Step 6: Load environment variables if .env exists
if [ -f .env ]; then
    print_info "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
else
    print_warn "No .env file found. Using default configuration."
    print_warn "Copy config/.env.example to .env for custom configuration."
fi

# Step 7: Set MLX environment
export MLX_FORCE_METAL=1
export PYTHONUNBUFFERED=1

# Step 8: Verify MLX installation
print_info "Verifying MLX installation..."
if python -c "import mlx.core as mx; print(f'MLX Device: {mx.default_device().type.name}')" 2>/dev/null; then
    print_info "MLX verified and GPU accessible ✓"
else
    print_error "MLX verification failed!"
    exit 1
fi

# Step 9: Check port availability
PORT=${PORT:-8000}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_error "Port $PORT is already in use!"
    print_info "Options:"
    print_info "  1. Kill existing process: kill \$(lsof -t -i:$PORT)"
    print_info "  2. Use different port: PORT=8001 $0"
    exit 1
fi

# Step 10: Check and install monitoring tools if needed
if [ "$ENABLE_PROMETHEUS" = "yes" ]; then
    print_info "Checking Prometheus installation..."
    
    if ! command -v prometheus &> /dev/null; then
        print_warn "Prometheus not found. Installing..."
        if command -v brew &> /dev/null; then
            brew install prometheus
        else
            print_error "Homebrew not found. Please install Prometheus manually:"
            print_error "  brew install prometheus"
            print_error "  Or download from: https://prometheus.io/download/"
            exit 1
        fi
    fi
    
    print_info "Prometheus found ✓"
fi

if [ "$ENABLE_GRAFANA" = "yes" ]; then
    print_info "Checking Grafana installation..."
    
    if ! command -v grafana-server &> /dev/null; then
        print_warn "Grafana not found. Installing..."
        if command -v brew &> /dev/null; then
            brew install grafana
        else
            print_error "Homebrew not found. Please install Grafana manually:"
            print_error "  brew install grafana"
            print_error "  Or download from: https://grafana.com/grafana/download"
            exit 1
        fi
    fi
    
    print_info "Grafana found ✓"
fi

# Step 11: Start monitoring services
PROMETHEUS_PID=""
GRAFANA_PID=""
cleanup() {
    print_info "Shutting down services..."
    if [ -n "$PROMETHEUS_PID" ] && kill -0 "$PROMETHEUS_PID" 2>/dev/null; then
        print_info "Stopping Prometheus..."
        kill "$PROMETHEUS_PID"
        wait "$PROMETHEUS_PID" 2>/dev/null
    fi
    if [ -n "$GRAFANA_PID" ] && [ "$GRAFANA_PID" != "service" ] && kill -0 "$GRAFANA_PID" 2>/dev/null; then
        print_info "Stopping Grafana..."
        kill "$GRAFANA_PID"
        wait "$GRAFANA_PID" 2>/dev/null
    fi
    # Also stop any system Grafana service
    brew services stop grafana >/dev/null 2>&1 || true
    exit 0
}

trap cleanup SIGINT SIGTERM

if [ "$ENABLE_PROMETHEUS" = "yes" ]; then
    print_info "Starting Prometheus on port 9090..."
    prometheus --config.file=config/prometheus.yml --web.listen-address=:9090 --storage.tsdb.path=./data/prometheus > logs/prometheus.log 2>&1 &
    PROMETHEUS_PID=$!
    print_info "Prometheus started with PID $PROMETHEUS_PID ✓"
    sleep 2  # Give Prometheus time to start
fi

if [ "$ENABLE_GRAFANA" = "yes" ]; then
    print_info "Starting Grafana on port 3000..."
    
    # Stop any existing Grafana service
    brew services stop grafana 2>/dev/null || true
    sleep 1
    
    # Create Grafana directories
    mkdir -p data/grafana/{data,logs,plugins,provisioning/datasources,provisioning/dashboards}
    
    # Copy provisioning files
    cp -r config/grafana-provisioning/* data/grafana/provisioning/ 2>/dev/null || true
    
    # Copy dashboard to provisioning directory
    mkdir -p /tmp/grafana-dashboards
    cp config/grafana-dashboard.json /tmp/grafana-dashboards/mlx-dashboard.json
    
    # Create custom Grafana configuration
    cat > data/grafana/grafana.ini << 'EOF'
[paths]
data = ./data/grafana/data
logs = ./data/grafana/logs
plugins = ./data/grafana/plugins
provisioning = ./data/grafana/provisioning

[server]
http_port = 3000

[security]
admin_user = admin
admin_password = admin

[users]
allow_sign_up = false
auto_assign_org_role = Admin

[auth.anonymous]
enabled = false
EOF
    
    # Start Grafana with custom config
    grafana server \
        --homepath $(brew --prefix grafana)/share/grafana \
        --config ./data/grafana/grafana.ini \
        > logs/grafana.log 2>&1 &
    GRAFANA_PID=$!
    print_info "Grafana started with PID $GRAFANA_PID ✓"
    sleep 5  # Give Grafana more time to start and load provisioning
fi

# Step 12: Start main server
print_info "Starting MLX Inference Server on port $PORT..."
echo ""

if [ "$ENABLE_PROMETHEUS" = "yes" ] || [ "$ENABLE_GRAFANA" = "yes" ]; then
    print_info "🚀 Services will be available at:"
    print_info "  📡 MLX API Server:  http://localhost:$PORT"
    
    if [ "$ENABLE_PROMETHEUS" = "yes" ]; then
        print_info "  📊 Prometheus:      http://localhost:9090"
        print_info "  📈 Raw Metrics:     http://localhost:$PORT/metrics/prometheus"
    fi
    
    if [ "$ENABLE_GRAFANA" = "yes" ]; then
        print_info "  📋 Grafana:         http://localhost:3000 (admin/admin)"
    fi
    
    print_info "  ❤️  Health Check:    http://localhost:$PORT/health"
    echo ""
    
    if [ "$ENABLE_GRAFANA" = "yes" ]; then
        print_info "💡 Grafana Setup:"
        print_info "  ✅ Prometheus data source automatically configured"
        print_info "  ✅ MLX Dashboard automatically imported and ready"
        print_info "  🔑 Login: admin/admin (change password when prompted)"
        print_info "  📊 Dashboard: MLX Inference Server Dashboard"
        echo ""
    fi
fi

if [ "$MODE" = "production" ]; then
    # Production mode
    uvicorn app.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --workers "${WORKERS:-1}" \
        --loop asyncio \
        --log-level "${LOG_LEVEL:-info}" \
        --access-log \
        --use-colors \
        --timeout-keep-alive 65 \
        --limit-concurrency 1000 \
        --limit-max-requests 10000 \
        --log-config config/logging_config.yaml
else
    # Development mode
    uvicorn app.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --reload \
        --log-level "${LOG_LEVEL:-info}" \
        --access-log \
        --use-colors \
        --log-config config/logging_config.yaml
fi