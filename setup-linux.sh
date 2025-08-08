#!/bin/bash
# Unified setup and run script for LLM Inference Server

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

# Function to install Prometheus from binary
install_prometheus_binary() {
    print_info "Installing Prometheus from binary..."
    PROMETHEUS_VERSION="2.45.0"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64) PROMETHEUS_ARCH="amd64" ;;
        aarch64|arm64) PROMETHEUS_ARCH="arm64" ;;
        *) print_error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    
    cd /tmp
    wget "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${PROMETHEUS_ARCH}.tar.gz"
    tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-${PROMETHEUS_ARCH}.tar.gz"
    sudo mv "prometheus-${PROMETHEUS_VERSION}.linux-${PROMETHEUS_ARCH}/prometheus" /usr/local/bin/
    sudo mv "prometheus-${PROMETHEUS_VERSION}.linux-${PROMETHEUS_ARCH}/promtool" /usr/local/bin/
    rm -rf "prometheus-${PROMETHEUS_VERSION}.linux-${PROMETHEUS_ARCH}"*
    cd - > /dev/null
    print_info "Prometheus installed from binary ✓"
}

# Function to install Grafana from binary
install_grafana_binary() {
    print_info "Installing Grafana from binary..."
    GRAFANA_VERSION="10.0.0"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64) GRAFANA_ARCH="amd64" ;;
        aarch64|arm64) GRAFANA_ARCH="arm64" ;;
        *) print_error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    
    cd /tmp
    wget "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${GRAFANA_ARCH}.tar.gz"
    tar xzf "grafana-${GRAFANA_VERSION}.linux-${GRAFANA_ARCH}.tar.gz"
    sudo mv "grafana-${GRAFANA_VERSION}" /opt/grafana
    sudo ln -sf /opt/grafana/bin/grafana-server /usr/local/bin/grafana-server
    sudo ln -sf /opt/grafana/bin/grafana-cli /usr/local/bin/grafana-cli
    rm -rf "grafana-${GRAFANA_VERSION}.linux-${GRAFANA_ARCH}.tar.gz"
    cd - > /dev/null
    print_info "Grafana installed from binary ✓"
}

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
            echo "  $0                     Run with Activate + Prometheus + Grafana"
            echo "  $0 --production        Production mode with full monitoring"
            echo "  $0 --no-grafana        Run with Activate + Prometheus only"
            echo "  $0 --no-prometheus     Run Activate server only (no monitoring)"
            echo "  PORT=8001 $0          Run on custom port 8001"
            exit 0
            ;;
    esac
done

print_info "Starting LLM Inference Server setup (Mode: $MODE)"

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

# Step 7: Set Activate environment
export MLX_FORCE_METAL=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Step 8: Verify Activate installation
print_info "Verifying Activate installation..."
if python -c "import mlx.core as mx; print(f'Activate Device: {mx.default_device().type.name}')" 2>/dev/null; then
    print_info "Activate verified and GPU accessible ✓"
else
    print_error "Activate verification failed!"
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
        
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        else
            OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        fi
        
        case "$OS" in
            ubuntu|debian)
                print_info "Installing Prometheus on Ubuntu/Debian..."
                sudo apt-get update
                sudo apt-get install -y prometheus
                ;;
            fedora|centos|rhel)
                print_info "Installing Prometheus on RHEL/Fedora/CentOS..."
                if command -v dnf &> /dev/null; then
                    sudo dnf install -y prometheus2
                elif command -v yum &> /dev/null; then
                    sudo yum install -y prometheus2
                else
                    print_error "Package manager not found. Installing from binary..."
                    install_prometheus_binary
                fi
                ;;
            arch)
                print_info "Installing Prometheus on Arch Linux..."
                sudo pacman -S --noconfirm prometheus
                ;;
            *)
                print_warn "Unsupported Linux distribution: $OS. Installing from binary..."
                install_prometheus_binary
                ;;
        esac
    fi
    
    print_info "Prometheus found ✓"
fi

if [ "$ENABLE_GRAFANA" = "yes" ]; then
    print_info "Checking Grafana installation..."
    
    if ! command -v grafana-server &> /dev/null; then
        print_warn "Grafana not found. Installing..."
        
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        else
            OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        fi
        
        case "$OS" in
            ubuntu|debian)
                print_info "Installing Grafana on Ubuntu/Debian..."
                sudo apt-get update
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main" -y
                wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
                sudo apt-get update
                sudo apt-get install -y grafana
                ;;
            fedora|centos|rhel)
                print_info "Installing Grafana on RHEL/Fedora/CentOS..."
                sudo tee /etc/yum.repos.d/grafana.repo > /dev/null << 'EOF'
[grafana]
name=grafana
baseurl=https://packages.grafana.com/oss/rpm
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://packages.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
EOF
                if command -v dnf &> /dev/null; then
                    sudo dnf install -y grafana
                else
                    sudo yum install -y grafana
                fi
                ;;
            arch)
                print_info "Installing Grafana on Arch Linux..."
                sudo pacman -S --noconfirm grafana
                ;;
            *)
                print_warn "Unsupported Linux distribution: $OS. Installing from binary..."
                install_grafana_binary
                ;;
        esac
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
    sudo systemctl stop grafana-server >/dev/null 2>&1 || true
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
    sudo systemctl stop grafana-server 2>/dev/null || true
    sudo apt update || true
    sudo apt install -y python3.12-venv || true
    sudo apt install -y nvidia-driver-550-server nvidia-utils-550-server || true

    pkill grafana-server 2>/dev/null || true
    sleep 1
    
    # Get current directory for absolute paths
    CURRENT_DIR=$(pwd)
    
    # Create Grafana directories
    mkdir -p data/grafana/{data,logs,plugins,provisioning/datasources,provisioning/dashboards}
    
    # Copy provisioning files
    cp -r config/grafana-provisioning/* data/grafana/provisioning/ 2>/dev/null || true
    
    # Ensure dashboard directory exists (dashboard should already be there from git)
    mkdir -p data/grafana-dashboards
    
    # Check if dashboard exists, if not create a basic one
    if [ ! -f "data/grafana-dashboards/llm-dashboard.json" ]; then
        print_warn "Dashboard file not found, creating basic dashboard..."
        cat > data/grafana-dashboards/llm-dashboard.json << 'EOF'
{
  "id": null,
  "title": "LLM Inference Server - Basic",
  "tags": ["llm", "inference", "metrics"],
  "timezone": "browser",
  "refresh": "10s",
  "time": {"from": "now-1h", "to": "now"},
  "panels": [
    {
      "id": 1,
      "title": "Total Requests",
      "type": "stat",
      "targets": [{"expr": "sum(http_requests_total)", "legendFormat": "Requests"}],
      "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}
    },
    {
      "id": 2,
      "title": "Memory Usage",
      "type": "stat",
      "targets": [{"expr": "memory_usage_bytes{type='used'} / 1024^3", "legendFormat": "Used GB"}],
      "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0}
    }
  ],
  "schemaVersion": 30,
  "version": 1
}
EOF
    fi
    
    # Update dashboard provisioning config with absolute path (use sed without -i '' for Linux)
    sed -i "s|path: data/grafana-dashboards|path: ${CURRENT_DIR}/data/grafana-dashboards|g" data/grafana/provisioning/dashboards/dashboards.yml
    
    # Create custom Grafana configuration with absolute paths
    cat > data/grafana/grafana.ini << EOF
[paths]
data = ${CURRENT_DIR}/data/grafana/data
logs = ${CURRENT_DIR}/data/grafana/logs
plugins = ${CURRENT_DIR}/data/grafana/plugins
provisioning = ${CURRENT_DIR}/data/grafana/provisioning

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
    
    # Determine Grafana homepath
    GRAFANA_HOMEPATH="/usr/share/grafana"
    if [ -d "/opt/grafana" ]; then
        GRAFANA_HOMEPATH="/opt/grafana"
    elif [ -d "/usr/local/share/grafana" ]; then
        GRAFANA_HOMEPATH="/usr/local/share/grafana"
    fi
    
    # Start Grafana with custom config
    grafana-server \
        --homepath "$GRAFANA_HOMEPATH" \
        --config ./data/grafana/grafana.ini \
        > logs/grafana.log 2>&1 &
    GRAFANA_PID=$!
    print_info "Grafana started with PID $GRAFANA_PID ✓"
    sleep 5  # Give Grafana more time to start and load provisioning
fi

# Step 12: Start main server
print_info "Starting LLM Inference Server on port $PORT..."
echo ""

if [ "$ENABLE_PROMETHEUS" = "yes" ] || [ "$ENABLE_GRAFANA" = "yes" ]; then
    print_info "🚀 Services will be available at:"
    print_info "  📡 Activate API Server:  http://localhost:$PORT"
    
    if [ "$ENABLE_PROMETHEUS" = "yes" ]; then
        print_info "  📊 Prometheus:      http://localhost:9090"
        print_info "  📈 Raw Metrics:     http://localhost:$PORT/metrics/prometheus"
    fi
    
    if [ "$ENABLE_GRAFANA" = "yes" ]; then
        print_info "  📋 Grafana:         http://localhost:3000 (admin/admin)"
    fi
    
    print_info "  ❤️  Health Check:    http://localhost:$PORT/health"
    print_info "  🔑 API Docs:        http://localhost:$PORT/docs"
    echo ""
    
    print_info "🔐 API Key Management:"
    print_info "  🌐 Web Interface:      http://localhost:$PORT/docs (Swagger UI)"
    print_info "  ➕ Create key:         POST http://localhost:$PORT/auth/keys"
    print_info "  📋 List keys:          GET  http://localhost:$PORT/auth/keys"
    print_info "  📊 Usage stats:        GET  http://localhost:$PORT/auth/keys/usage/stats"
    echo ""
    
    if [ "$ENABLE_GRAFANA" = "yes" ]; then
        print_info "💡 Grafana Setup:"
        print_info "  ✅ Prometheus data source automatically configured"
        print_info "  ✅ Activate Dashboard automatically imported and ready"
        print_info "  🔑 Login: admin/admin (change password when prompted)"
        print_info "  📊 Dashboard: LLM Inference Server Dashboard"
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
