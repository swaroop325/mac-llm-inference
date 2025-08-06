#!/bin/bash
# Optional Grafana setup script

print_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
print_warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

print_info "Setting up Grafana for MLX Inference Server..."

# Check if Grafana is installed
if ! command -v grafana-server &> /dev/null; then
    print_warn "Grafana not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install grafana
    else
        print_error "Homebrew not found. Please install Grafana manually:"
        print_error "  brew install grafana"
        print_error "  Or visit: https://grafana.com/docs/grafana/latest/installation/"
        exit 1
    fi
fi

# Create Grafana data directory
mkdir -p grafana-data

# Start Grafana
print_info "Starting Grafana on port 3000..."
grafana-server \
    --homepath /usr/local/share/grafana \
    --config /usr/local/etc/grafana/grafana.ini \
    cfg:default.paths.data=./grafana-data \
    cfg:default.paths.logs=./grafana-data/logs \
    cfg:default.paths.plugins=./grafana-data/plugins \
    cfg:default.paths.provisioning=./grafana-provisioning \
    &

GRAFANA_PID=$!
print_info "Grafana started with PID $GRAFANA_PID"

echo ""
print_info "🎯 Grafana Setup Complete!"
print_info "  📊 Grafana:     http://localhost:3000 (admin/admin)"
print_info "  📈 Prometheus:  http://localhost:9090"
print_info ""
print_info "Next steps:"
print_info "  1. Login to Grafana (admin/admin)"  
print_info "  2. Add Prometheus data source: http://localhost:9090"
print_info "  3. Import the dashboard from grafana-dashboard.json"
print_info ""
print_info "Press Ctrl+C to stop Grafana"

# Wait for interrupt
trap "kill $GRAFANA_PID; exit 0" SIGINT SIGTERM
wait $GRAFANA_PID