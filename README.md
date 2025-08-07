# LLM Inference Server

Enterprise-grade LLM inference server built with Activate and FastAPI, providing an OpenAI-compatible API with production-ready features.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completion endpoints
- **Model Caching**: Intelligent LRU cache for multiple models with automatic memory management
- **Enterprise Security**: API key authentication, CORS, rate limiting
- **Comprehensive Monitoring**: 16+ metrics including token generation rate, memory usage, API key tracking
- **Production Ready**: Async processing, graceful shutdown, error handling
- **MLX Optimized**: Native Metal GPU acceleration on Apple Silicon
- **Grafana Dashboard**: Pre-configured dashboard with all telemetry metrics
- **Smart Response Handling**: Prevents LLM response continuation beyond prompts

## Quick Start

1. **Clone and Setup**
```bash
git clone <repository>
cd llm-inference-server
./setup.sh  # Starts server + Prometheus automatically
```

2. **Configure Environment (Optional)**
```bash
cp config/.env.example .env
# Edit .env with your settings
```

3. **Access Services**
```bash
# Activate API Server:     http://localhost:8000
# Prometheus:         http://localhost:9090
# Grafana Dashboard:  http://localhost:3000 (admin/admin)  
# API Documentation:  http://localhost:8000/docs
```

## API Usage

### Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics
```bash
# JSON format
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:8000/metrics/prometheus
```

## Configuration

Key environment variables:

- `API_KEYS`: Comma-separated list of API keys (optional)
- `DEFAULT_MODEL`: Default model to load (mlx-community/Llama-3.2-1B-Instruct-bf16)
- `MAX_MODEL_CACHE_SIZE`: Number of models to keep in memory
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `RATE_LIMIT_REQUESTS`: Requests per minute limit
- `TIMEOUT_SECONDS`: Request timeout in seconds

See `.env.example` for all options.

## Architecture

```
llm-inference-server/
├── app/                        # Main application code
│   ├── api/v1/                # API endpoints  
│   ├── core/                  # Configuration, logging
│   ├── models/                # Pydantic schemas
│   ├── services/              # Model management
│   └── utils/                 # Security, middleware, metrics
├── config/                     # Configuration files
│   ├── prometheus.yml         # Prometheus config
│   ├── logging_config.yaml    # Logging setup
│   ├── grafana-*.yml          # Grafana configs
│   └── .env.example           # Environment template
├── data/                       # Runtime data
│   ├── grafana-dashboards/    # Dashboard definitions
│   ├── grafana/               # Grafana runtime data
│   └── prometheus/            # Time series data
├── logs/                      # Application logs
└── scripts/                   # Utility scripts
    └── update_dashboards.sh   # Dashboard updates
```

## Usage Options

```bash
# Full stack (default) - Activate + Prometheus + Grafana
./setup.sh

# Production mode with full monitoring
./setup.sh --production

# Without Grafana (Prometheus only)
./setup.sh --no-grafana

# Without any monitoring
./setup.sh --no-prometheus

# Skip dependency installation
./setup.sh --no-install

# Custom port
PORT=8001 ./setup.sh
```

## Monitoring & Metrics

The server provides comprehensive telemetry:

**16+ Metrics Available:**
- Token generation rate (tokens/second)
- Time to first token
- Concurrent inference tracking
- Memory usage (used/total/percent)
- GPU utilization
- API key usage tracking
- Model cache operations (hit/miss/eviction)
- Request success/failure rates
- Error categorization
- Response cleaning operations

**Access Metrics:**
- Grafana Dashboard: `http://localhost:3000` (admin/admin)
- Prometheus: `http://localhost:9090`
- JSON API: `http://localhost:8000/metrics`

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- 8GB+ RAM recommended
- MLX-compatible models

## License

[Your License]