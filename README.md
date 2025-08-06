# Activate LLM Inference Server

Enterprise-grade LLM inference server built with Activate and FastAPI, providing an OpenAI-compatible API with production-ready features.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completion endpoints
- **Model Caching**: Intelligent LRU cache for multiple models with automatic memory management
- **Enterprise Security**: API key authentication, CORS, rate limiting
- **Observability**: Structured JSON logging, Prometheus metrics, health checks
- **Production Ready**: Async processing, graceful shutdown, error handling
- **Activate Optimized**: Native Metal GPU acceleration on Apple Silicon

## Quick Start

1. **Clone and Setup**
```bash
git clone <repository>
cd mlx-inference-server
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
    "model": "mlx-community/Llama-3.2-1B-Instruct",
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
- `MODEL_PATH`: Default model to load
- `MAX_MODEL_CACHE_SIZE`: Number of models to keep in memory
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `RATE_LIMIT_REQUESTS`: Requests per minute limit

See `.env.example` for all options.

## Architecture

```
mlx-inference-server/
├── app/                    # Main application code
│   ├── api/v1/            # API endpoints  
│   ├── core/              # Configuration, logging
│   ├── models/            # Pydantic schemas
│   ├── services/          # Model management
│   └── utils/             # Security, middleware, metrics
├── config/                 # Configuration files
│   ├── prometheus.yml     # Prometheus config
│   ├── logging_config.yaml # Logging setup
│   ├── grafana-*.yml      # Grafana configs
│   └── .env.example       # Environment template
├── scripts/               # Utility scripts
│   ├── monitor_metrics.sh # Real-time monitoring
│   ├── start_grafana.sh   # Optional Grafana
│   └── test_openai_api.sh # API testing
├── docs/                  # Documentation
├── examples/              # Usage examples
├── logs/                  # Application logs
└── data/                  # Prometheus data
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

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- 8GB+ RAM recommended
- Activate-compatible models

## License

[Your License]