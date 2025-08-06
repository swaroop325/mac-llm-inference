from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from functools import lru_cache


class Settings(BaseSettings):
    model_config = {
        "protected_namespaces": (),
        "case_sensitive": False
    }
    
    app_name: str = "Activate LLM Inference Server"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # Activate requires single worker for GPU access
    
    # Model Configuration
    model_path: str = "mlx-community/Llama-3.2-1B-Instruct"
    model_cache_dir: Optional[str] = os.path.expanduser("~/.cache/mlx-models")
    max_model_cache_size: int = 5  # Maximum number of models to keep in memory
    
    # Inference Configuration
    default_temperature: float = 0.7
    default_max_tokens: int = 256
    max_allowed_tokens: int = 4096
    timeout_seconds: int = 300
    model_load_timeout_seconds: int = 600  # 10 minutes for model loading
    
    # Memory Management
    max_memory_gb: float = 8.0
    gpu_memory_fraction: float = 0.8
    
    # Security
    api_key_header: str = "X-API-Key"
    api_keys: List[str] = []  # Load from environment
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_period: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = "logs/mlx_server.log"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
        
    @property
    def is_production(self) -> bool:
        return not self.debug
        
    def model_post_init(self, __context=None):
        # Parse API keys from environment
        api_keys_env = os.getenv("API_KEYS", "")
        if api_keys_env:
            self.api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()