from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = "local" # Default to local
    # Updated to point to the local-models service (Docker Compose service name)
    # Defaulting to localhost for host-based execution (uvicorn),
    # Docker containers will override this via ENV vars if needed.
    local_llm_base_url: str = "http://localhost:8081/v1"
    embedding_service_url: str = "http://localhost:8082"
    local_llm_model: str = "DeepSeek-R1-Distill-Qwen-1.5B"
    local_llm_api_key: str = "sk-no-key-required"
    context_window_size: int = 75 # Very low for demo granularity

    # Infrastructure
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_host: str = "localhost"
    redis_port: int = 6379
    mongo_uri: str = "mongodb://localhost:27017"
    embedding_dim: int = 384
    
    # Model Weights
    model_storage_path: str = "./local_models/weights" 


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields (like what might be in .env but not used here yet)
        extra = "ignore"

settings = Settings()
