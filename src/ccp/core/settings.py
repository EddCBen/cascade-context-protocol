from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = "local" # Default to local
    local_llm_base_url: str = "http://local-models:8081/v1"
    local_llm_model: str = "DeepSeek-R1-Distill-Qwen-1.5B"
    local_llm_api_key: str = "sk-no-key-required"

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
