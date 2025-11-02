import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Configuration
    app_name: str = "Persian RAG System"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Document Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".xlsx"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Vector Store Configuration
    vector_store_path: str = Field(default="./vector_store", env="VECTOR_STORE_PATH")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    
    # LLM Configuration
    llm_provider: str = Field(default="qwen", env="LLM_PROVIDER")  # "qwen" or "openai"
    
    # Qwen Configuration
    qwen_model_name: str = Field(default="Qwen/Qwen2-7B-Instruct", env="QWEN_MODEL")
    qwen_max_tokens: int = Field(default=2048, env="QWEN_MAX_TOKENS")
    qwen_temperature: float = Field(default=0.7, env="QWEN_TEMPERATURE")
    qwen_device: str = Field(default="auto", env="QWEN_DEVICE")  # "cpu", "cuda", or "auto"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2048, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Reranker Configuration
    rerank_enabled: bool = Field(default=True, env="RERANK_ENABLED")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        env="RERANK_MODEL"
    )
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")
    
    # Search Configuration
    search_k: int = Field(default=10, env="SEARCH_K")
    search_type: str = Field(default="similarity", env="SEARCH_TYPE")  # "similarity" or "mmr"
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # Security
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.vector_store_path).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)
