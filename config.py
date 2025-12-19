import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Configuration
    app_name: str = "Persian RAG System"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 50 * 1024 * 1024 # 50MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".xlsx"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # CromaDB Configuration
    chromadb_path: str = "./chromadb"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_dimension: int = 384
    
    # LLM Configuration
    llm_provider: str = "openai" # "qwen" or "openai"
    
    # Qwen Configuration
    qwen_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    qwen_max_tokens: int = 512
    qwen_temperature: float = 0.7
    qwen_device: str = "auto"  # "cpu", "cuda", or "auto"
    
    # OpenAI Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    openai_max_tokens: int = 4096
    openai_temperature: float = 1
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Reranker Configuration
    use_reranking: bool = True
    rerank_model: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"
    rerank_top_k: int = 5
    
    # Search Configuration
    search_k: int = 5
    search_type: str = "similarity"  # "similarity" or "mmr"
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # Security
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.vector_store_path).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)
