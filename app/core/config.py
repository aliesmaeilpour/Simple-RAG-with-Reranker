from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings"""

    # API Configuration
    app_name: str = "Persian RAG System"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1)

    # Document Processing
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = Field(default=[".pdf", ".txt", ".docx"])
    # Total character budget for retrieved document context sent to the LLM
    max_context_chars: int = Field(default=24_000, gt=0)
    # Hard cap on incoming question/message length (schema-level validation)
    max_question_length: int = Field(default=8_000, gt=0)

    # Vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: int = Field(default=30, gt=0)
    qdrant_messages_collection: str = "chat_messages_1"
    qdrant_documents_collection: str = "chat_documents"
    qdrant_batch_size: int = Field(default=128, gt=0)

    # Embeddings: "openai" or "local"; None follows the LLM provider
    embedding_provider: Optional[str] = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM Configuration
    llm_provider: str = "openai"  # "qwen" or "openai"

    # Qwen Configuration
    qwen_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    qwen_max_tokens: int = 512
    qwen_temperature: float = 0.7
    qwen_device: str = "auto"  # "cpu", "cuda", or "auto"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    openai_max_tokens: int = 4096
    openai_temperature: float = 1
    openai_timeout: float = Field(default=60, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: Optional[int] = Field(default=None, gt=0)
    embedding_batch_size: int = Field(default=128, gt=0, le=2048)

    # Reranker Configuration
    use_reranking: bool = True
    rerank_model: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"
    rerank_top_k: int = Field(default=5, gt=0)

    # Search Configuration
    # Minimum cosine similarity for vector-search results; 0 disables
    # pre-filtering. text-embedding-3-* models rarely score above ~0.5 even
    # for relevant text, so a higher cutoff silently empties retrieval.
    search_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    hybrid_search_enabled: bool = True
    hybrid_prefetch_limit: int = Field(default=50, gt=0)
    bm25_k1: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_average_length: float = Field(default=256, gt=0, allow_inf_nan=False)

    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/app.log")

    # Security
    cors_origins: List[str] = Field(default=["*"])
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, value: str) -> str:
        allowed = {"openai", "qwen"}
        if value not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(allowed)}")
        return value

    @field_validator("embedding_provider")
    @classmethod
    def _validate_embedding_provider(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in {"openai", "local"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'local'")
        return value

    @property
    def resolved_embedding_provider(self) -> str:
        """Embedding backend: explicit setting, otherwise follows llm_provider."""
        return self.embedding_provider or (
            "openai" if self.llm_provider == "openai" else "local"
        )


# Global settings instance
settings = Settings()
