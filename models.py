from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"


class DocumentType(str, Enum):
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class LLMProvider(str, Enum):
    """LLM provider enumeration"""
    QWEN = "qwen"
    OPENAI = "openai"


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    file_name: str
    file_content: bytes
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('file_name')
    def validate_file_name(cls, v):
        allowed_extensions = ['.pdf', '.docx', '.xlsx']
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"File type not supported. Allowed: {allowed_extensions}")
        return v


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    doc_id: str
    file_name: str
    doc_type: DocumentType
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    page_count: int = 0
    chunk_count: int = 0
    table_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = {}


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    success: bool
    message: str
    doc_id: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    total: int
    documents: List[DocumentMetadata]


class QueryRequest(BaseModel):
    """Request model for querying documents"""
    question: str
    session_id: str
    doc_ids: Optional[List[str]] = None
    llm_provider: Optional[LLMProvider] = LLMProvider.OPENAI
    top_k: Optional[int] = 5
    include_tables: bool = True
    rerank: bool = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class TableQueryRequest(QueryRequest):
    """Request model for table-specific queries"""
    table_only: bool = True
    return_markdown: bool = True


class ChunkMetadata(BaseModel):
    """Metadata for document chunks"""
    chunk_id: str
    doc_id: str
    page_number: int
    content_type: ContentType
    confidence_score: float = 0.0
    text_preview: str


class QueryResponse(BaseModel):
    """Response model for queries"""
    success: bool
    question: str
    answer: str
    sources: List[ChunkMetadata]
    llm_provider: LLMProvider
    processing_time: float
    metadata: Dict[str, Any] = {}
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime: float
    components: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")


class ReindexRequest(BaseModel):
    """Request model for reindexing"""
    doc_ids: Optional[List[str]] = None
    force: bool = False
    optimize: bool = True


class ReindexResponse(BaseModel):
    """Response model for reindexing"""
    success: bool
    message: str
    documents_processed: int
    chunks_indexed: int
    processing_time: float


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    input_tokens: int = 0
    output_tokens: int = 0


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None


class SessionHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int


class SessionStats(BaseModel):
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    total_characters: Optional[int] = None


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: bool = True
    status_code: int
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class HealthStatus(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"