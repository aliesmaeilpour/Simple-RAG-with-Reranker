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
    file_name: str = Field(..., description="Name of the uploaded file")
    file_content: bytes = Field(..., description="Binary content of the file")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    @validator('file_name')
    def validate_file_name(cls, v):
        allowed_extensions = ['.pdf', '.docx', '.xlsx']
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"File type not supported. Allowed: {allowed_extensions}")
        return v


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    doc_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    doc_type: DocumentType = Field(..., description="Document type")
    upload_time: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    page_count: int = Field(default=0, description="Number of pages")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    table_count: int = Field(default=0, description="Number of tables extracted")
    size_bytes: int = Field(default=0, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    doc_id: Optional[str] = Field(None, description="Document ID if applicable")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    total: int = Field(..., description="Total number of documents")
    documents: List[DocumentMetadata] = Field(..., description="List of document metadata")


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
    table_only: bool = Field(True, description="Search only in table content")
    return_markdown: bool = Field(True, description="Return tables in markdown format")


class ChunkMetadata(BaseModel):
    """Metadata for document chunks"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document ID")
    page_number: int = Field(..., description="Page number in document")
    content_type: ContentType = Field(..., description="Type of content")
    confidence_score: float = Field(0.0, description="Retrieval confidence score")
    text_preview: str = Field(..., description="Preview of chunk content")


class QueryResponse(BaseModel):
    """Response model for queries"""
    success: bool = Field(..., description="Query success status")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[ChunkMetadata] = Field(..., description="Source chunks used")
    llm_provider: LLMProvider = Field(..., description="LLM provider used")
    processing_time: float = Field(..., description="Query processing time in seconds")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="System health status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")
    components: Dict[str, bool] = Field(..., description="Component health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")


class ReindexRequest(BaseModel):
    """Request model for reindexing"""
    doc_ids: Optional[List[str]] = Field(None, description="Specific documents to reindex")
    force: bool = Field(False, description="Force reindex even if up to date")
    optimize: bool = Field(True, description="Optimize index after reindexing")


class ReindexResponse(BaseModel):
    """Response model for reindexing"""
    success: bool = Field(..., description="Reindex success status")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    processing_time: float = Field(..., description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
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