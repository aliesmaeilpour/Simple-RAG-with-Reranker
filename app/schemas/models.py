from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import settings

# Session IDs become Qdrant payload values and URL path segments; keep them
# short and restricted to safe characters.
SESSION_ID_PATTERN = r"^[A-Za-z0-9._:@-]{1,128}$"


class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT = "text"
    TABLE = "table"


class DocumentType(str, Enum):
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class LLMProvider(str, Enum):
    """LLM provider enumeration"""
    QWEN = "qwen"
    OPENAI = "openai"


class DocumentMetadata(BaseModel):
    """Metadata about an indexed document"""
    doc_id: str
    file_name: str
    doc_type: Optional[DocumentType] = None
    upload_time: Optional[datetime] = None
    page_count: int = 0
    chunk_count: int = 0
    text_chunks: int = 0
    table_count: int = 0
    image_count: int = 0
    filtered_chunks: int = 0
    hierarchy_levels: int = Field(default=0, description="Deepest detected heading level")
    heading_count: int = Field(default=0, description="Number of detected section headings")
    tables_requiring_ocr: int = Field(default=0, description="Image tables without extracted cell text")
    sheet_count: Optional[int] = None
    size_bytes: int = 0


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    success: bool
    message: str
    doc_id: Optional[str] = None
    indexed_chunks: Optional[int] = None
    metadata: Optional[DocumentMetadata] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    total: int
    documents: List[DocumentMetadata]


class DeleteDocument(BaseModel):
    """Response model for deleting one document's chunks"""
    success: bool
    session_id: str
    doc_id: str


class ContextEvaluationOptions(BaseModel):
    evaluation_mode: Literal["diagnostics", "llm", "off"] = "diagnostics"
    evaluate_faithfulness: bool = Field(
        default=False,
        description="Judge factual answer claims against the prompt sources after generation; adds model calls",
    )
    relevant_chunk_ids: Optional[List[Annotated[str, Field(min_length=1, max_length=256)]]] = Field(
        default=None, min_length=1, max_length=10_000,
        description="Known relevant source chunk IDs; required for reference-based ranking metrics",
    )


class QueryRequest(ContextEvaluationOptions):
    """Request model for querying documents"""
    question: str = Field(..., min_length=1, max_length=settings.max_question_length)
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN)
    doc_ids: Optional[List[str]] = None
    llm_provider: Optional[LLMProvider] = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_tables: bool = True
    rerank: bool = True
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


class QueryRequestStream(ContextEvaluationOptions):
    """Request model for streaming queries"""
    question: str = Field(..., min_length=1, max_length=settings.max_question_length)
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN)
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=100)
    include_tables: bool = True
    rerank: bool = True
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


class ChunkMetadata(BaseModel):
    """A retrieved chunk returned as a cited source"""
    chunk_id: str
    doc_id: Optional[str] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    content_type: Optional[ContentType] = None
    score: float = 0.0
    text_preview: str = ""


class ContextAssessment(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: Optional[str] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    query_term_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    matched_query_terms: List[str] = Field(default_factory=list)
    requires_ocr: bool = False
    relevance_score: Optional[int] = Field(default=None, ge=0, le=3)
    relevance_reason: Optional[str] = None


class RetrievalMetrics(BaseModel):
    k: int = Field(ge=1)
    relevant_count: int = Field(ge=1)
    relevant_retrieved: int = Field(ge=0)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)


class ContextEvaluation(BaseModel):
    method: Literal["diagnostics", "llm"]
    status: Literal["completed", "no_contexts", "unavailable"]
    retrieved_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    dropped_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    query_term_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    ocr_required_count: int = Field(ge=0)
    contexts: List[ContextAssessment] = Field(default_factory=list)
    ranking_metrics: Optional[RetrievalMetrics] = None
    mean_relevance_score: Optional[float] = Field(default=None, ge=0, le=3)
    judged_context_precision: Optional[float] = Field(default=None, ge=0, le=1)
    judge_usage: Dict[str, int] = Field(default_factory=dict)
    evaluation_time: float = Field(default=0, ge=0)
    notes: List[str] = Field(default_factory=list)


class FaithfulnessEvidence(BaseModel):
    context_rank: int = Field(ge=1)
    chunk_id: Optional[str] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    quote: str


class FaithfulnessClaim(BaseModel):
    claim_id: int = Field(ge=1)
    statement: str
    verdict: Literal["supported", "contradicted", "not_enough_evidence"]
    reason: str
    evidence: List[FaithfulnessEvidence] = Field(default_factory=list)


class FaithfulnessEvaluation(BaseModel):
    method: Literal["llm_claim_verification"] = "llm_claim_verification"
    status: Literal["completed", "no_contexts", "no_answer", "no_claims", "unavailable"]
    score: Optional[float] = Field(default=None, ge=0, le=1)
    claim_count: Optional[int] = Field(default=None, ge=0)
    supported_claim_count: Optional[int] = Field(default=None, ge=0)
    unsupported_claim_count: Optional[int] = Field(default=None, ge=0)
    claims: List[FaithfulnessClaim] = Field(default_factory=list)
    judge_usage: Dict[str, int] = Field(default_factory=dict)
    evaluation_time: float = Field(default=0, ge=0)
    notes: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response model for queries"""
    success: bool
    question: str
    answer: str
    sources: List[ChunkMetadata] = []
    context_evaluation: Optional[ContextEvaluation] = None
    faithfulness_evaluation: Optional[FaithfulnessEvaluation] = None
    llm_provider: LLMProvider
    processing_time: float
    metadata: Dict[str, Any] = {}
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    uptime: float
    components: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: bool = True
    status_code: int
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(ContextEvaluationOptions):
    message: str = Field(..., min_length=1, max_length=settings.max_question_length)
    session_id: str = Field(..., pattern=SESSION_ID_PATTERN)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    input_tokens: int = 0
    output_tokens: int = 0
    context_evaluation: Optional[ContextEvaluation] = None
    faithfulness_evaluation: Optional[FaithfulnessEvaluation] = None


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


class DeleteSession(BaseModel):
    success: bool
    session_id: str


class Pong(BaseModel):
    ok: bool


class DocumentStatusResponse(BaseModel):
    session_id: str
    has_documents: bool


class DocumentDetailResponse(BaseModel):
    session_id: str
    has_documents: bool
    doc_ids: list[str]
    total_chunks: int | None = None