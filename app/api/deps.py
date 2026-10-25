import io
import zipfile
from typing import Optional, Any

from fastapi import HTTPException

from app.schemas.models import (
    ChunkMetadata,
    ContentType,
)
from app.service.document_processor import DocumentProcessor
from app.service.rag_engine import RAGEngine

_chatbot: Optional[RAGEngine] = None
_processor: Optional[DocumentProcessor] = None


def get_chatbot() -> RAGEngine:
    """Dependency to get chatbot instance."""
    if _chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot service not initialized"
        )
    return _chatbot


def set_chatbot(chatbot: Optional[RAGEngine]):
    """Set the global chatbot instance."""
    global _chatbot
    _chatbot = chatbot


def get_processor() -> DocumentProcessor:
    """Lazily create the document processor (avoids import-time side effects)."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


def _detect_document_type(content: bytes) -> Optional[str]:
    """Sniff the real file type from magic bytes; None when unrecognized.

    Extension checks alone accept renamed payloads; the ZIP case looks at
    the archive members to tell DOCX (word/) from XLSX (xl/).
    """
    if content.startswith(b"%PDF-"):
        return "pdf"
    if content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
        except Exception:
            return None
        if any(name.startswith("word/") for name in names):
            return "docx"
        if any(name.startswith("xl/") for name in names):
            return "xlsx"
        return None
    return None


def _to_sources(contexts: list[dict[str, Any]]) -> list[ChunkMetadata]:
    """Map retrieved chunks onto the citation schema returned to clients."""
    sources: list[ChunkMetadata] = []
    for chunk in contexts:
        raw_type = chunk.get("content_type")
        try:
            content_type = ContentType(raw_type) if raw_type else None
        except ValueError:
            content_type = None
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        sources.append(ChunkMetadata(
            chunk_id=str(chunk.get("id", "")),
            doc_id=chunk.get("doc_id"),
            file_name=chunk.get("file_name") or chunk.get("source"),
            page_number=chunk.get("page_number"),
            content_type=content_type,
            score=float(score or 0.0),
            text_preview=(chunk.get("content") or "")[:200],
        ))
    return sources