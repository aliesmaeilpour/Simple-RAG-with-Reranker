import asyncio
import uuid
from pathlib import Path as FilePath
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile

from app.core.config import settings
from app.schemas.models import (
    DeleteDocument,
    DocumentListResponse,
    DocumentResponse,
)
from app.service.rag_engine import RAGEngine
from app.api.deps import _detect_document_type, get_chatbot, get_processor


logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/documents/{session_id}/upload",
    response_model=DocumentResponse,
    tags=["Documents"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        413: {"description": "File too large"},
        503: {"description": "Service unavailable"},
    }
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    file_path = None
    try:
        file_extension = FilePath(file.filename or "").suffix.lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        content = await file.read(settings.max_file_size + 1)
        if len(content) > settings.max_file_size:
            raise HTTPException(status_code=413, detail="File exceeds MAX_FILE_SIZE")

        detected = _detect_document_type(content)
        if detected != file_extension.lstrip("."):
            raise HTTPException(
                status_code=400,
                detail="File content does not match its extension",
            )

        processor = get_processor()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = processor.upload_dir / unique_filename
        await asyncio.to_thread(file_path.write_bytes, content)

        chunks, meta = await processor.process_file(
            file_path, original_name=file.filename
        )
        for chunk in chunks:
            chunk.metadata["file_name"] = file.filename
            chunk.metadata["source"] = file.filename
        meta["file_name"] = file.filename
        meta["size_bytes"] = len(content)

        async with chatbot.session_locks.acquire(session_id):
            point_ids = await chatbot.history_manager.add_document_chunks(
                document_chunks=chunks,
                session_id=session_id,
                doc_type=meta.get("doc_type", file_extension[1:]),
            )

        return DocumentResponse(
            success=True,
            message="Document uploaded and indexed",
            doc_id=meta.get("doc_id"),
            indexed_chunks=len(point_ids),
            metadata=meta,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document upload failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Document processing error")
    finally:
        if file_path is not None:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)


@router.get(
    "/documents/{session_id}",
    response_model=DocumentListResponse,
    tags=["Documents"],
    description="List documents indexed for a session.",
)
async def list_documents(
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    try:
        documents = await chatbot.list_documents(session_id)
        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list documents for session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.delete(
    "/documents/{session_id}/{doc_id}",
    response_model=DeleteDocument,
    tags=["Documents"],
    description="Delete one document's indexed chunks from a session.",
)
async def delete_document(
    session_id: str = Path(..., description="Session ID"),
    doc_id: str = Path(..., description="Document ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    try:
        success = await chatbot.delete_document(session_id, doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return DeleteDocument(success=success, session_id=session_id, doc_id=doc_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Failed to delete document %s for session %s", doc_id, session_id
        )
        raise HTTPException(status_code=500, detail="Failed to delete document")