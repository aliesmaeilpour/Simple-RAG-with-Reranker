from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.routes import chat, db_stats, health, doc

router = APIRouter(prefix="/v1")

# Liveness/readiness stays public for orchestrators.
router.include_router(health.router)
router.include_router(doc.router)
router.include_router(chat.router)
router.include_router(db_stats.router)




# import asyncio
# import io
# import json
# import logging
# import time
# import uuid
# import zipfile
# from contextlib import aclosing
# from pathlib import Path as FilePath
# from typing import Any, Optional, Annotated

# from fastapi import APIRouter, Depends, File, HTTPException, Path, Request, UploadFile, Query
# from fastapi.responses import StreamingResponse

# from app.core.config import settings
# from app.schemas.models import (
#     ChatMessage,
#     ChatRequest,
#     ChatResponse,
#     ChunkMetadata,
#     ContentType,
#     DeleteDocument,
#     DeleteSession,
#     DocumentListResponse,
#     DocumentResponse,
#     HealthCheckResponse,
#     LLMProvider,
#     MessageRole,
#     Pong,
#     QueryRequest,
#     QueryRequestStream,
#     QueryResponse,
#     SessionHistory,
#     SessionInfo,
#     SessionListResponse,
#     SessionStats,
#     DocumentStatusResponse,
#     DocumentDetailResponse
# )
# from app.service.document_processor import DocumentProcessor
# from app.service.rag_engine import RAGEngine

# logger = logging.getLogger(__name__)

# router = APIRouter()

# _chatbot: Optional[RAGEngine] = None
# _processor: Optional[DocumentProcessor] = None


# def get_chatbot() -> RAGEngine:
#     """Dependency to get chatbot instance."""
#     if _chatbot is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Chatbot service not initialized"
#         )
#     return _chatbot


# def set_chatbot(chatbot: Optional[RAGEngine]):
#     """Set the global chatbot instance."""
#     global _chatbot
#     _chatbot = chatbot


# def get_processor() -> DocumentProcessor:
#     """Lazily create the document processor (avoids import-time side effects)."""
#     global _processor
#     if _processor is None:
#         _processor = DocumentProcessor()
#     return _processor


# def _detect_document_type(content: bytes) -> Optional[str]:
#     """Sniff the real file type from magic bytes; None when unrecognized.

#     Extension checks alone accept renamed payloads; the ZIP case looks at
#     the archive members to tell DOCX (word/) from XLSX (xl/).
#     """
#     if content.startswith(b"%PDF-"):
#         return "pdf"
#     if content.startswith(b"PK\x03\x04"):
#         try:
#             with zipfile.ZipFile(io.BytesIO(content)) as archive:
#                 names = archive.namelist()
#         except Exception:
#             return None
#         if any(name.startswith("word/") for name in names):
#             return "docx"
#         if any(name.startswith("xl/") for name in names):
#             return "xlsx"
#         return None
#     return None


# def _to_sources(contexts: list[dict[str, Any]]) -> list[ChunkMetadata]:
#     """Map retrieved chunks onto the citation schema returned to clients."""
#     sources: list[ChunkMetadata] = []
#     for chunk in contexts:
#         raw_type = chunk.get("content_type")
#         try:
#             content_type = ContentType(raw_type) if raw_type else None
#         except ValueError:
#             content_type = None
#         score = chunk.get("rerank_score", chunk.get("score", 0.0))
#         sources.append(ChunkMetadata(
#             chunk_id=str(chunk.get("id", "")),
#             doc_id=chunk.get("doc_id"),
#             file_name=chunk.get("file_name") or chunk.get("source"),
#             page_number=chunk.get("page_number"),
#             content_type=content_type,
#             score=float(score or 0.0),
#             text_preview=(chunk.get("content") or "")[:200],
#         ))
#     return sources


# @router.get("/ping", tags=["Ping"], response_model=Pong)
# async def ping():
#     return Pong(ok=True)


# @router.get(
#     "/health",
#     response_model=HealthCheckResponse,
#     tags=["Health"],
#     description="Readiness of the process and its backing components.",
# )
# async def health(request: Request):
#     chatbot: Optional[RAGEngine] = getattr(request.app.state, "chatbot", None)
#     if chatbot is None:
#         return HealthCheckResponse(
#             status="unhealthy",
#             version=settings.app_version,
#             uptime=0.0,
#             components={"qdrant": False, "embeddings": False, "llm": False},
#         )
#     components = await chatbot.component_health()
#     status = "healthy" if all(components.values()) else "degraded"
#     uptime = time.monotonic() - getattr(request.app.state, "start_time", time.monotonic())
#     return HealthCheckResponse(
#         status=status,
#         version=settings.app_version,
#         uptime=uptime,
#         components=components,
#     )


# @router.post(
#     "/documents/{session_id}/upload",
#     response_model=DocumentResponse,
#     tags=["Documents"],
#     responses={
#         200: {"description": "Successful response"},
#         400: {"description": "Invalid request"},
#         413: {"description": "File too large"},
#         503: {"description": "Service unavailable"},
#     }
# )
# async def upload_document(
#     file: UploadFile = File(...),
#     session_id: str = Path(..., description="Session ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     file_path = None
#     try:
#         file_extension = FilePath(file.filename or "").suffix.lower()
#         if file_extension not in settings.allowed_extensions:
#             raise HTTPException(status_code=400, detail="Unsupported file type")

#         content = await file.read(settings.max_file_size + 1)
#         if len(content) > settings.max_file_size:
#             raise HTTPException(status_code=413, detail="File exceeds MAX_FILE_SIZE")

#         detected = _detect_document_type(content)
#         if detected != file_extension.lstrip("."):
#             raise HTTPException(
#                 status_code=400,
#                 detail="File content does not match its extension",
#             )

#         processor = get_processor()
#         unique_filename = f"{uuid.uuid4()}{file_extension}"
#         file_path = processor.upload_dir / unique_filename
#         await asyncio.to_thread(file_path.write_bytes, content)

#         chunks, meta = await processor.process_file(
#             file_path, original_name=file.filename
#         )
#         for chunk in chunks:
#             chunk.metadata["file_name"] = file.filename
#             chunk.metadata["source"] = file.filename
#         meta["file_name"] = file.filename
#         meta["size_bytes"] = len(content)

#         async with chatbot.session_locks.acquire(session_id):
#             point_ids = await chatbot.history_manager.add_document_chunks(
#                 document_chunks=chunks,
#                 session_id=session_id,
#                 doc_type=meta.get("doc_type", file_extension[1:]),
#             )

#         return DocumentResponse(
#             success=True,
#             message="Document uploaded and indexed",
#             doc_id=meta.get("doc_id"),
#             indexed_chunks=len(point_ids),
#             metadata=meta,
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Document upload failed for session %s", session_id)
#         raise HTTPException(status_code=500, detail="Document processing error")
#     finally:
#         if file_path is not None:
#             await asyncio.to_thread(file_path.unlink, missing_ok=True)


# @router.get(
#     "/documents/{session_id}",
#     response_model=DocumentListResponse,
#     tags=["Documents"],
#     description="List documents indexed for a session.",
# )
# async def list_documents(
#     session_id: str = Path(..., description="Session ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     try:
#         documents = await chatbot.list_documents(session_id)
#         return DocumentListResponse(
#             documents=documents,
#             total=len(documents),
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Failed to list documents for session %s", session_id)
#         raise HTTPException(status_code=500, detail="Failed to list documents")


# @router.delete(
#     "/documents/{session_id}/{doc_id}",
#     response_model=DeleteDocument,
#     tags=["Documents"],
#     description="Delete one document's indexed chunks from a session.",
# )
# async def delete_document(
#     session_id: str = Path(..., description="Session ID"),
#     doc_id: str = Path(..., description="Document ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     try:
#         success = await chatbot.delete_document(session_id, doc_id)
#         if not success:
#             raise HTTPException(status_code=404, detail="Document not found")
#         return DeleteDocument(success=success, session_id=session_id, doc_id=doc_id)
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception(
#             "Failed to delete document %s for session %s", doc_id, session_id
#         )
#         raise HTTPException(status_code=500, detail="Failed to delete document")


# @router.post(
#     "/chat",
#     response_model=QueryResponse,
#     tags=["Chat"],
#     responses={
#         200: {"description": "Successful response"},
#         400: {"description": "Invalid request"},
#         503: {"description": "Service unavailable"},
#     }
# )
# async def query_docs(
#     req: QueryRequest,
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     try:
#         provider = req.llm_provider or LLMProvider(settings.llm_provider)
#         if provider.value != settings.llm_provider:
#             raise HTTPException(status_code=400, detail="Requested LLM provider is not configured")

#         result = await chatbot.answer_question(
#             req.question,
#             session_id=req.session_id,
#             llm_provider=provider.value,
#             top_k=req.top_k,
#             doc_ids=req.doc_ids,
#             include_tables=req.include_tables,
#             rerank=req.rerank,
#             temperature=req.temperature,
#             max_tokens=req.max_tokens,
#             evaluation_mode=req.evaluation_mode,
#             relevant_chunk_ids=req.relevant_chunk_ids,
#             evaluate_faithfulness=req.evaluate_faithfulness,
#         )
#         return QueryResponse(
#             success=True,
#             question=req.question,
#             answer=result["response"],
#             sources=_to_sources(result["metadata"]["context_used"]),
#             context_evaluation=result["metadata"].get("context_evaluation"),
#             faithfulness_evaluation=result["metadata"].get("faithfulness_evaluation"),
#             processing_time=result["processing_time"],
#             llm_provider=provider,
#             input_tokens=result["input_tokens"],
#             output_tokens=result["output_tokens"],
#             reasoning_tokens=result["reasoning_tokens"]
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Chat processing failed")
#         raise HTTPException(status_code=500, detail="Chat processing error")


# @router.post(
#     "/chat/stream",
#     tags=["Chat"],
#     responses={
#         200: {"description": "Successful response"},
#         400: {"description": "Invalid request"},
#         503: {"description": "Service unavailable"},
#     }
# )
# async def chat_stream(
#     req: QueryRequestStream,
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     async def generate():
#         try:
#             async with aclosing(chatbot.answer_question_stream(
#                 query=req.question,
#                 session_id=req.session_id,
#                 top_k=req.top_k,
#                 doc_ids=req.doc_ids,
#                 include_tables=req.include_tables,
#                 rerank=req.rerank,
#                 temperature=req.temperature,
#                 max_tokens=req.max_tokens,
#                 evaluation_mode=req.evaluation_mode,
#                 relevant_chunk_ids=req.relevant_chunk_ids,
#                 evaluate_faithfulness=req.evaluate_faithfulness,
#             )) as stream:
#                 async for token in stream:
#                     if isinstance(token, dict):
#                         yield f"data: {json.dumps(token, ensure_ascii=False, default=str)}\n\n"
#                     else:
#                         yield "".join(f"data: {line}\n" for line in token.split("\n")) + "\n"
#             # [DONE] only emitted on clean completion
#             yield "data: [DONE]\n\n"
#         except Exception:
#             logger.exception("Streaming chat failed")
#             yield 'data: {"type": "error", "message": "Chat processing error"}\n\n'

#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no",
#         },
#     )


# @router.post(
#     "/query",
#     response_model=ChatResponse,
#     tags=["query"],
#     responses={
#         200: {"description": "Successful response"},
#         400: {"description": "Invalid request"},
#         503: {"description": "Service unavailable"},
#     }
# )
# async def chat(
#     request: ChatRequest,
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     """
#     Process a chat message and return the assistant's response.

#     The chatbot will:
#     - Retrieve relevant document chunks uploaded to the session
#     - Use conversation history for context
#     - Generate a helpful, grounded response with citations
#     """
#     try:
#         result = await chatbot.answer_question(
#             query=request.message,
#             session_id=request.session_id,
#             evaluation_mode=request.evaluation_mode,
#             relevant_chunk_ids=request.relevant_chunk_ids,
#             evaluate_faithfulness=request.evaluate_faithfulness,
#         )

#         metadata = result.get("metadata", {})

#         return ChatResponse(
#             response=result.get("response", ""),
#             session_id=request.session_id,
#             timestamp=metadata.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S"),
#             input_tokens=result.get("input_tokens", 0),
#             output_tokens=result.get("output_tokens", 0),
#             context_evaluation=metadata.get("context_evaluation"),
#             faithfulness_evaluation=metadata.get("faithfulness_evaluation"),
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Query processing failed")
#         raise HTTPException(status_code=500, detail="Chat processing error")


# @router.get(
#     "/sessions/all_sessions",
#     response_model=SessionListResponse,
#     description="Get a list of all chat sessions.",
#     tags=["Sessions"]
# )
# async def list_sessions(
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     """List all active chat sessions."""
#     try:
#         sessions_data = await chatbot.get_all_sessions()

#         sessions = []
#         for sid in sessions_data:
#             sessions.append(SessionInfo(
#                 session_id=sid.get("session_id"),
#                 message_count=sid.get("message_count", 0),
#                 created_at=sid.get("first_message"),
#                 last_activity=sid.get("last_message")
#             ))

#         return SessionListResponse(
#             sessions=sessions,
#             total=len(sessions)
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Failed to list sessions")
#         raise HTTPException(status_code=500, detail="Failed to list sessions")


# @router.get(
#     "/sessions/{session_id}/history",
#     response_model=SessionHistory,
#     description="Retrieve the full chat history for a session.",
#     tags=["Sessions"]
# )
# async def get_session_history(
#     session_id: str = Path(..., description="Session ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     """Get full conversation history for a session."""
#     try:
#         history = await chatbot.get_session_history(session_id)

#         messages = []
#         for msg in history:
#             messages.append(ChatMessage(
#                 role=MessageRole(msg.get("role", "user")),
#                 content=msg.get("content", ""),
#                 timestamp=msg.get("timestamp"),
#                 metadata=msg.get("metadata", {}) or (
#                     {"partial": True} if msg.get("partial") else {}
#                 ),
#             ))

#         return SessionHistory(
#             session_id=session_id,
#             messages=messages,
#             total_messages=len(messages)
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Failed to get history for session %s", session_id)
#         raise HTTPException(status_code=500, detail="Failed to get session history")


# @router.get(
#     "/sessions/{session_id}/stats",
#     response_model=SessionStats,
#     description="Get statistics about a chat session.",
#     tags=["Sessions"]
# )
# async def get_session_stats(
#     session_id: str = Path(..., description="Session ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     """Get statistics for a specific chat session."""
#     try:
#         stats = await chatbot.get_session_stats(session_id)

#         return SessionStats(
#             session_id=session_id,
#             total_messages=stats.get("message_count", 0),
#             user_messages=stats.get("user_messages", 0),
#             assistant_messages=stats.get("assistant_messages", 0),
#             first_message_at=stats.get("first_message_at"),
#             last_message_at=stats.get("last_message_at"),
#             total_characters=stats.get("total_characters")
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Failed to get stats for session %s", session_id)
#         raise HTTPException(status_code=500, detail="Failed to get session stats")


# @router.delete(
#     "/sessions/{session_id}/delete",
#     response_model=DeleteSession,
#     status_code=200,
#     description="Delete a chat session and its history.",
#     tags=["Sessions"]
# )
# async def delete_session(
#     session_id: str = Path(..., description="Session ID"),
#     chatbot: RAGEngine = Depends(get_chatbot)
# ):
#     """Delete a session and all its history."""
#     try:
#         success = await chatbot.delete_session(session_id)

#         return DeleteSession(
#             success=success,
#             session_id=session_id
#         )
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Failed to delete session %s", session_id)
#         raise HTTPException(status_code=500, detail="Failed to delete session")


# # @router.get(
# #     "/status",
# #     response_model=DocumentStatusResponse,
# #     summary="Check whether a session has any uploaded documents",
# # )
# # async def document_status(
# #     session_id: Annotated[str, Query(min_length=1)],
# #     store: StoreDep,
# # ) -> DocumentStatusResponse:
# #     """Cheap O(1) check — safe to call per-request."""
# #     has = await store.session_has_documents(session_id)
# #     return DocumentStatusResponse(session_id=session_id, has_documents=has)


# # @router.get(
# #     "/detail",
# #     response_model=DocumentDetailResponse,
# #     summary="List doc_ids and optionally chunk counts for a session",
# # )
# # async def document_detail(
# #     session_id: Annotated[str, Query(min_length=1)],
# #     store: StoreDep,
# #     include_chunk_count: Annotated[
# #         bool,
# #         Query(description="Set true to include total chunk count (full scroll — slow)"),
# #     ] = False,
# # ) -> DocumentDetailResponse:
# #     """Full-scroll endpoint — use for diagnostics, not real-time requests."""
# #     doc_ids = await store.list_session_doc_ids(session_id)
# #     total_chunks: int | None = None
# #     if include_chunk_count:
# #         total_chunks = await store.session_chunk_count(session_id)
# #     return DocumentDetailResponse(
# #         session_id=session_id,
# #         has_documents=bool(doc_ids),
# #         doc_ids=doc_ids,
# #         total_chunks=total_chunks,
# #     )
