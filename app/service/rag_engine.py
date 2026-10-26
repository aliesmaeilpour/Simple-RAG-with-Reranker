import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from typing import Any, Optional

import jdatetime

from app.core.config import settings
from app.core.concurrency import SessionLocks
from app.db.chat_history import QdrantChatHistory
from app.llm.provider import LanguageModel
from app.retrieval.evaluation import evaluate_contexts
from app.retrieval.faithfulness import evaluate_faithfulness as evaluate_answer_faithfulness
from app.llm.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant for question answering over the user's documents \
(Persian and English). When the user's message contains a "Document Results" \
section, follow these rules strictly:
- Answer using the provided document context; do not contradict it.
- Cite the sources you used inline, e.g. [report.pdf, صفحه ۳] or \
[قرارداد.docx], based on the file names and page markers in the context.
- If the context does not contain the answer, say so plainly; do not invent \
information, numbers, tables, or citations.
- Tables are rendered as Markdown; preserve numeric values exactly as given.

When no document context is provided, answer from general knowledge and say \
so when you are uncertain. Always respond in the language of the user's \
question. Be clear, concise, and direct.

Current date (Jalali): {jalali_date} | Gregorian: {gregorian_date}
"""


class RAGEngine:
    """Async retrieval and chat pipeline."""

    def __init__(self) -> None:
        self.history_manager = QdrantChatHistory()
        self.language_model = LanguageModel()
        self.query_classifier = QueryClassifier()
        self.session_locks = SessionLocks()
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"RAGEngine(initialized={self._initialized})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            await self.history_manager.initialize()
            await self.language_model.initialize()
            self._initialized = True

    async def close(self) -> None:
        try:
            await self.language_model.close()
        finally:
            await self.history_manager.close()
            self._initialized = False

    async def __aenter__(self) -> "RAGEngine":
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RAGEngine is not initialized; call initialize() first"
            )

    @staticmethod
    def _build_system_prompt() -> str:
        now = datetime.now()
        jalali = jdatetime.date.fromgregorian(date=now.date()).strftime("%Y/%m/%d")
        gregorian = now.strftime("%Y-%m-%d")
        return _SYSTEM_PROMPT_TEMPLATE.format(
            jalali_date=jalali,
            gregorian_date=gregorian,
        )

    @staticmethod
    def _select_context(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep retrieved chunks within the character budget, dropping
        exact duplicates. Always keeps at least one chunk."""
        budget = settings.max_context_chars
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        used = 0
        for chunk in contexts:
            content = chunk.get("content", "")
            if not content or content in seen:
                continue
            if selected and used + len(content) > budget:
                break
            seen.add(content)
            selected.append(chunk)
            used += len(content)
        return selected

    def _build_messages(
        self,
        user_message: str,
        chat_history: list[dict[str, str]],
        document: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, str]]:
        current_content = user_message
        if document:
            document_context = "\n\n".join(
                f"### منبع: {RAGEngine._context_label(chunk)}\n{chunk.get('content', '')}"
                for chunk in document
            )
            current_content = (
                f"User Query: {user_message}\n\n"
                f"---\n"
                f"Document Results:\n{document_context}\n"
                f"---\n\n"
                f"Please provide a helpful response based on the user's query and the "
                f"context above. If the context doesn't contain relevant information, "
                f"say so honestly."
            )
        return [
            {"role": "system", "content": self._build_system_prompt()},
            *chat_history,
            {"role": "user", "content": current_content},
        ]

    @staticmethod
    def _context_label(chunk: dict[str, Any]) -> str:
        name = chunk.get("file_name") or chunk.get("source") or "document"
        page = chunk.get("page_number")
        return f"{name} - صفحه {page}" if isinstance(page, int) else str(name)

    async def _prepare_messages(
        self,
        query: str,
        session_id: str,
        top_k: int,
        doc_ids: Optional[list[str]] = None,
        include_tables: bool = True,
        rerank: bool = True,
        evaluation_mode: str = "diagnostics",
        relevant_chunk_ids: Optional[list[str]] = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        # Embed the query once and reuse it for both document and history search.
        query_vector = await self.history_manager.embed_query(query)
        history, has_documents = await asyncio.gather(
            self.history_manager.get_formatted_history_for_context(
                session_id, query, query_vector=query_vector
            ),
            self.history_manager.has_uploaded_documents(session_id)
        )
        contexts: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []
        if has_documents:
            parsed  = await self.query_classifier.parse(query)
            retrieved = await self.history_manager.search_documents(
                query=query,
                session_id=session_id,
                n_results=top_k,
                doc_ids=doc_ids,
                include_tables=include_tables,
                use_reranker=rerank,
                query_vector=query_vector,
                parsed_query=parsed
            )
            logger.debug(
                "Retrieved %d document chunks for session %s", len(retrieved), session_id
            )
            if not retrieved:
                logger.warning(
                    "Session %s has indexed documents but the search returned "
                    "nothing for query %r", session_id, query[:200],
                )
            contexts = self._select_context(retrieved)
        evaluation = await evaluate_contexts(
            query, retrieved, contexts, top_k,
            mode=evaluation_mode,
            relevant_chunk_ids=relevant_chunk_ids,
            language_model=self.language_model,
        )
        metadata: dict[str, Any] = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_search_performed": has_documents,
            "context_found": len(contexts),
            "context_used": contexts,
            "context_evaluation": evaluation.model_dump() if evaluation is not None else None,
        }
        return self._build_messages(query, history, contexts), metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer_question(
        self,
        query: str,
        session_id: str,
        llm_provider: Optional[str] = None,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
        include_tables: bool = True,
        rerank: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        evaluation_mode: str = "diagnostics",
        relevant_chunk_ids: Optional[list[str]] = None,
        evaluate_faithfulness: bool = False,
    ) -> dict[str, Any]:
        self._require_initialized()
        if llm_provider is not None and llm_provider != settings.llm_provider:
            raise ValueError(
                f"This server is configured for the {settings.llm_provider!r} provider"
            )
        start = time.perf_counter()
        async with self.session_locks.acquire(session_id):
            messages, metadata = await self._prepare_messages(
                query, session_id, top_k, doc_ids, include_tables, rerank,
                evaluation_mode, relevant_chunk_ids,
            )
            result = await self.language_model.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            await self.history_manager.add_conversation(
                query, result["response"], session_id
            )
            faithfulness = await evaluate_answer_faithfulness(
                query, result["response"], metadata["context_used"],
                enabled=evaluate_faithfulness, language_model=self.language_model,
            )
            metadata["faithfulness_evaluation"] = (
                faithfulness.model_dump() if faithfulness is not None else None
            )
        return {
            **result,
            "metadata": metadata,
            "processing_time": time.perf_counter() - start,
        }

    async def answer_question_stream(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
        include_tables: bool = True,
        rerank: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        evaluation_mode: str = "diagnostics",
        relevant_chunk_ids: Optional[list[str]] = None,
        evaluate_faithfulness: bool = False,
    ) -> AsyncIterator[str | dict[str, Any]]:
        self._require_initialized()
        start = time.perf_counter()
        async with self.session_locks.acquire(session_id):
            messages, metadata = await self._prepare_messages(
                query, session_id, top_k, doc_ids, include_tables, rerank,
                evaluation_mode, relevant_chunk_ids,
            )
            parts: list[str] = []
            usage: dict[str, int] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }
            saved = False
            try:
                async with aclosing(
                    self.language_model.stream(
                        messages, temperature=temperature, max_tokens=max_tokens
                    )
                ) as stream:
                    async for chunk in stream:
                        if isinstance(chunk, dict):
                            usage = chunk
                        else:
                            parts.append(chunk)
                            yield chunk
                full_response = "".join(parts)
                await self.history_manager.add_conversation(
                    query, full_response, session_id
                )
                saved = True
                faithfulness = await evaluate_answer_faithfulness(
                    query, full_response, metadata["context_used"],
                    enabled=evaluate_faithfulness, language_model=self.language_model,
                )
                metadata["faithfulness_evaluation"] = (
                    faithfulness.model_dump() if faithfulness is not None else None
                )
            finally:
                if not saved and parts:
                    # Client disconnected or the stream failed mid-answer:
                    # persist whatever was generated, marked as partial.
                    partial_save = asyncio.ensure_future(
                        self.history_manager.add_conversation(
                            query, "".join(parts), session_id, partial=True
                        )
                    )
                    try:
                        await asyncio.shield(partial_save)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "Failed to persist partial answer for session %s",
                            session_id,
                        )
            # single terminal event — consumers check `type == "done"`
            yield {
                "type": "done",
                "full_response": "".join(parts),
                "sources": metadata["context_used"],
                "context_evaluation": metadata["context_evaluation"],
                "faithfulness_evaluation": metadata["faithfulness_evaluation"],
                "usage": {
                    "prompt_tokens": usage["input_tokens"],
                    "completion_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage["reasoning_tokens"],
                },
                "processing_time": time.perf_counter() - start,
            }

    async def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return await self.history_manager.get_recent_messages(session_id)

    async def delete_session(self, session_id: str) -> bool:
        async with self.session_locks.acquire(session_id):
            return await self.history_manager.delete_session(session_id)

    async def get_session_stats(self, session_id: str) -> dict[str, Any]:
        return await self.history_manager.get_session_stats(session_id)

    async def get_all_sessions(self) -> list[dict[str, Any]]:
        return await self.history_manager.list_sessions()

    async def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        self._require_initialized()
        return await self.history_manager.list_documents(session_id)

    async def delete_document(self, session_id: str, doc_id: str) -> bool:
        self._require_initialized()
        async with self.session_locks.acquire(session_id):
            return await self.history_manager.delete_document(session_id, doc_id)

    async def component_health(self) -> dict[str, bool]:
        """Readiness of backing components, for the /health endpoint."""
        components = {
            "llm": self._initialized and self.language_model._initialized,
            "embeddings": (
                self._initialized
                and self.history_manager.embeddings.dimension is not None
            ),
            "qdrant": False,
        }
        if self._initialized:
            try:
                await self.history_manager.store.client.get_collections()
                components["qdrant"] = True
            except Exception:
                logger.warning("Qdrant health check failed", exc_info=True)
        return components
