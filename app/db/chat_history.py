import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.concurrency import SessionLocks
from app.db.vector_store import QdrantVectorStore
from app.llm.embeddings import EmbeddingService
from app.llm.query_classifier import ParsedQuery  # adjust import path


class QdrantChatHistory:
    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        embeddings: EmbeddingService | None = None,
    ):
        self.store = QdrantVectorStore(client)
        self.embeddings = embeddings if embeddings is not None else EmbeddingService()
        self.messages_collection_name = settings.qdrant_messages_collection
        self.documents_collection_name = settings.qdrant_documents_collection
        self._session_locks = SessionLocks()
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            await self.embeddings.initialize()
            await self.store.initialize(self.embeddings.dimension)
            self._initialized = True

    async def close(self) -> None:
        try:
            await self.store.close()
        finally:
            await self.embeddings.close()
            self._initialized = False  # allow re-initialization after close

    async def __aenter__(self) -> "QdrantChatHistory":
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"QdrantChatHistory("
            f"initialized={self._initialized}, "
            f"messages_col={self.messages_collection_name!r}, "
            f"documents_col={self.documents_collection_name!r})"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def _add_messages(self, messages: list[dict[str, Any]], session_id: str) -> None:
        await self.initialize()
        vectors = await self.embeddings.embed([message["content"] for message in messages])
        async with self._session_locks.acquire(session_id):
            next_index = 1 + await self.store.max_int_field(
                self.messages_collection_name, {"session_id": session_id}
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            points = [
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        **message,
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "message_index": next_index + index,
                    },
                )
                for index, (message, vector) in enumerate(zip(messages, vectors))
            ]
            await self.store.upsert(self.messages_collection_name, points)

    async def add_message(self, role: str, content: str, session_id: str) -> None:
        await self._add_messages([{"role": role, "content": content}], session_id)

    async def add_conversation(
        self,
        user_message: str,
        assistant_message: str,
        session_id: str,
        partial: bool = False,
    ) -> None:
        """Store a user/assistant turn.

        `partial=True` marks the assistant message as cut short (e.g. the
        streaming client disconnected mid-answer).
        """
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message,
        }
        if partial:
            assistant["partial"] = True
        await self._add_messages([
            {"role": "user", "content": user_message},
            assistant,
        ], session_id)

    async def add_document_chunks(
        self,
        document_chunks: list[Document],
        session_id: str,
        doc_type: str = "text",
    ) -> list[str]:
        await self.initialize()
        if not document_chunks:
            return []
        vectors = await self.embeddings.embed([chunk.page_content for chunk in document_chunks])
        group_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "session_id": session_id,
                    "content": chunk.page_content,
                    "doc_group_id": group_id,
                    "doc_type": doc_type,
                    "chunk_index": index,
                    "total_chunks": len(document_chunks),
                    "timestamp": timestamp,
                    "char_count": len(chunk.page_content),
                },
            )
            for index, (chunk, vector) in enumerate(zip(document_chunks, vectors))
        ]
        async with self._session_locks.acquire(session_id):
            await self.store.upsert(self.documents_collection_name, points)
        return [str(point.id) for point in points]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query once; pass the vector to the search methods."""
        await self.initialize()
        vectors = await self.embeddings.embed(query)
        return vectors[0]

    async def has_uploaded_documents(self, session_id: str) -> bool:
        await self.initialize()
        return await self.store.exists(
            self.documents_collection_name, {"session_id": session_id}
        )

    async def search_documents(
        self,
        query: str,
        session_id: str,
        n_results: int = 5,
        score_threshold: float | None = None,
        source: str | None = None,
        doc_ids: list[str] | None = None,
        include_tables: bool = True,
        use_reranker: bool = True,
        query_vector: list[float] | None = None,
        parsed_query: ParsedQuery | None = None,   # <-- new
    ) -> list[dict[str, Any]]:
        await self.initialize()
        if n_results <= 0:
            return []
        if score_threshold is None:
            score_threshold = settings.search_score_threshold

        # --- base identity filters (unchanged) ---
        filters: dict[str, Any] = {"session_id": session_id}
        if source is not None:
            filters["source"] = source
        if doc_ids:
            filters["doc_id"] = list(doc_ids)

        # --- unpack ParsedQuery ---
        embed_query = query
        section_hint: str | None = None
        prefer_tables = False
        prefer_text = False

        if parsed_query is not None:
            embed_query = parsed_query.semantic_query or query
            meta = dict(parsed_query.filters)          # work on a copy
            section_hint = meta.pop("section_hint", None)
            prefer_tables = parsed_query.prefer_tables
            prefer_text   = parsed_query.prefer_text

            # content_type from classifier overrides the include_tables flag
            if "content_type" in meta:
                filters["content_type"] = meta.pop("content_type")

            # exact metadata filters — page_number, sheet_name, file_name
            filters.update(meta)

        # legacy include_tables flag (only applies when classifier didn't set content_type)
        if not include_tables and "content_type" not in filters:
            filters["content_type"] = "text"

        # --- embedding ---
        vector = query_vector
        if vector is None:
            vector = (await self.embeddings.embed(embed_query))[0]

        # fetch extra candidates when we need post-filtering or score boosting
        needs_post = bool(section_hint or prefer_tables or prefer_text)
        rerank = use_reranker and self.embeddings.reranker is not None
        fetch_limit = max(n_results, settings.rerank_top_k) if rerank else n_results
        if needs_post:
            fetch_limit = max(fetch_limit, n_results * 3)

        points = await self.store.search_documents(
            embed_query,
            vector,
            filters,
            fetch_limit,
            score_threshold,
            exclude=None,           # content_type is now in filters, not exclude
        )
        matches = [
            {**(point.payload or {}), "id": str(point.id), "score": float(point.score)}
            for point in points
        ]

        # --- section_hint: soft boost on hierarchy_path (fuzzy, not a hard gate) ---
        if section_hint:
            hint = section_hint.lower()
            for m in matches:
                if hint in (m.get("hierarchy_path") or "").lower():
                    m["score"] *= 1.25

        # --- prefer_tables / prefer_text: soft score boost ---
        if prefer_tables or prefer_text:
            target = "table" if prefer_tables else "text"
            for m in matches:
                if m.get("content_type") == target:
                    m["score"] *= 1.3

        # re-sort after any adjustments before reranking
        matches.sort(key=lambda m: m["score"], reverse=True)

        if rerank:
            # use original raw query for reranker — it has more context than semantic_query
            matches = await self.embeddings.rerank(query, matches)

        return matches[:n_results]

    @staticmethod
    def _message(point) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "role": payload.get("role", "user"),
            "content": payload.get("content", ""),
            "timestamp": payload.get("timestamp"),
            "index": payload.get("message_index", 0),
            "partial": bool(payload.get("partial", False)),
        }

    @staticmethod
    def _message_order(message: dict[str, Any]):
        return message["index"], message.get("timestamp") or "", message["id"]

    async def search(
        self,
        query: str,
        session_id: str,
        limit: int = 5,
        score_threshold: float | None = None,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        if score_threshold is None:
            score_threshold = settings.search_score_threshold
        vector = query_vector
        if vector is None:
            vector = (await self.embeddings.embed(query))[0]
        points = await self.store.search(
            self.messages_collection_name,
            vector,
            {"session_id": session_id},
            limit,
            score_threshold,
        )
        messages = [{**self._message(point), "score": float(point.score)} for point in points]
        return sorted(messages, key=self._message_order)

    async def get_recent_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        await self.initialize()
        if limit is not None and limit > 0:
            # Newest-first server-side ordering via the integer payload index
            points = await self.store.scroll_recent(
                self.messages_collection_name, {"session_id": session_id}, limit
            )
            messages = [self._message(point) for point in reversed(points)]
        else:
            messages = [
                self._message(point)
                async for point in self.store.iter_scroll(
                    self.messages_collection_name, {"session_id": session_id}
                )
            ]
        return sorted(messages, key=self._message_order)

    async def get_formatted_history_for_context(
        self,
        session_id: str,
        user_message: str,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, str]]:
        await self.initialize()
        recent, relevant = await asyncio.gather(
            self.get_recent_messages(session_id, limit=6),
            self.search(user_message, session_id, query_vector=query_vector),
        )
        combined = {message["id"]: message for message in recent + relevant}
        return [
            {"role": message["role"], "content": message["content"]}
            for message in sorted(combined.values(), key=self._message_order)
        ]

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> bool:
        await self.initialize()
        session_filter = {"session_id": session_id}
        async with self._session_locks.acquire(session_id):
            exists = await asyncio.gather(
                self.store.exists(self.messages_collection_name, session_filter),
                self.store.exists(self.documents_collection_name, session_filter),
            )
            if not any(exists):
                return False
            await asyncio.gather(*(
                self.store.delete(collection, session_filter)
                for collection in (self.messages_collection_name, self.documents_collection_name)
            ))
            return True

    async def get_session_stats(self, session_id: str) -> dict[str, Any]:
        await self.initialize()
        message_count = 0
        user_messages = 0
        assistant_messages = 0
        total_characters = 0
        timestamps: list[str] = []
        async for point in self.store.iter_scroll(
            self.messages_collection_name, {"session_id": session_id}
        ):
            payload = point.payload or {}
            role = payload.get("role", "user")
            message_count += 1
            user_messages += role == "user"
            assistant_messages += role == "assistant"
            total_characters += len(payload.get("content", ""))
            if payload.get("timestamp"):
                timestamps.append(payload["timestamp"])
        timestamps.sort()
        return {
            "session_id": session_id,
            "exists": message_count > 0,
            "message_count": message_count,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_characters": total_characters,
            "first_message_at": timestamps[0] if timestamps else None,
            "last_message_at": timestamps[-1] if timestamps else None,
        }

    async def list_sessions(self) -> list[dict[str, Any]]:
        await self.initialize()
        sessions: dict[str, dict[str, Any]] = {}
        async for point in self.store.iter_scroll(self.messages_collection_name):
            payload = point.payload or {}
            session_id = payload.get("session_id")
            if not session_id:
                continue
            timestamp = payload.get("timestamp") or ""
            session = sessions.setdefault(session_id, {
                "session_id": session_id,
                "message_count": 0,
                "first_message": timestamp,
                "last_message": timestamp,
            })
            session["message_count"] += 1
            session["first_message"] = min(session["first_message"], timestamp)
            session["last_message"] = max(session["last_message"], timestamp)
        return sorted(sessions.values(), key=lambda session: session["last_message"], reverse=True)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    async def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        """Group the session's indexed chunks by document."""
        await self.initialize()
        documents: dict[str, dict[str, Any]] = {}
        async for point in self.store.iter_scroll(
            self.documents_collection_name, {"session_id": session_id}
        ):
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            entry = documents.setdefault(doc_id, {
                "doc_id": doc_id,
                "file_name": payload.get("file_name") or payload.get("source"),
                "doc_type": payload.get("doc_type"),
                "chunk_count": 0,
                "page_count": 0,
                "upload_time": payload.get("timestamp"),
            })
            entry["chunk_count"] += 1
            page = payload.get("page_number")
            if isinstance(page, int):
                entry["page_count"] = max(entry["page_count"], page)
        return list(documents.values())

    async def delete_document(self, session_id: str, doc_id: str) -> bool:
        await self.initialize()
        filters = {"session_id": session_id, "doc_id": doc_id}
        async with self._session_locks.acquire(session_id):
            if not await self.store.exists(self.documents_collection_name, filters):
                return False
            await self.store.delete(self.documents_collection_name, filters)
            return True
