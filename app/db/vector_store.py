import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.retrieval.bm25 import BM25Encoder, BM25_VECTOR_NAME

logger = logging.getLogger(__name__)


_SCROLL_PAGE_SIZE = 256


class QdrantVectorStore:
    def __init__(self, client: Optional[AsyncQdrantClient] = None):
        self.client = client if client is not None else AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
            check_compatibility=False,
        )
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._documents_have_bm25 = False
        self.bm25 = BM25Encoder(
            settings.bm25_k1, settings.bm25_b, settings.bm25_average_length
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, dimension: int) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return

            messages_col = settings.qdrant_messages_collection
            documents_col = settings.qdrant_documents_collection

            if messages_col == documents_col:
                raise ValueError(
                    "Messages and documents require different Qdrant collections"
                )

            results = await asyncio.gather(
                self._ensure_collection(messages_col, dimension, is_documents=False),
                self._ensure_collection(documents_col, dimension, is_documents=True),
                return_exceptions=True,
            )

            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                for err in errors:
                    logger.error("Collection setup failed: %s", err)
                raise errors[0]

            self._initialized = True

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "QdrantVectorStore":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"QdrantVectorStore("
            f"initialized={self._initialized}, "
            f"bm25={self._documents_have_bm25})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "QdrantVectorStore.initialize() must be awaited before use"
            )

    async def _ensure_collection(
        self, collection: str, dimension: int, *, is_documents: bool
    ) -> None:
        if not await self.client.collection_exists(collection):
            try:
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        BM25_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    } if is_documents else None,
                )
            except UnexpectedResponse:
                # Another process may have created it concurrently; verify.
                if not await self.client.collection_exists(collection):
                    raise

        info = await self.client.get_collection(collection)
        vectors = info.config.params.vectors
        if (
            not isinstance(vectors, models.VectorParams)
            or vectors.size != dimension
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError(
                f"Qdrant collection {collection!r} must use {dimension}-dimensional "
                "cosine vectors. Use a new collection name when changing embedding models."
            )

        if is_documents:
            sparse_vectors = info.config.params.sparse_vectors or {}
            bm25_config = sparse_vectors.get(BM25_VECTOR_NAME)
            self._documents_have_bm25 = (
                bm25_config is not None
                and bm25_config.modifier == models.Modifier.IDF
            )
            if settings.hybrid_search_enabled and not self._documents_have_bm25:
                raise ValueError(
                    f"Qdrant collection {collection!r} needs a '{BM25_VECTOR_NAME}' "
                    "sparse vector with the IDF modifier for hybrid search. "
                    "Set QDRANT_DOCUMENTS_COLLECTION to a new name and re-upload "
                    "documents, or set HYBRID_SEARCH_ENABLED=false to keep dense-only "
                    "retrieval. Existing data is not deleted."
                )

        # Payload indexes: session_id on all collections; doc fields only on
        # documents; message ordering on messages. create_payload_index is
        # idempotent, so this also upgrades existing collections.
        index_fields: list[tuple[str, models.PayloadSchemaType]] = [
            ("session_id", models.PayloadSchemaType.KEYWORD),
        ]
        if is_documents:
            index_fields += [
                ("source", models.PayloadSchemaType.KEYWORD),
                ("doc_id", models.PayloadSchemaType.KEYWORD),
            ]
        else:
            index_fields.append(
                ("message_index", models.PayloadSchemaType.INTEGER)
            )

        for field_name, schema in index_fields:
            await self.client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )

    @staticmethod
    def make_filter(
        values: dict[str, Any],
        exclude: Optional[dict[str, Any]] = None,
    ) -> models.Filter:
        """Build a Qdrant filter from must-match and must-secret-a5a1a289-match pairs.

        A list value becomes a MatchAny (any of the values). Passing empty
        dicts produces a filter with no conditions, which matches every
        point. Callers should ensure values is non-empty when a selective
        filter is intended.
        """
        must = []
        for key, value in values.items():
            if isinstance(value, (list, tuple, set)):
                match: models.BaseMatch = models.MatchAny(any=list(value))
            else:
                match = models.MatchValue(value=value)
            must.append(models.FieldCondition(key=key, match=match))
        must_not = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (exclude or {}).items()
        ]
        return models.Filter(must=must or None, must_not=must_not or None)

    @staticmethod
    def _sanitize_filters(filters: dict[str, Any]) -> dict[str, Any]:
        """Strip None, empty-string, or placeholder 'string' values from a filters dict."""
        _PLACEHOLDERS = {None, "", "string"}
        cleaned: dict[str, Any] = {}
        for k, v in (filters or {}).items():
            if isinstance(v, list):
                if not v or all(i == "string" for i in v):
                    continue
            elif v in _PLACEHOLDERS:
                continue
            cleaned[k] = v
        return cleaned

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(
        self, collection: str, points: list[models.PointStruct]
    ) -> None:
        self._require_initialized()
        for start in range(0, len(points), settings.qdrant_batch_size):
            batch = points[start:start + settings.qdrant_batch_size]
            if (
                self._documents_have_bm25
                and collection == settings.qdrant_documents_collection
            ):
                batch = await asyncio.to_thread(self._with_sparse_vectors, batch)
            await self.client.upsert(
                collection_name=collection,
                points=batch,
                wait=True,
            )

    def _with_sparse_vectors(
        self, points: list[models.PointStruct]
    ) -> list[models.PointStruct]:
        result = []
        for point in points:
            # point.vector may already be a named-vector dict; extract the
            # dense part under the unnamed ("") key, or use the raw list.
            existing = point.vector
            if isinstance(existing, dict):
                dense = existing.get("", existing)
            else:
                dense = existing

            sparse = self.bm25.encode_document(
                (point.payload or {}).get("content", "")
            )
            result.append(point.model_copy(update={"vector": {
                "": dense,
                BM25_VECTOR_NAME: sparse,
            }}))
        return result

    async def delete(self, collection: str, filters: dict[str, Any]) -> None:
        await self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=self.make_filter(filters)),
            wait=True,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search_documents(
        self,
        query: str,
        vector: list[float],
        filters: dict[str, Any],
        limit: int,
        score_threshold: float = 0.0,
        exclude: Optional[dict[str, Any]] = None,
    ) -> list[models.ScoredPoint]:
        self._require_initialized()
        collection = settings.qdrant_documents_collection
        if limit <= 0:
            return []

        # Drop placeholder / unset filter values before they reach make_filter.
        filters = self._sanitize_filters(filters)

        if settings.hybrid_search_enabled:
            sparse = await asyncio.to_thread(self.bm25.encode_query, query)
        else:
            sparse = None
        # print('sparse: ', sparse)
        if sparse is None or not sparse.indices:
            return await self.search(
                collection, vector, filters, limit, score_threshold, exclude
            )

        print('exclude: ', exclude)

        query_filter = self.make_filter(filters, exclude)
        print("query_filter: ", query_filter)
        # print("query: ", query)
        # print("vector: ", vector)
        prefetch_limit = max(limit, settings.hybrid_prefetch_limit)
        result = await self.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=vector,
                    filter=query_filter,
                    limit=prefetch_limit,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                ),
                models.Prefetch(
                    query=sparse,
                    using=BM25_VECTOR_NAME,
                    filter=query_filter,
                    limit=prefetch_limit,
                    # no score_threshold on sparse — BM25 scores aren't comparable
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            # filter applied per-prefetch; omitting here avoids double-filtering
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        print("vector_store_result: ", result.points)
        return result.points

    async def search(
        self,
        collection: str,
        vector: list[float],
        filters: dict[str, Any],
        limit: int,
        score_threshold: float = 0.0,
        exclude: Optional[dict[str, Any]] = None,
    ) -> list[models.ScoredPoint]:
        self._require_initialized()
        # Drop placeholder / unset filter values before they reach make_filter.
        filters = self._sanitize_filters(filters)
        result = await self.client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=self.make_filter(filters, exclude),
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
            with_vectors=False,
        )
        return result.points

    async def iter_scroll(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        page_size: int = _SCROLL_PAGE_SIZE,
    ) -> AsyncIterator[models.Record]:
        """Yield records page by page so callers can stream-process them."""
        offset = None
        while True:
            page, offset = await self.client.scroll(
                collection_name=collection,
                scroll_filter=self.make_filter(filters) if filters else None,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in page:
                yield record
            if offset is None:
                return

    async def scroll_recent(
        self,
        collection: str,
        filters: dict[str, Any],
        limit: int,
        order_field: str = "message_index",
    ) -> list[models.Record]:
        """Newest `limit` records by an integer payload field, descending.

        Falls back to a full scan when the server cannot order by the field
        (e.g. an older collection without the integer index).
        """
        try:
            records, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=self.make_filter(filters),
                limit=limit,
                order_by=models.OrderBy(
                    key=order_field, direction=models.Direction.DESC
                ),
                with_payload=True,
                with_vectors=False,
            )
            return records
        except Exception:
            logger.debug(
                "order_by scroll failed for %s; falling back to full scan",
                collection,
                exc_info=True,
            )
            records = [record async for record in self.iter_scroll(collection, filters)]
            records.sort(
                key=lambda r: (r.payload or {}).get(order_field, -1), reverse=True
            )
            return records[:limit]

    async def max_int_field(
        self,
        collection: str,
        filters: dict[str, Any],
        field: str = "message_index",
    ) -> int:
        """Highest value of an integer payload field, or -1 if none exist."""
        try:
            records, _ = await self.client.scroll(
                collection_name=collection,
                scroll_filter=self.make_filter(filters),
                limit=1,
                order_by=models.OrderBy(key=field, direction=models.Direction.DESC),
                with_payload=True,
                with_vectors=False,
            )
            return max(
                (int((r.payload or {}).get(field, -1)) for r in records),
                default=-1,
            )
        except Exception:
            logger.debug(
                "order_by scroll failed for %s; falling back to full scan",
                collection,
                exc_info=True,
            )
            best = -1
            async for record in self.iter_scroll(collection, filters):
                value = (record.payload or {}).get(field, -1)
                if isinstance(value, int):
                    best = max(best, value)
            return best

    async def exists(self, collection: str, filters: dict[str, Any]) -> bool:
        self._require_initialized()
        records, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=self.make_filter(filters),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    # ------------------------------------------------------------------
    # Document presence helpers
    # ------------------------------------------------------------------

    async def document_exists(
        self,
        doc_id: str,
        session_id: str | None = None,
    ) -> bool:
        """Return True if at least one chunk for *doc_id* is stored.

        Optionally scope the check to a *session_id* when documents are
        tenant-isolated. A cheap limit=1 scroll is used — Qdrant stops
        after the first matching point.
        """
        self._require_initialized()
        filters: dict[str, Any] = {"doc_id": doc_id}
        if session_id is not None:
            filters["session_id"] = session_id
        records, _ = await self.client.scroll(
            collection_name=settings.qdrant_documents_collection,
            scroll_filter=self.make_filter(filters),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    async def document_chunk_count(
        self,
        doc_id: str,
        session_id: str | None = None,
    ) -> int:
        """Return the number of stored chunks for *doc_id*.

        Useful for upload verification and progress checks. Iterates all
        matching points via scroll pages, so avoid calling this in hot paths.
        """
        self._require_initialized()
        filters: dict[str, Any] = {"doc_id": doc_id}
        if session_id is not None:
            filters["session_id"] = session_id
        count = 0
        async for _ in self.iter_scroll(
            settings.qdrant_documents_collection, filters
        ):
            count += 1
        return count

    async def session_has_documents(self, session_id: str) -> bool:
        """True if the session has at least one stored document chunk."""
        self._require_initialized()
        records, _ = await self.client.scroll(
            collection_name=settings.qdrant_documents_collection,
            scroll_filter=self.make_filter({"session_id": session_id}),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    async def list_session_doc_ids(self, session_id: str) -> list[str]:
        """Distinct doc_ids stored under *session_id*, sorted alphabetically.

        Performs a full scroll — use only for diagnostics / admin endpoints,
        not in hot paths.
        """
        self._require_initialized()
        doc_ids: set[str] = set()
        async for record in self.iter_scroll(
            settings.qdrant_documents_collection,
            {"session_id": session_id},
        ):
            doc_id = (record.payload or {}).get("doc_id")
            if doc_id is not None:
                doc_ids.add(str(doc_id))
        return sorted(doc_ids)

    async def session_chunk_count(self, session_id: str) -> int:
        """Total stored chunks across all documents in *session_id*.

        Full scroll — diagnostics only.
        """
        self._require_initialized()
        count = 0
        async for _ in self.iter_scroll(
            settings.qdrant_documents_collection,
            {"session_id": session_id},
        ):
            count += 1
        return count
