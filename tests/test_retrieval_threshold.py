"""Regression tests for document retrieval at realistic embedding similarities.

OpenAI text-embedding-3-* models score query<->chunk cosine similarity well
below 0.5 even for relevant text. The retrieval path used to hard-code a
0.5 score threshold, which silently emptied results while documents were
indexed. These tests pin the behavior that such similarity levels still
retrieve chunks.
"""

import math

import pytest
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.db.chat_history import QdrantChatHistory

QUERY = "مبلغ قرارداد ۱۲۳ چقدر است؟"
CHUNK_A = "مبلغ کل قرارداد ۱۲۳ برابر است با ۵۰۰ میلیون تومان و پرداخت آن در سه قسط انجام می‌شود."
CHUNK_B = "مدت زمان اجرای پروژه هجده ماه از تاریخ ابلاغ است و تمدید آن نیاز به توافق کتبی دارد."


class LowSimilarityEmbedder:
    """Deterministic embedder simulating the text-embedding-3 similarity regime.

    Every unique text gets its own orthogonal direction; the query is a fixed
    mix with `similarity` cosine toward CHUNK_A and the rest noise, so the
    best dense score stays below any meaningful cutoff.
    """

    dimension = 8
    reranker = None

    def __init__(self, similarity: float = 0.45):
        self.similarity = similarity
        self._directions: dict[str, list[float]] = {}
        self._noise = self._next_direction("noise")

    def _next_direction(self, name: str) -> list[float]:
        if name not in self._directions:
            vector = [0.0] * self.dimension
            vector[len(self._directions) % self.dimension] = 1.0
            self._directions[name] = vector
        return self._directions[name]

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(component ** 2 for component in vector)) or 1.0
        return [component / norm for component in vector]

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        output = []
        for text in texts:
            if text == QUERY:
                direction = self._next_direction(CHUNK_A)
                query = [self.similarity * component for component in direction]
                noise_scale = math.sqrt(max(0.0, 1.0 - self.similarity ** 2))
                query = [
                    component + noise_scale * noise
                    for component, noise in zip(query, self._noise)
                ]
                output.append(self._normalize(query))
            else:
                output.append(list(self._next_direction(text)))
        return output


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_low_similarity_query_still_retrieves(monkeypatch):
    monkeypatch.setattr(settings, "search_score_threshold", 0.0)
    history = QdrantChatHistory(AsyncQdrantClient(":memory:"), LowSimilarityEmbedder())
    try:
        await history.initialize()
        await history.add_document_chunks(
            [
                Document(CHUNK_A, {"source": "report.pdf", "doc_id": "d1"}),
                Document(CHUNK_B, {"source": "report.pdf", "doc_id": "d1"}),
            ],
            "session",
        )
        query_vector = await history.embed_query(QUERY)
        matches = await history.search_documents(
            QUERY, "session", n_results=5, query_vector=query_vector
        )
        assert matches, "relevant chunk below 0.5 cosine must still be retrieved"
        assert matches[0]["content"] == CHUNK_A
    finally:
        await history.close()


@pytest.mark.anyio
async def test_explicit_threshold_still_filters(monkeypatch):
    monkeypatch.setattr(settings, "search_score_threshold", 0.0)
    history = QdrantChatHistory(AsyncQdrantClient(":memory:"), LowSimilarityEmbedder())
    try:
        await history.initialize()
        await history.add_document_chunks(
            [Document(CHUNK_A, {"source": "report.pdf", "doc_id": "d1"})],
            "session",
        )
        query_vector = await history.embed_query(QUERY)
        matches = await history.search_documents(
            QUERY, "session", n_results=5, score_threshold=0.99,
            query_vector=query_vector,
        )
        assert matches == []
    finally:
        await history.close()
