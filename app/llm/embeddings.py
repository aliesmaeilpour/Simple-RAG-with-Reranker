import asyncio
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.client: AsyncOpenAI | None = None
        self.model = None
        self.reranker = None
        self.dimension: int | None = None
        self._initialize_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self.dimension is not None:
                return
            if settings.resolved_embedding_provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required for the OpenAI embedding provider"
                    )
                self.client = AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    timeout=settings.openai_timeout,
                    max_retries=settings.openai_max_retries,
                )
                try:
                    vectors = await self.embed(["embedding dimension"])
                    dimension = len(vectors[0])
                except Exception:
                    await self.client.close()
                    self.client = None
                    raise
            else:
                await asyncio.to_thread(self._load_local_model)
                dimension = self.model.get_sentence_embedding_dimension()
            if settings.use_reranking:
                await asyncio.to_thread(self._load_reranker)
            self.dimension = dimension

    async def close(self) -> None:
        try:
            if self.client is not None:
                await self.client.close()
        finally:
            self.client = None
            self.model = None
            self.reranker = None
            self.dimension = None  # allow re-initialization after close

    async def __aenter__(self) -> "EmbeddingService":
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"EmbeddingService("
            f"provider={settings.resolved_embedding_provider!r}, "
            f"dimension={self.dimension}, "
            f"reranker={'loaded' if self.reranker is not None else 'none'})"
        )

    # ------------------------------------------------------------------
    # Local model loaders (run in a thread)
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)

    def _load_reranker(self) -> None:
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(settings.rerank_model, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    async def embed(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        if settings.resolved_embedding_provider != "openai":
            if self.model is None:
                raise RuntimeError("EmbeddingService is not initialized; call initialize() first")
            vectors = await asyncio.to_thread(
                self.model.encode,
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vectors.tolist()

        if self.client is None:
            raise RuntimeError("EmbeddingService is not initialized; call initialize() first")

        options: dict[str, Any] = {}
        if settings.openai_embedding_dimensions is not None:
            options["dimensions"] = settings.openai_embedding_dimensions

        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[start : start + settings.embedding_batch_size]
            response = await self.client.embeddings.create(
                input=[text.replace("\n", " ") for text in batch],
                model=settings.openai_embedding_model,
                **options,
            )
            vectors.extend(
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            )
        return vectors

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    async def rerank(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.reranker is None or not results:
            return results

        scores = await asyncio.to_thread(
            self.reranker.predict,
            [(query, result["content"]) for result in results],
        )
        # work on copies so we don't mutate the caller's list
        ranked = [
            {**result, "rerank_score": float(score)}
            for result, score in zip(results, scores)
        ]
        return sorted(ranked, key=lambda r: r["rerank_score"], reverse=True)
