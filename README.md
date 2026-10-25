# Persian Retrieval-Augmented Generation (RAG) System

FastAPI application for Persian PDF, Word, and Excel retrieval, with OpenAI or local Qwen generation and an async Qdrant vector database.

## Project layout

```text
app/
  api/routes.py                 HTTP endpoints and dependencies
  core/config.py                Environment settings
  core/concurrency.py           Per-session async locks
  core/middleware.py            Rate limiting and request-id middleware
  db/vector_store.py            Async Qdrant collections, filters, and operations
  db/chat_history.py            Session messages and document retrieval
  llm/embeddings.py             Async embeddings and off-thread local reranking
  llm/provider.py               Async OpenAI and off-thread Qwen generation
  retrieval/bm25.py             Persian-aware sparse BM25 encoding
  retrieval/evaluation.py       Context diagnostics and relevance judgments
  retrieval/faithfulness.py     Answer claim extraction and source verification
  schemas/models.py             Request and response models
  service/document_processor.py PDF, DOCX, and XLSX parsing and chunking
  service/rag_engine.py         Retrieval and conversation orchestration
  main.py                       FastAPI lifespan and server
main.py                         Compatible server entry point
tests/                          Unit tests (pytest)
Dockerfile / compose.yaml       Container image and local stack
```

## Setup

Use Python 3.10 or newer and Docker Compose (or an existing Qdrant server).

```sh
pip install -r requirements.txt          # add -r requirements-dev.txt for tests
docker compose up -d qdrant
```

Copy `.env.example` to `.env` and adjust. Set `OPENAI_API_KEY` for the default OpenAI provider. An empty `QDRANT_API_KEY` works for the bundled local server; set the URL and key for Qdrant Cloud or another secured deployment.

```sh
python main.py
# Alternatively:
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Full stack (Qdrant + app):
docker compose up --build
```

Open `/docs` for interactive API documentation. `uvicorn main:app` remains supported.

For local generation, set `LLM_PROVIDER=qwen`. Embeddings can use a different backend than generation: `EMBEDDING_PROVIDER` accepts `openai` or `local` and otherwise follows `LLM_PROVIDER`. Local models load only when needed; set `USE_RERANKING=false` to skip the reranker. For Qwen's `device_map=auto`, install `accelerate` and a PyTorch build matching your hardware.

## Storage and concurrency

Uploaded files are parsed, chunked, embedded, and indexed into a per-session documents collection on upload — the upload response reports `doc_id` and `indexed_chunks`. `doc_id` is derived from the content hash plus the original filename, so re-uploading identical bytes under the same name produces the same ID (deduplication). PDF images are extracted (deduplicated by content) to `uploads/images/{doc_id}/` and counted in metadata, but not indexed — empty-content chunks would only pollute retrieval.

Messages and document chunks live in separate Qdrant collections with UUID point IDs and indexed session filters. The embedding dimension is determined during startup; existing collection dimensions and cosine distance are validated. Changing the embedding model requires new collection names and reindexing; use `OPENAI_EMBEDDING_DIMENSIONS` only with embedding models that support that option.

Database and OpenAI operations use native async clients with configured timeouts and retries (`OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`). File parsing, local embeddings, reranking, and Qwen inference run outside the event loop. The same query embedding is reused for document and history search. Turn ordering per session is enforced with process-local locks; message indices are assigned via an indexed `message_index` payload field instead of rescanning history.

Per-session locks and the rate limiter are process-local: with multiple workers or replicas (see `WORKERS`), use a single application worker or add distributed coordination (e.g. Redis) when strict per-session ordering or global rate limits are required. Existing ChromaDB/FAISS files are not imported; re-upload documents to index them.

## Retrieval quality

Retrieval combines dense semantic search and sparse BM25 in one Qdrant query using reciprocal rank fusion (RRF), with optional cross-encoder reranking afterwards. The retrieved chunks are returned as `sources` in the `/chat` response (and in the streaming `done` event) with file name, page number, score, and preview; the system prompt instructs the model to cite them inline. The document context sent to the model is bounded by `MAX_CONTEXT_CHARS` and deduplicated, so large `top_k` values cannot overflow the prompt. BM25 handles Persian/Arabic character variants, diacritics, half-spaces, and Persian/Arabic digits; `BM25_K1`, `BM25_B`, and `BM25_AVERAGE_LENGTH` control lexical weighting (changing these or the tokenizer requires reindexing).

### Context evaluation

`/chat`, `/query`, and the `/chat/stream` terminal `done` event return a
`context_evaluation` object. Evaluation uses the exact sources sent to generation,
after deduplication and the context budget. It does not change their ranking or
discard sources. Requests accept `evaluation_mode`:

- `diagnostics` (default): per-source query-term coverage, matched terms, raw
  retrieval/reranker scores, normalized duplicate counts among retrieved chunks,
  dropped-context counts, and image-table OCR flags. This runs locally without
  additional embedding or generation calls. Coverage uses Persian/Arabic and digit
  normalization and excludes common English/Persian function words. It measures
  lexical overlap, not semantic relevance or answer correctness.
- `llm`: adds one call to the configured generation model to judge each selected
  context. Scores are `0` (unrelated), `1` (on topic without answer evidence), `2`
  (partial evidence), and `3` (direct evidence), with a brief reason per source.
  `mean_relevance_score` uses this 0–3 scale; `judged_context_precision` is the share
  scoring at least 2. This adds latency and, for hosted models, token cost. Judge
  tokens are reported separately in `judge_usage`; answer token counts are unchanged.
  Failed or malformed judgments set `status: unavailable`, retain the diagnostics,
  and allow answer generation to continue. These judgments are estimates, not
  ground-truth recall or answer-faithfulness measurements.
- `off`: returns `context_evaluation: null`.

For reference-based retrieval evaluation, supply `relevant_chunk_ids` with the
known relevant IDs for that question, using `sources[].chunk_id` from the indexed
document version. `ranking_metrics` then reports binary precision@k, recall@k,
reciprocal rank, and nDCG@k over the selected sources. `k` is the requested `top_k`
(5 for `/query`); missing positions count as nonrelevant and repeated IDs earn
credit once. Duplicate labels are ignored. Without reference IDs, these metrics
remain `null`; search/fusion/reranker scores are never treated as relevance labels
or calibrated probabilities. A session without sources returns `status: no_contexts`.

```json
{
  "question": "What are the figure resolution requirements?",
  "session_id": "example",
  "top_k": 5,
  "evaluation_mode": "llm"
}
```

Read `response.context_evaluation.contexts` for per-source judgments and
`response.context_evaluation.ranking_metrics` for labeled metrics. Set
`relevant_chunk_ids` only when those labels are available; an empty label list is
rejected. For streaming, read these fields from the final JSON event.

### Answer faithfulness

Set `evaluate_faithfulness: true` on `/chat`, `/query`, or `/chat/stream` to
check the generated answer against the exact document contexts sent to the model.
This option is independent of `evaluation_mode` and defaults to `false`.

```json
{
  "question": "What are the figure resolution requirements?",
  "session_id": "example",
  "evaluate_faithfulness": true
}
```

The response includes `faithfulness_evaluation`. Its `score` is the fraction of
extracted factual answer claims supported by those contexts, from 0 to 1. The
`claims` list reports each statement, a reason, and a verdict: `supported`,
`contradicted`, or `not_enough_evidence`. The latter two count as unsupported.
Supporting and contradicting evidence includes source chunk IDs, file names,
page numbers, and quotes; quote text is checked against the referenced context,
allowing whitespace differences from PDF line wrapping.

Evaluation normally adds two calls to the configured generation model: one
extracts answer claims without seeing the sources, then another verifies all
extracted claims against the sources. Judge tokens are summed separately in
`judge_usage`; answer token counts are unchanged. This adds latency and hosted
model cost. For streams, tokens arrive normally and evaluation appears only in
the terminal `done` event after generation. Partial streams are not evaluated.

Missing sources, empty answers, and answers without factual claims return
`no_contexts`, `no_answer`, and `no_claims` respectively, with a null score.
Malformed judgments, missing verdicts, invented evidence quotes, and judge errors
return `status: unavailable` with a null score and preserve the generated answer.
Disabled evaluation returns `faithfulness_evaluation: null`.

No reference answer or relevance labels are needed. This is a model-based
estimate of grounding in supplied sources; it does not measure answer
completeness or external factual correctness. Review claim verdicts alongside
the score, especially for image tables marked as requiring OCR.

Session IDs are validated (`^[A-Za-z0-9._:@-]{1,128}$`) and questions are capped at `MAX_QUESTION_LENGTH`. Uploads are sniffed by magic bytes (PDF signature; DOCX/XLSX zip members) so renamed files are rejected. Rate limiting is enforced per client IP at `RATE_LIMIT_REQUESTS` per `RATE_LIMIT_PERIOD` seconds.

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/ping` | GET | Process liveness |
| `/health` | GET | Readiness: Qdrant, embeddings, LLM |
| `/documents/{session_id}/upload` | POST | Upload, index a PDF/DOCX/XLSX for a session |
| `/documents/{session_id}` | GET | List documents indexed for a session |
| `/documents/{session_id}/{doc_id}` | DELETE | Delete one document's chunks |
| `/chat` | POST | Ask a question; answer with cited sources and token usage |
| `/chat/stream` | POST | Stream an answer as server-sent events |
| `/query` | POST | Simple chat request using `message` and `session_id` |
| `/sessions/all_sessions` | GET | List sessions containing messages |
| `/sessions/{session_id}/history` | GET | Retrieve ordered messages |
| `/sessions/{session_id}/stats` | GET | Message counts and timestamps |
| `/sessions/{session_id}/delete` | DELETE | Delete that session's messages and documents |

`/chat` accepts optional `llm_provider` (must match the configured provider), `doc_ids` (restrict retrieval to specific documents), `include_tables`, `rerank`, `temperature`, `max_tokens`, `evaluation_mode`, `relevant_chunk_ids`, and `evaluate_faithfulness`; `top_k` controls retrieved chunk count. `sources` in the response list the chunks actually used, each with `file_name`, `page_number`, `score`, and a text preview.

```python
import requests

session_id = "example"
with open("report.pdf", "rb") as document:
    response = requests.post(
        f"http://localhost:8000/documents/{session_id}/upload",
        files={"file": document},
    )
    response.raise_for_status()

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "جمع فروش در جدول‌ها چقدر است؟", "session_id": session_id},
)
response.raise_for_status()
print(response.json()["answer"])
for source in response.json()["sources"]:
    print(source["file_name"], source["page_number"], source["score"])
```

Streaming text uses SSE `data:` fields; multiline tokens are encoded as multiple data lines. The final statistics event is JSON (`type: done`, including `sources` and usage), followed by `data: [DONE]`. If the client disconnects mid-stream, whatever was generated is saved to history and marked `partial`. Failed streams emit a JSON `type: error` event before closing.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

## Known limitations (pre-production checklist)

- **No authentication.** Anyone who can reach the API can read/write/delete any session. Put an authenticating gateway in front, or add API-key/JWT auth with per-user session ownership before public exposure.
- CORS with `["*"]` disallows credentialed requests; set real origins to enable credentials.
- The rate limiter and session locks are per process; scale-out requires Redis or similar.
- Persian PDFs stored in visual glyph order may extract as reversed/disjoint text; bidi repair is not implemented.
- No metrics/tracing export yet; logs are plain text with request IDs in responses (`X-Request-ID`).
