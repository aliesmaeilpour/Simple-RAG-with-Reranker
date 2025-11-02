# Production RAG with FastAPI, LangChain, FAISS, Qwen, and Table Awareness

## Features
- Upload and index PDF, DOCX, and Excel. Text and tables are extracted and preserved.
- Vector search with FAISS, sentence-transformers embeddings.
- Cross encoder reranker for better relevance.
- Qwen integration, local HuggingFace or DashScope API.
- Table aware prompts and table only querying.
- Citations with filename and page numbers.
- Async ready FastAPI with rate limiting and API key check.
- Health endpoint, reindex endpoint, robust logging.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export API_KEY=dev-secret-key
python run.py
```

Set custom config via environment variables:
```
export API_KEY="your-prod-key"
export LLM_BACKEND="local"            # or api
export QWEN_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export CHUNK_SIZE=1200
export CHUNK_OVERLAP=150
export RETRIEVAL_K=12
export RERANK_K=6
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export RATE_LIMIT="100/minute"
```

For DashScope:
```
export LLM_BACKEND=api
export DASHSCOPE_API_KEY=sk-...
```

## API

Header `x-api-key: <your key>` is required on protected endpoints.

### Health
- `GET /health`

### Upload
- `POST /documents/upload`
  - multipart form, `files` is a list of files
  - supports pdf, docx, xlsx

### List docs
- `GET /documents`

### Delete doc metadata
- `DELETE /documents/{doc_id}`
  - Removes metadata only. Call `/reindex` to rebuild FAISS without the document.

### Reindex
- `POST /reindex`
  - Rebuilds FAISS from files in `data/uploads`

### Query
- `POST /query`
```json
{
  "query": "What is the pressure drop in the reactor table",
  "top_k": 5,
  "table_only": false,
  "filters": {"doc_id":"<uuid>"},
  "generation_params": {"temperature": 0.2, "max_new_tokens": 512}
}
```
- Returns `answer` and `sources` with filename and page.

### Query tables only
- `POST /query/tables`
  - Same body as `/query` but searches only table chunks.

## Notes
- Table extraction uses pdfplumber, python-docx, and pandas.
- FAISS delete by metadata is limited. This template rebuilds on `/reindex`.
- For GPU, install torch with CUDA. The Qwen wrapper selects device automatically unless `qwen_device=cpu`.
