# Persian RAG with Reranked Search, FastAPI

A modular Retrieval Augmented Generation system focused on Persian PDFs with semantic search, cross-encoder reranking, information extraction, and FastAPI endpoints.

## Features
- Persian PDF text extraction using PyMuPDF, Hazm normalization and sentence tokenization.
- Chunking with overlap and sentence boundary awareness.
- Embeddings via multilingual Sentence Transformers, default is `distiluse-base-multilingual-cased`.
- Vector store using FAISS, persisted to disk with sidecar JSON metadata.
- Two-stage retrieval, bi-encoder semantic search then cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Pattern based information extraction for figures, dates, currency, and summary generation.
- FastAPI endpoints for upload, query, extraction results, and health.
- Model and component interfaces to allow swapping models or stores.

## Quickstart
1. Create a virtualenv and install:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn app.main:app --reload --port 8000
```

3. Open the docs at:
```
http://127.0.0.1:8000/docs
```

## Folders
- `app/components` core classes, PDF processing, embeddings, search, extraction, QA engine.
- `app/storage` persistence helpers for FAISS index and metadata JSON.
- `data/uploads` user uploaded PDFs.
- `data/index` FAISS index files and metadata.
- `data/extracted` extracted info JSON per document.

## Notes
- This repo assumes CPU usage. If CUDA is available, SentenceTransformers and Torch will use it automatically.
- Persian Right To Left is handled at the text level. The API returns UTF-8 JSON. Clients can render RTL.
- For production, consider a SQLite or Postgres store for metadata, and S3 like storage for uploads.
