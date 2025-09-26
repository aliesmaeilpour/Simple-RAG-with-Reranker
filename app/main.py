from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import shutil
import os

from .config import (
    UPLOADS_DIR, INDEX_DIR, EXTRACTED_DIR,
    EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, FINAL_K,
    SEM_WEIGHT, RERANK_WEIGHT
)
from .components.pdf_processor import PDFProcessor
from .components.embedding_manager import EmbeddingManager
from .components.search_system import SearchSystem
from .components.info_extraction import InfoExtractionAgent
from .components.qa_engine import QAEngine
from .storage.meta_store import MetaStore

app = FastAPI(title="Persian RAG with Reranked Search")

# Ensure data dirs
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

pdf = PDFProcessor()
em = EmbeddingManager(EMBEDDING_MODEL_NAME, INDEX_DIR)
search = SearchSystem(em, CROSS_ENCODER_MODEL_NAME, sem_weight=SEM_WEIGHT, rerank_weight=RERANK_WEIGHT)
extractor = InfoExtractionAgent(EXTRACTED_DIR)
qa_engine = QAEngine(search, llm=None)  # provide callable to use an LLM here
meta_store = MetaStore(INDEX_DIR)

class AskPayload(BaseModel):
    question: str
    top_k: Optional[int] = None
    final_k: Optional[int] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/docs_list")
def list_docs():
    return meta_store.list_documents()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    doc_id = str(uuid.uuid4())[:8]
    out_path = UPLOADS_DIR / f"{doc_id}.pdf"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process PDF
    try:
        text = pdf.extract_text(str(out_path))
        chunks = pdf.chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        metas = []
        for i, c in enumerate(chunks):
            metas.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "source": str(out_path.name),
                "text": c
            })
        if chunks:
            em.add(chunks, metas)
        extracted = extractor.extract(doc_id, text, chunks)
        meta_store.add_document(doc_id, {
            "filename": out_path.name,
            "num_chunks": len(chunks),
            "extracted_path": str((EXTRACTED_DIR / f"{doc_id}.json").name)
        })
        return {"doc_id": doc_id, "num_chunks": len(chunks), "extracted": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.get("/extract/{doc_id}")
def get_extracted(doc_id: str):
    path = EXTRACTED_DIR / f"{doc_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(content=path.read_text(encoding="utf-8"))

@app.post("/ask")
def ask(payload: AskPayload):
    top_k = payload.top_k or TOP_K
    final_k = payload.final_k or FINAL_K
    result = qa_engine.answer(payload.question, top_k=top_k, final_k=final_k)
    return result
