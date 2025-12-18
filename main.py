import uvicorn
from fastapi import FastAPI, UploadFile, File, Path
import time
import logging
from models import *
from config import settings
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
import os
import uuid
# from pathlib import Path
from route import router

app = FastAPI(title=settings.app_name)
app.include_router(router)
processor = DocumentProcessor()
rag = RAGEngine()
startup_time = time.time()
UPLOAD_DIR = "uploads"

@app.get("/health")
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        uptime=(time.time() - startup_time),
        components={"vector_store": True, "llm": True},
    )

@app.post("/documents/{session_id}/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), 
    session_id: str = Path(..., description="Session ID")
):
    content = await file.read()
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR + '/' + unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    chunks, meta = await processor.process_file(file_path)
    print(meta.get('doc_id'))
    rag.history_manager.add_document_chunks(document_chunks=chunks, session_id=session_id)
    os.remove(file_path)
    return DocumentResponse(
        success=True,
        message="Document uploaded", 
        doc_id=meta['doc_id'], 
        metadata=meta
    )

@app.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    result = rag.answer_question(
        req.question, 
        session_id=req.session_id, 
        llm_provider=req.llm_provider.value, 
        top_k=req.top_k
    )
    return QueryResponse(
        success=True, 
        question=req.question, 
        answer=result["answer"], 
        sources=[], 
        processing_time=result["processing_time"], 
        llm_provider=req.llm_provider
    )

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug)