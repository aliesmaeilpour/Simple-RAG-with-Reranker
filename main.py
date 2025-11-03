import uvicorn
from fastapi import FastAPI, UploadFile, File
import time
import logging
from models import *
from config import settings
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

app = FastAPI(title=settings.app_name)
processor = DocumentProcessor()
rag = RAGEngine()
startup_time = time.time()

@app.get("/health")
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        uptime=(time.time() - startup_time),
        components={"vector_store": True, "llm": True},
    )

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    chunks, meta = await processor.process_document(content, file.filename)
    rag.vector_service.add_documents(chunks)
    return DocumentResponse(success=True, message="Document uploaded", doc_id=meta.doc_id, metadata=meta)

@app.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    result = rag.answer_question(req.question, llm_provider=req.llm_provider.value, top_k=req.top_k)
    return QueryResponse(success=True, question=req.question, answer=result["answer"], sources=[], processing_time=result["processing_time"], llm_provider=req.llm_provider)

if __name__ == "__main__":
    uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, reload=settings.debug))