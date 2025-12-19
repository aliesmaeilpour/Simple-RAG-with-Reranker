from fastapi import APIRouter, HTTPException, Path, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from datetime import datetime
from models import *
from rag_engine import RAGEngine
import time
import os
import uuid
from document_processor import DocumentProcessor

router = APIRouter()
processor = DocumentProcessor()
startup_time = time.time()
UPLOAD_DIR = "uploads"
_chatbot = RAGEngine()
# _chatbot: Optional[RAGEngine] = None

def get_chatbot() -> RAGEngine:
    """Dependency to get chatbot instance."""
    if _chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot service not initialized"
        )
    return _chatbot


def set_chatbot(chatbot: RAGEngine):
    """Set the global chatbot instance."""
    global _chatbot
    _chatbot = chatbot


@router.get('/ping', tags=["Ping"], response_model=Pong)
async def get_uploadfile():

    return Pong(
        ok=True
    )


@router.post(
        "/documents/{session_id}/upload", 
        response_model=DocumentResponse,
        tags=["Documents"],
        responses={
            200: {"description": "Successful response"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
async def upload_document(
    file: UploadFile = File(...), 
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    # try:
    content = await file.read()
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR + '/' + unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    chunks, meta = await processor.process_file(file_path)
    print(meta.get('doc_id'))
    chatbot.history_manager.add_document_chunks(document_chunks=chunks, session_id=session_id)
    os.remove(file_path)
    return DocumentResponse(
        success=True,
        message="Document uploaded", 
        doc_id=meta['doc_id'], 
        metadata=meta
    )
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")
    

@router.post(
        "/chat", 
        response_model=QueryResponse,
        tags=["Chat"],
        responses={
            200: {"description": "Successful response"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
async def query_docs(
    req: QueryRequest,
    chatbot: RAGEngine = Depends(get_chatbot)
):
    try:
        result = chatbot.answer_question(
            req.question, 
            session_id=req.session_id, 
            llm_provider=req.llm_provider.value, 
            top_k=req.top_k
        )
        return QueryResponse(
            success=True, 
            question=req.question, 
            answer=result["response"], 
            sources=[], 
            processing_time=result["processing_time"], 
            llm_provider=req.llm_provider,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            reasoning_tokens=result["reasoning_tokens"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post(
        "/chat/stream",
        response_model=QueryResponse,
        tags=["Chat"],
        responses={
            200: {"description": "Successful response"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
def chat_stream(
    req: QueryRequestStream,
    chatbot: RAGEngine = Depends(get_chatbot)
):

    try:

        def generate():
            for token in chatbot.answer_question_stream(req.question, req.session_id):
                # print(token, end='')
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post(
    "/query",
    response_model=ChatResponse,
    tags=["query"],
    responses={
        200: {"description": "Successful response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
async def chat(
    request: ChatRequest,
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """
    Process a chat message and return the assistant's response.
    
    The chatbot will:
    - Detect if location search is needed
    - Search for relevant locations if applicable
    - Use conversation history for context
    - Generate a helpful response
    """
    try:
        result = chatbot.answer_question(
            user_message=request.message,
            session_id=request.session_id
        )
        
        metadata = result.get("metadata", {})
        
        # Convert location dicts to LocationSearchResult
        
        return ChatResponse(
            response=result.get("response", ""),
            session_id=request.session_id,
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
            location_search_performed=metadata.get("location_search_performed", False),
            locations_found=metadata.get("locations_found", 0),
            filters_applied=metadata.get("filters_applied"),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.get(
    "/sessions/all_sessions",
    response_model=SessionListResponse,
    description="Get a list of all chat sessions.",
    tags=["Sessions"]
)
async def list_sessions(
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """List all active chat sessions."""
    try:
        session_ids = chatbot.get_all_sessions()
        # print("session_ids: ", session_ids)
        
        sessions = []
        for sid in session_ids:
            sessions.append(SessionInfo(
                session_id=sid.get("session_id"),
                message_count=sid.get("message_count", 0),
                created_at=sid.get("first_message"),
                last_activity=sid.get("last_message")
            ))
        
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistory,
    description="Retrieve the full chat history for a session.",
    tags=["Sessions"]
)
async def get_session_history(
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """Get full conversation history for a session."""
    try:
        history = chatbot.get_session_history(session_id)
        
        from models import ChatMessage, MessageRole
        
        messages = []
        for msg in history:
            messages.append(ChatMessage(
                role=MessageRole(msg.get("role", "user")),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
                metadata=msg.get("metadata", {})
            ))
        
        return SessionHistory(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages)
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")


@router.get(
    "/sessions/{session_id}/stats",
    response_model=SessionStats,
    description="Get statistics about a chat session.",
    tags=["Sessions"]
)
async def get_session_stats(
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """Get statistics for a specific session."""
    try:
        stats = chatbot.get_session_stats(session_id)
        
        return SessionStats(
            session_id=session_id,
            total_messages=stats.get("message_count", 0),
            user_messages=stats.get("user_messages", 0),
            assistant_messages=stats.get("assistant_messages", 0),
            first_message_at=stats.get("first_message_at"),
            last_message_at=stats.get("last_message_at"),
            total_characters=stats.get("total_characters")
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)


@router.delete(
    "/sessions/{session_id}/delete",
    status_code=204,
    description="Delete a chat session and its history.",
    tags=["Sessions"]
)
async def delete_session(
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """Delete a session and all its history."""
    try:
        success = chatbot.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return None
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")