import logging

from fastapi import APIRouter, Depends, HTTPException, Path

from app.schemas.models import (
    ChatMessage,
    DeleteSession,
    MessageRole,
    SessionHistory,
    SessionInfo,
    SessionListResponse,
    SessionStats,
)
from app.service.rag_engine import RAGEngine
from app.api.deps import get_chatbot


router = APIRouter()


logger = logging.getLogger(__name__)


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
        sessions_data = await chatbot.get_all_sessions()

        sessions = []
        for sid in sessions_data:
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


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
        history = await chatbot.get_session_history(session_id)

        messages = []
        for msg in history:
            messages.append(ChatMessage(
                role=MessageRole(msg.get("role", "user")),
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp"),
                metadata=msg.get("metadata", {}) or (
                    {"partial": True} if msg.get("partial") else {}
                ),
            ))

        return SessionHistory(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages)
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get history for session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to get session history")


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
    """Get statistics for a specific chat session."""
    try:
        stats = await chatbot.get_session_stats(session_id)

        return SessionStats(
            session_id=session_id,
            total_messages=stats.get("message_count", 0),
            user_messages=stats.get("user_messages", 0),
            assistant_messages=stats.get("assistant_messages", 0),
            first_message_at=stats.get("first_message_at"),
            last_message_at=stats.get("last_message_at"),
            total_characters=stats.get("total_characters")
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get stats for session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to get session stats")


@router.delete(
    "/sessions/{session_id}/delete",
    response_model=DeleteSession,
    status_code=200,
    description="Delete a chat session and its history.",
    tags=["Sessions"]
)
async def delete_session(
    session_id: str = Path(..., description="Session ID"),
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """Delete a session and all its history."""
    try:
        success = await chatbot.delete_session(session_id)

        return DeleteSession(
            success=success,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete session %s", session_id)
        raise HTTPException(status_code=500, detail="Failed to delete session")