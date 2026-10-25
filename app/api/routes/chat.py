import json
import logging
import time
from contextlib import aclosing

from fastapi import Depends, HTTPException, APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    QueryRequest,
    QueryRequestStream,
    QueryResponse
)
from app.service.rag_engine import RAGEngine
from app.api.deps import get_chatbot, _to_sources


logger = logging.getLogger(__name__)


router = APIRouter()

@router.post(
    "/chat",
    response_model=QueryResponse,
    tags=["Chat"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        503: {"description": "Service unavailable"},
    }
)
async def query_docs(
    req: QueryRequest,
    chatbot: RAGEngine = Depends(get_chatbot)
):
    try:
        provider = req.llm_provider or LLMProvider(settings.llm_provider)
        if provider.value != settings.llm_provider:
            raise HTTPException(status_code=400, detail="Requested LLM provider is not configured")

        result = await chatbot.answer_question(
            req.question,
            session_id=req.session_id,
            llm_provider=provider.value,
            top_k=req.top_k,
            doc_ids=req.doc_ids,
            include_tables=req.include_tables,
            rerank=req.rerank,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            evaluation_mode=req.evaluation_mode,
            relevant_chunk_ids=req.relevant_chunk_ids,
            evaluate_faithfulness=req.evaluate_faithfulness,
        )
        return QueryResponse(
            success=True,
            question=req.question,
            answer=result["response"],
            sources=_to_sources(result["metadata"]["context_used"]),
            context_evaluation=result["metadata"].get("context_evaluation"),
            faithfulness_evaluation=result["metadata"].get("faithfulness_evaluation"),
            processing_time=result["processing_time"],
            llm_provider=provider,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            reasoning_tokens=result["reasoning_tokens"]
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail="Chat processing error")


@router.post(
    "/chat/stream",
    tags=["Chat"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        503: {"description": "Service unavailable"},
    }
)
async def chat_stream(
    req: QueryRequestStream,
    chatbot: RAGEngine = Depends(get_chatbot)
):
    async def generate():
        try:
            async with aclosing(chatbot.answer_question_stream(
                query=req.question,
                session_id=req.session_id,
                top_k=req.top_k,
                doc_ids=req.doc_ids,
                include_tables=req.include_tables,
                rerank=req.rerank,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                evaluation_mode=req.evaluation_mode,
                relevant_chunk_ids=req.relevant_chunk_ids,
                evaluate_faithfulness=req.evaluate_faithfulness,
            )) as stream:
                async for token in stream:
                    if isinstance(token, dict):
                        yield f"data: {json.dumps(token, ensure_ascii=False, default=str)}\n\n"
                    else:
                        yield "".join(f"data: {line}\n" for line in token.split("\n")) + "\n"
            # [DONE] only emitted on clean completion
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("Streaming chat failed")
            yield 'data: {"type": "error", "message": "Chat processing error"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/query",
    response_model=ChatResponse,
    tags=["query"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid request"},
        503: {"description": "Service unavailable"},
    }
)
async def chat(
    request: ChatRequest,
    chatbot: RAGEngine = Depends(get_chatbot)
):
    """
    Process a chat message and return the assistant's response.

    The chatbot will:
    - Retrieve relevant document chunks uploaded to the session
    - Use conversation history for context
    - Generate a helpful, grounded response with citations
    """
    try:
        result = await chatbot.answer_question(
            query=request.message,
            session_id=request.session_id,
            evaluation_mode=request.evaluation_mode,
            relevant_chunk_ids=request.relevant_chunk_ids,
            evaluate_faithfulness=request.evaluate_faithfulness,
        )

        metadata = result.get("metadata", {})

        return ChatResponse(
            response=result.get("response", ""),
            session_id=request.session_id,
            timestamp=metadata.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            context_evaluation=metadata.get("context_evaluation"),
            faithfulness_evaluation=metadata.get("faithfulness_evaluation"),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Chat processing error")