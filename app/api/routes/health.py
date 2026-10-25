import time
from typing import Optional

from fastapi import Request, APIRouter

from app.core.config import settings
from app.schemas.models import (
    HealthCheckResponse,
    Pong
)
from app.service.rag_engine import RAGEngine

router = APIRouter()

@router.get("/ping", tags=["Ping"], response_model=Pong)
async def ping():
    return Pong(ok=True)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    description="Readiness of the process and its backing components.",
)
async def health(request: Request):
    chatbot: Optional[RAGEngine] = getattr(request.app.state, "chatbot", None)
    if chatbot is None:
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.app_version,
            uptime=0.0,
            components={"qdrant": False, "embeddings": False, "llm": False},
        )
    components = await chatbot.component_health()
    status = "healthy" if all(components.values()) else "degraded"
    uptime = time.monotonic() - getattr(request.app.state, "start_time", time.monotonic())
    return HealthCheckResponse(
        status=status,
        version=settings.app_version,
        uptime=uptime,
        components=components,
    )