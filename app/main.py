import logging
import sys
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import router
from app.api.deps import set_chatbot
from app.core.config import settings
from app.core.middleware import RateLimitMiddleware, RequestIDMiddleware
from app.service.rag_engine import RAGEngine


def _build_log_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                settings.log_file,
                maxBytes=10_000_000,
                backupCount=5,
                encoding="utf-8",
            )
        )
    return handlers


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=_build_log_handlers(),
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.start_time = time.monotonic()
    chatbot = RAGEngine()
    try:
        await chatbot.initialize()
        set_chatbot(chatbot)
        application.state.chatbot = chatbot
        logger.info("Application initialized with Qdrant at %s", settings.qdrant_url)
        yield
    finally:
        set_chatbot(None)
        application.state.chatbot = None
        await chatbot.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# "*" origins cannot be combined with credentialed CORS; credentials are only
# allowed once origins are pinned to real hosts.
allow_credentials = settings.cors_origins and "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    period_seconds=settings.rate_limit_period,
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "details": [{
                "field": ".".join(str(location) for location in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            } for error in exc.errors()],
            "status_code": 422,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "detail": str(exc) if settings.debug else None,
        },
    )


app.include_router(router)


def main():
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        # reload mode forks a reloader and cannot be combined with workers
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
