import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import logging
from models import *
from config import settings
from rag_engine import RAGEngine
# from pathlib import Path
from route import router
import sys
import os
from contextlib import asynccontextmanager
from route import set_chatbot


def setup_logging():
    """Configure application logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level.lower(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


logger = setup_logging()

_chatbot_instance: Optional[RAGEngine] = None


def initialize_chatbot() -> RAGEngine:
    """Initialize the chatbot with all required components."""
    global _chatbot_instance
    
    logger.info("Initializing Location Chatbot...")
    
    # Validate OpenAI API key
    if not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key not configured. "
            "Set the OPENAI_API_KEY environment variable."
        )
    
    # Create Chormadb data directory
    os.makedirs(settings.chromadb_path, exist_ok=True)
    logger.info(f"Chromadb data directory: {settings.chromadb_path}")
    
    # Initialize chatbot
    try:
        _chatbot_instance = RAGEngine()
        
        logger.info("Chatbot initialized successfully")
        logger.info(f"  - Chromadb: {settings.chromadb_path}")
        logger.info(f"  - Embedding Model: {settings.openai_embedding_model}")
        logger.info(f"  - Chat Model: {settings.openai_model}")
        
        return _chatbot_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        raise


def shutdown_chatbot():
    """Gracefully shutdown chatbot resources."""
    global _chatbot_instance
    
    if _chatbot_instance:
        logger.info("Shutting down chatbot...")
        try:
            _chatbot_instance.shutdown()
        except Exception as e:
            logger.error(f"Error during chatbot shutdown: {e}")
        finally:
            _chatbot_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Location-Aware Chatbot API")
    logger.info("=" * 60)
    
    try:
        chatbot = initialize_chatbot()
        set_chatbot(chatbot)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Location-Aware Chatbot API")
    logger.info("=" * 60)
    
    shutdown_chatbot()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "details": errors,
            "status_code": 422
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "detail": str(exc) if settings.debug else None
        }
    )


app.include_router(router)

def main():
    """Run the application server."""
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()