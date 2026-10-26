from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.routes import chat, db_stats, health, doc

router = APIRouter(prefix="/v1")

# Liveness/readiness stays public for orchestrators.
router.include_router(health.router)
router.include_router(doc.router)
router.include_router(chat.router)
router.include_router(db_stats.router)
