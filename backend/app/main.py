"""
app/main.py
───────────
FastAPI application entry point for NeuralScope.
Configures CORS, lifespan loading of model adapters, and routing.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    health_router,
    models_router,
    sessions_router,
    tensors_router,
    embeddings_router,
)
from app.api.websocket import router as ws_router
from app.adapters.gpt2 import GPT2Adapter
from app.config import get_settings
from app.inference.engine import inference_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Register default GPT-2 adapter
    gpt2_adapter = GPT2Adapter()
    inference_engine.register_adapter("gpt2", gpt2_adapter)
    logger.info("Registered default adapter 'gpt2'")

    # Try pre-loading GPT-2 if torch & transformers are present
    try:
        gpt2_adapter.load()
        logger.info("GPT-2 adapter successfully loaded at startup")
    except Exception as exc:
        logger.warning(
            "Could not pre-load GPT-2 on startup: %s. "
            "Model will be loaded on first request when dependencies are available.",
            exc,
        )

    yield

    logger.info("Shutting down %s", settings.app_name)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Interactive LLM Inference Visualization and Exploration Platform API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(sessions_router)
app.include_router(tensors_router)
app.include_router(embeddings_router)
app.include_router(ws_router)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "health": "/health",
    }
