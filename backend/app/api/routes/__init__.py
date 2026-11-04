# app/api/routes/__init__.py
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.tensors import router as tensors_router
from app.api.routes.embeddings import router as embeddings_router

__all__ = [
    "health_router",
    "models_router",
    "sessions_router",
    "tensors_router",
    "embeddings_router",
]
