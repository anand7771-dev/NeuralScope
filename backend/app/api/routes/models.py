"""
app/api/routes/models.py
────────────────────────
Endpoints for inspecting and loading model architectures.
"""
from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException
from app.adapters.base import ModelInfo
from app.inference.engine import inference_engine

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    """List all registered models and their status."""
    return inference_engine.list_available_models()


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Retrieve metadata for a specific model."""
    adapter = inference_engine.get_adapter(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return adapter.get_info()


@router.post("/{model_id}/load")
async def load_model(model_id: str):
    """Load model weights and tokenizer into memory."""
    adapter = inference_engine.get_adapter(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    try:
        adapter.load()
        return {"status": "loaded", "model_id": model_id, "info": adapter.get_info()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_id}': {exc}")

