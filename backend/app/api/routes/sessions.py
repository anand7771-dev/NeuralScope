"""
app/api/routes/sessions.py
──────────────────────────
Endpoints for managing inference sessions and retrieving recorded events.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.events import InferenceEvent
from app.core.session import session_manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class InferenceRequest(BaseModel):
    model_id: str = Field(default="gpt2", description="Target model identifier")
    prompt: str = Field(..., min_length=1, description="Input text prompt")
    max_new_tokens: int = Field(default=50, ge=1, le=500, description="Max tokens to generate")
    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, description="Top-k filtering threshold (0 = disabled)")
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, description="Top-p nucleus threshold")
    do_sample: bool = Field(default=True, description="Whether to sample or use greedy decoding")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="Repetition penalty")


@router.post("")
async def create_session(request: InferenceRequest) -> Dict[str, Any]:
    """Create a new inference session.

    Returns the session ID which the client connects to via WebSocket
    to initiate execution.
    """
    config = request.model_dump()
    model_id = config.pop("model_id")
    prompt = config.pop("prompt")

    session = session_manager.create_session(
        model_id=model_id,
        prompt=prompt,
        config=config,
    )

    return {
        "session_id": session.session_id,
        "model_id": session.model_id,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "prompt": session.prompt,
        "config": session.config,
    }


@router.get("")
async def list_sessions() -> List[Dict[str, Any]]:
    """List all tracked sessions."""
    return [s.to_dict() for s in session_manager.list_sessions()]


@router.get("/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get metadata and status of a specific session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.to_dict()


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its associated tensors."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/{session_id}/events", response_model=List[InferenceEvent])
async def get_session_events(session_id: str) -> List[InferenceEvent]:
    """Retrieve all recorded events for replay."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.events

