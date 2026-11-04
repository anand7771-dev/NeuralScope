"""
app/api/routes/embeddings.py
────────────────────────────
Endpoints for embedding vector inspection, PCA projections, and similarity analysis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np

from app.core.session import session_manager
from app.core.tensor_store import tensor_store
from app.utils.dim_reduction import compute_pca, compute_token_similarities

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.get("/sessions/{session_id}/pca")
async def get_session_embedding_pca(
    session_id: str,
    n_components: int = Query(default=2, ge=2, le=3),
) -> Dict[str, Any]:
    """Compute PCA coordinates for the token embeddings in a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Look for embeddings tensor reference in session events
    ref_id = None
    for evt in session.events:
        if evt.event_type.value == "embeddings_generated" and evt.tensor_references:
            ref_id = evt.tensor_references[0].ref_id
            break

    if not ref_id:
        # Fallback to inspecting any 3D/2D tensor stored for this session
        all_refs = tensor_store._session_map.get(session_id, set())
        for r in all_refs:
            arr = tensor_store.get(r)
            if arr is not None and (arr.ndim in (2, 3)):
                ref_id = r
                break

    if not ref_id:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding tensor found for session '{session_id}'."
        )

    arr = tensor_store.get(ref_id)
    if arr is None:
        raise HTTPException(status_code=404, detail="Embedding tensor data not in store")

    # Squeeze batch dimension if present: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Extract token labels if available in session
    token_labels = []
    for evt in session.events:
        if evt.event_type.value == "tokenization_completed" and "tokens" in evt.data:
            token_labels = evt.data["tokens"]
            break

    pca_result = compute_pca(arr, n_components=n_components, labels=token_labels)
    pca_result["session_id"] = session_id
    pca_result["tensor_ref_id"] = ref_id
    return pca_result


@router.get("/sessions/{session_id}/similarity")
async def get_session_token_similarity(session_id: str) -> Dict[str, Any]:
    """Compute pairwise cosine similarities across all input token embeddings."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ref_id = None
    for evt in session.events:
        if evt.event_type.value == "embeddings_generated" and evt.tensor_references:
            ref_id = evt.tensor_references[0].ref_id
            break

    if not ref_id:
        raise HTTPException(status_code=404, detail="Embeddings not available for this session")

    arr = tensor_store.get(ref_id)
    if arr is None:
        raise HTTPException(status_code=404, detail="Tensor data not found")

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    token_labels = []
    for evt in session.events:
        if evt.event_type.value == "tokenization_completed" and "tokens" in evt.data:
            token_labels = evt.data["tokens"]
            break

    if not token_labels:
        token_labels = [f"Token {i}" for i in range(arr.shape[0])]

    result = compute_token_similarities(arr, token_labels)
    result["session_id"] = session_id
    return result

