"""
app/api/routes/tensors.py
─────────────────────────
Endpoints for fetching on-demand slices, histograms, and summaries of stored tensors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np

from app.core.tensor_store import tensor_store
from app.utils.tensor_utils import parse_slice_spec, serialize_tensor_response, tensor_summary

router = APIRouter(prefix="/api/tensors", tags=["tensors"])


@router.get("/{session_id}/{ref_id}")
async def get_tensor(
    session_id: str,
    ref_id: str,
    slice_spec: Optional[str] = Query(
        default=None,
        description="NumPy slice notation string (e.g., '0,:,0:5' or '0,0,:,:')",
    ),
    format: str = Query(default="json", description="Response format ('json')"),
) -> Dict[str, Any]:
    """Retrieve tensor values by reference ID with optional slicing."""
    arr = tensor_store.get(ref_id)
    if arr is None:
        raise HTTPException(status_code=404, detail=f"Tensor reference '{ref_id}' not found")

    if slice_spec:
        sliced_arr = tensor_store.get_slice(ref_id, slice_spec)
        if sliced_arr is None:
            raise HTTPException(status_code=400, detail=f"Invalid slice specification: {slice_spec}")
        arr = sliced_arr

    return serialize_tensor_response(arr, ref_id=ref_id)


@router.get("/{session_id}/{ref_id}/summary")
async def get_tensor_summary(session_id: str, ref_id: str) -> Dict[str, Any]:
    """Retrieve statistical summary for a stored tensor without transmitting full array."""
    summary = tensor_store.get_summary(ref_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Tensor reference '{ref_id}' not found")
    return summary


@router.get("/{session_id}/{ref_id}/histogram")
async def get_tensor_histogram(
    session_id: str,
    ref_id: str,
    bins: int = Query(default=20, ge=5, le=100, description="Number of histogram bins"),
) -> Dict[str, Any]:
    """Compute value distribution histogram for weight/activation tensor inspection."""
    arr = tensor_store.get(ref_id)
    if arr is None:
        raise HTTPException(status_code=404, detail=f"Tensor reference '{ref_id}' not found")

    flat = arr.ravel().astype(np.float64)
    if flat.size == 0 or np.isnan(flat).all():
        return {
            "ref_id": ref_id,
            "counts": [],
            "bin_edges": [],
            "min": 0.0,
            "max": 0.0,
        }

    clean = flat[np.isfinite(flat)]
    if clean.size == 0:
        return {"ref_id": ref_id, "counts": [], "bin_edges": []}

    counts, bin_edges = np.histogram(clean, bins=bins)

    return {
        "ref_id": ref_id,
        "counts": counts.tolist(),
        "bin_edges": [round(float(b), 4) for b in bin_edges],
        "total_elements": int(clean.size),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
    }


@router.get("/{session_id}/{ref_id}/matrix2d")
async def get_tensor_2d_slice(
    session_id: str,
    ref_id: str,
    row_offset: int = Query(default=0, ge=0),
    col_offset: int = Query(default=0, ge=0),
    max_rows: int = Query(default=20, ge=1, le=100),
    max_cols: int = Query(default=20, ge=1, le=100),
    dim_indices: Optional[str] = Query(
        default=None,
        description="Comma-separated leading indices for >2D tensors, e.g. '0,0' for head 0, batch 0",
    ),
) -> Dict[str, Any]:
    """Paginated 2D matrix slice viewer for large tensors."""
    arr = tensor_store.get(ref_id)
    if arr is None:
        raise HTTPException(status_code=404, detail=f"Tensor reference '{ref_id}' not found")

    target_2d = arr
    if arr.ndim > 2:
        if dim_indices:
            idx_tuple = tuple(int(x.strip()) for x in dim_indices.split(","))
            target_2d = arr[idx_tuple]
        else:
            slc = tuple(0 for _ in range(arr.ndim - 2))
            target_2d = arr[slc]

    if target_2d.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Extracted slice has shape {target_2d.shape}, expected 2D matrix",
        )

    total_rows, total_cols = target_2d.shape
    row_slice = target_2d[row_offset : row_offset + max_rows, col_offset : col_offset + max_cols]

    return {
        "ref_id": ref_id,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "rows_returned": row_slice.shape[0],
        "cols_returned": row_slice.shape[1],
        "values": row_slice.tolist(),
    }
