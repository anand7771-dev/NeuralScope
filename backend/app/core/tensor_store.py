"""
app/core/tensor_store.py
─────────────────────────
In-memory store for large NumPy arrays produced during inference.

Tensors are stored by a UUID reference ID and never embedded directly in
WebSocket frames. The REST endpoint ``GET /api/tensors/{session_id}/{ref_id}``
retrieves them on demand, optionally applying a slice specification.

Memory usage is tracked and capped at ``max_mb``.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Set

import numpy as np

from app.config import get_settings
from app.core.events import DataSource, TensorReference
from app.utils.tensor_utils import parse_slice_spec, tensor_summary

__all__ = ["TensorStore", "tensor_store"]

logger = logging.getLogger(__name__)


class TensorStore:
    """Thread-safe (asyncio-level) in-memory NumPy array store.

    Attributes:
        _store:         Maps ``ref_id -> np.ndarray``.
        _session_map:   Maps ``session_id -> set of ref_ids``.
        _max_bytes:     Hard memory ceiling in bytes.
        _used_bytes:    Running total of stored bytes.
    """

    def __init__(self, max_mb: int = 512) -> None:
        self._store: Dict[str, np.ndarray] = {}
        self._session_map: Dict[str, Set[str]] = {}
        self._max_bytes: int = max_mb * 1024 * 1024
        self._used_bytes: int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        name: str,
        tensor: np.ndarray,
        data_source: DataSource,
        description: str = "",
    ) -> TensorReference:
        """Store a NumPy array and return its ``TensorReference``.

        Args:
            session_id:  Owning session UUID.
            name:        Human-readable label (e.g. ``"layer_3_attention"``).
            tensor:      The array to store. It is **not** copied — ensure the
                         caller does not mutate it afterwards.
            data_source: Provenance tag.
            description: Optional free-text note.

        Returns:
            A ``TensorReference`` that can be embedded in an ``InferenceEvent``.

        Raises:
            MemoryError: When adding the tensor would exceed ``max_mb``.
        """
        # Ensure contiguous C-order layout for predictable memory accounting
        array = np.ascontiguousarray(tensor)
        nbytes = array.nbytes

        if self._used_bytes + nbytes > self._max_bytes:
            raise MemoryError(
                f"TensorStore limit reached ({self._max_bytes // (1024**2)} MB). "
                f"Cannot store '{name}' ({nbytes / (1024**2):.2f} MB)."
            )

        ref_id = str(uuid.uuid4())
        self._store[ref_id] = array
        self._used_bytes += nbytes

        # Register under session for bulk cleanup
        self._session_map.setdefault(session_id, set()).add(ref_id)

        ref = TensorReference(
            ref_id=ref_id,
            name=name,
            shape=list(array.shape),
            dtype=str(array.dtype),
            device="cpu",  # Always CPU after detach/numpy conversion
            data_source=data_source,
            description=description,
            num_elements=int(array.size),
            memory_bytes=nbytes,
        )
        logger.debug(
            "Stored tensor '%s' (%s) → ref_id=%s  size=%.2f MB",
            name,
            array.shape,
            ref_id,
            nbytes / (1024**2),
        )
        return ref

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, ref_id: str) -> Optional[np.ndarray]:
        """Return the stored array for ``ref_id``, or ``None``."""
        return self._store.get(ref_id)

    def get_slice(self, ref_id: str, slice_spec: str) -> Optional[np.ndarray]:
        """Return a slice of the stored array.

        Args:
            ref_id:     Reference ID of the stored tensor.
            slice_spec: Comma-separated slice notation, e.g. ``"0,:,0:5"``.

        Returns:
            Sliced NumPy array, or ``None`` if ``ref_id`` is unknown.
        """
        array = self._store.get(ref_id)
        if array is None:
            return None

        idx = parse_slice_spec(slice_spec, array.shape)
        return array[idx]

    def get_summary(self, ref_id: str) -> Optional[Dict]:
        """Return statistical summary for the stored tensor.

        Returns ``None`` if ``ref_id`` is not found.
        """
        array = self._store.get(ref_id)
        if array is None:
            return None
        return tensor_summary(array)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Remove all tensors belonging to ``session_id``.

        Called automatically by ``SessionManager.delete_session()``.
        """
        ref_ids: Set[str] = self._session_map.pop(session_id, set())
        freed: int = 0
        for ref_id in ref_ids:
            arr = self._store.pop(ref_id, None)
            if arr is not None:
                freed += arr.nbytes
        self._used_bytes = max(0, self._used_bytes - freed)
        logger.info(
            "Cleared %d tensors for session %s (freed %.2f MB)",
            len(ref_ids),
            session_id,
            freed / (1024**2),
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def memory_used_mb(self) -> float:
        """Return approximate total MB currently held in the store."""
        return self._used_bytes / (1024**2)

    def __len__(self) -> int:
        return len(self._store)


# ── Singleton ─────────────────────────────────────────────────────────────────

tensor_store: TensorStore = TensorStore(max_mb=get_settings().tensor_store_max_mb)
