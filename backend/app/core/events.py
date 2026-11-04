"""
app/core/events.py
──────────────────
Pydantic v2 event schema for NeuralScope inference streaming.

Every piece of data that travels over the WebSocket is modelled here.
Tensor payloads are stored by reference (TensorReference) so the WebSocket
frame stays small even when attention tensors are gigabytes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "DataSource",
    "EventType",
    "TensorReference",
    "InferenceEvent",
]


# ── Enumerations ──────────────────────────────────────────────────────────────


class DataSource(str, Enum):
    """Provenance tag attached to every tensor and event.

    - ``ACTUAL``      – Values that come directly from the model's forward pass.
    - ``DERIVED``     – Computed from actual data (e.g. softmax of raw logits).
    - ``CONCEPTUAL``  – Schematic / diagram values used for explanation only.
    - ``SIMULATED``   – Demo / placeholder values; never used in production.
    """

    ACTUAL = "actual"
    DERIVED = "derived"
    CONCEPTUAL = "conceptual"
    SIMULATED = "simulated"


class EventType(str, Enum):
    """All distinct event types that the inference engine may emit."""

    # ── Session lifecycle ──────────────────────────────────────────────────
    SESSION_STARTED = "session_started"

    # ── Prompt handling ───────────────────────────────────────────────────
    PROMPT_RECEIVED = "prompt_received"
    PROMPT_FORMATTED = "prompt_formatted"

    # ── Tokenization ──────────────────────────────────────────────────────
    TOKENIZATION_STARTED = "tokenization_started"
    TOKENIZATION_COMPLETED = "tokenization_completed"

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDINGS_GENERATED = "embeddings_generated"

    # ── Input preparation ─────────────────────────────────────────────────
    INPUT_TENSOR_PREPARED = "input_tensor_prepared"

    # ── Transformer layers ────────────────────────────────────────────────
    TRANSFORMER_LAYER_STARTED = "transformer_layer_started"
    TRANSFORMER_LAYER_COMPLETED = "transformer_layer_completed"
    ATTENTION_COMPUTED = "attention_computed"
    MLP_COMPUTED = "mlp_computed"
    NORMALIZATION_COMPLETED = "normalization_completed"
    RESIDUAL_COMPUTED = "residual_computed"

    # ── KV cache ──────────────────────────────────────────────────────────
    KV_CACHE_UPDATED = "kv_cache_updated"

    # ── Output ────────────────────────────────────────────────────────────
    LOGITS_GENERATED = "logits_generated"
    PROBABILITIES_CALCULATED = "probabilities_calculated"
    TOKEN_SELECTED = "token_selected"
    TOKEN_GENERATED = "token_generated"

    # ── Completion ────────────────────────────────────────────────────────
    GENERATION_COMPLETED = "generation_completed"

    # ── Errors ────────────────────────────────────────────────────────────
    INFERENCE_ERROR = "inference_error"

    # ── Progress ──────────────────────────────────────────────────────────
    PROGRESS_UPDATE = "progress_update"


# ── Models ────────────────────────────────────────────────────────────────────


class TensorReference(BaseModel):
    """Lightweight descriptor that points to a tensor stored in TensorStore.

    The WebSocket frame carries only this reference; the full tensor data is
    fetched lazily via ``GET /api/tensors/{session_id}/{ref_id}``.
    """

    ref_id: str = Field(..., description="UUID key inside TensorStore")
    name: str = Field(..., description="Human-readable tensor name")
    shape: List[int] = Field(..., description="Tensor dimensions")
    dtype: str = Field(..., description="NumPy / PyTorch dtype string")
    device: str = Field(..., description="Device the tensor was computed on (cpu/cuda:0/…)")
    data_source: DataSource = Field(..., description="Provenance of this tensor")
    description: str = Field("", description="Optional free-text description")
    num_elements: int = Field(..., description="Total number of scalar elements")
    memory_bytes: int = Field(..., description="Approximate memory footprint in bytes")


class InferenceEvent(BaseModel):
    """A single structured event emitted during the inference pipeline.

    Events are streamed over WebSocket as newline-delimited JSON objects.
    Large payloads (attention weights, hidden states) are referenced via
    ``tensor_references`` rather than embedded in ``data``.
    """

    session_id: str = Field(..., description="Parent session UUID")
    event_id: str = Field(..., description="Unique event UUID")
    event_type: EventType = Field(..., description="Discriminator for the event kind")
    sequence_number: int = Field(..., description="Monotonically increasing counter within session")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    stage: str = Field(..., description="Pipeline stage label (e.g. 'tokenization', 'layer_3')")
    layer_index: Optional[int] = Field(None, description="Transformer layer index, if applicable")
    generation_step: Optional[int] = Field(None, description="Autoregressive step index (0-based)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key/value context")
    tensor_references: List[TensorReference] = Field(
        default_factory=list,
        description="References to tensors stored in TensorStore",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Small, directly-serialisable payload (top-k logits, token strings, etc.)",
    )
    data_source: DataSource = Field(
        DataSource.ACTUAL,
        description="Provenance tag for the event's primary data",
    )
    schema_version: str = Field("1.0", description="Schema version for forward-compatibility")
