"""
app/adapters/base.py
─────────────────────
Abstract base class and shared data models for all model adapters.

Each supported model (GPT-2, GPT-J, LLaMA, …) implements
``AbstractModelAdapter`` and is registered with the ``InferenceEngine`` at
startup. The engine then delegates all inference calls through this interface,
keeping model-specific logic completely isolated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict

from pydantic import BaseModel, Field

from app.core.events import InferenceEvent

__all__ = ["ModelInfo", "AbstractModelAdapter"]


class ModelInfo(BaseModel):
    """Static metadata describing a loaded (or loadable) model.

    Returned by ``GET /api/models`` and ``GET /api/models/{model_id}``.
    All architecture details are populated from the HuggingFace config object
    at load time.
    """

    model_id: str = Field(..., description="HuggingFace model identifier")
    display_name: str = Field(..., description="Human-readable label")
    architecture: str = Field(..., description="Architecture family (e.g. 'GPT2')")
    num_layers: int = Field(..., description="Number of transformer blocks")
    num_heads: int = Field(..., description="Attention heads per layer")
    hidden_size: int = Field(..., description="Residual stream dimensionality")
    intermediate_size: int = Field(..., description="MLP hidden layer size")
    vocab_size: int = Field(..., description="Token vocabulary size")
    max_position_embeddings: int = Field(..., description="Maximum sequence length")
    num_parameters: int = Field(..., description="Total trainable parameter count")
    supports_kv_cache: bool = Field(..., description="Whether the model supports past_key_values")
    supports_attention_output: bool = Field(
        ..., description="Whether the model can return per-layer attention weights"
    )
    positional_encoding_type: str = Field(
        ..., description="Positional encoding scheme: 'learned', 'rope', or 'sinusoidal'"
    )
    normalization_type: str = Field(
        ..., description="Layer normalisation variant: 'layer_norm' or 'rms_norm'"
    )
    activation_function: str = Field(..., description="MLP activation function name")
    dtype: str = Field(..., description="Parameter dtype (e.g. 'float32', 'float16')")
    device: str = Field(..., description="Device the model is loaded on")
    is_loaded: bool = Field(..., description="Whether weights are currently in memory")


class AbstractModelAdapter(ABC):
    """Interface that all NeuralScope model adapters must implement.

    Implementations are responsible for:
    - Loading / unloading model weights.
    - Tokenising raw text.
    - Running a *fully instrumented* forward pass that emits structured
      ``InferenceEvent`` objects via the supplied async callback.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (may download from HuggingFace Hub)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free device memory."""
        ...

    # ── Info ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return static metadata for the managed model."""
        ...

    # ── Inference ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def run_instrumented_inference(
        self,
        prompt: str,
        config: Dict[str, Any],
        event_callback: Callable[[InferenceEvent], Awaitable[None]],
        session_id: str,
    ) -> Dict[str, Any]:
        """Run a full, instrumented forward pass and stream events.

        Args:
            prompt:         Raw input text.
            config:         Generation hyper-parameters (temperature, top_k, …).
            event_callback: Async callable invoked for every emitted event.
                            Must be awaited inside the implementation.
            session_id:     Owning session UUID (used by TensorStore).

        Returns:
            A dict containing at minimum:
            ``{"generated_text": str, "token_count": int, "duration_ms": float}``
        """
        ...

    # ── Tokenisation ──────────────────────────────────────────────────────────

    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenise *text* and return rich metadata.

        Returns:
            A dict with keys::

                input_ids:      list[int]
                tokens:         list[str]   – individual decoded token strings
                token_count:    int
                special_tokens: list[{"token": str, "id": int, "position": int}]
                bos_token:      str
                bos_token_id:   int
                eos_token:      str
                eos_token_id:   int
                vocab_size:     int
                tokenizer_type: str
                encoding:       str
        """
        ...
