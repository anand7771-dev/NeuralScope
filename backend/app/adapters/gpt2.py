"""
app/adapters/gpt2.py
─────────────────────
Full GPT-2 model adapter for NeuralScope.

This adapter loads ``GPT2LMHeadModel`` via HuggingFace Transformers, registers
PyTorch forward hooks on every transformer layer, and runs a manual
autoregressive generation loop that emits fine-grained ``InferenceEvent``
objects for every computational step.

**Instrumentation levels**

- ``lightweight`` – Only timing, token-selection, and generation-completion
  events.  No tensor references for internal states.
- ``standard``   – All events including attention weights and MLP outputs.
  Tensors stored in ``TensorStore``; references emitted via WebSocket.
- ``deep``       – Like ``standard`` plus residual stream snapshots and
  full per-head attention matrices.

The manual generation loop provides access to ``past_key_values`` (KV cache),
per-layer attention weights, and raw logits at each step — information that
the HuggingFace ``model.generate()`` API does not easily expose.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.adapters.base import AbstractModelAdapter, ModelInfo
from app.config import get_settings
from app.core.events import DataSource, EventType, InferenceEvent, TensorReference
from app.core.tensor_store import tensor_store
from app.instrumentation.hooks import HookManager, capture_attention_weights, capture_module_io
from app.instrumentation.profiler import InferenceProfiler
from app.tokenization.service import TokenizationService
from app.utils.math_utils import softmax, top_k_filter, top_p_filter
from app.utils.tensor_utils import tensor_to_numpy

__all__ = ["GPT2Adapter"]

logger = logging.getLogger(__name__)

# ── Lazy torch / transformers imports ─────────────────────────────────────────
# We defer the actual import to load() so that importing this module in a
# test environment (without PyTorch installed) does not immediately crash.

_TORCH_MISSING_MSG = (
    "PyTorch is required. Install dependencies with:\n"
    "    pip install -r requirements.txt"
)
_TRANSFORMERS_MISSING_MSG = (
    "HuggingFace Transformers is required. Install dependencies with:\n"
    "    pip install -r requirements.txt"
)


def _require_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise ImportError(_TORCH_MISSING_MSG) from exc


def _require_transformers() -> Tuple[Any, Any]:
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: PLC0415
        return GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as exc:
        raise ImportError(_TRANSFORMERS_MISSING_MSG) from exc


# ── Adapter ───────────────────────────────────────────────────────────────────


class GPT2Adapter(AbstractModelAdapter):
    """NeuralScope adapter for OpenAI GPT-2.

    Wraps ``GPT2LMHeadModel`` and instruments every transformer block with
    forward hooks.  The ``run_instrumented_inference`` method implements a
    manual token-by-token generation loop so that internal states can be
    captured and streamed at each step.

    Args:
        model_id:  HuggingFace model identifier (default: ``"gpt2"``).
        cache_dir: Local directory for caching downloaded weights.
    """

    def __init__(
        self,
        model_id: str = "gpt2",
        cache_dir: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._model_id = model_id
        self._cache_dir = cache_dir or settings.model_cache_dir
        self._instrumentation_level: str = settings.instrumentation_level

        # Set after load()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if necessary) and load GPT-2 weights into memory.

        Automatically selects CUDA if available, falls back to CPU.
        The model is placed in evaluation mode and gradient computation is
        disabled permanently via ``torch.no_grad()`` context (applied at
        inference time).
        """
        torch = _require_torch()
        GPT2LMHeadModel, GPT2Tokenizer = _require_transformers()

        logger.info("Loading GPT-2 model '%s' …", self._model_id)

        # Device selection
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self._device)

        # Load tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
        )
        # GPT-2 has no pad token; set it to EOS so that batch encoding works
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = GPT2LMHeadModel.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
        )
        self._model.to(self._device)
        self._model.eval()

        self._is_loaded = True
        logger.info(
            "GPT-2 loaded. Parameters: %s  Device: %s",
            f"{sum(p.numel() for p in self._model.parameters()):,}",
            self._device,
        )

    def unload(self) -> None:
        """Release model weights and free device memory."""
        if not self._is_loaded:
            return

        torch = _require_torch()

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("GPT-2 unloaded.")

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> ModelInfo:
        """Return static metadata for the loaded GPT-2 model."""
        cfg = self._model.config if self._is_loaded else None
        num_params = (
            sum(p.numel() for p in self._model.parameters())
            if self._is_loaded
            else 0
        )

        # GPT-2 defaults if model not loaded yet
        num_layers = getattr(cfg, "n_layer", 12)
        num_heads = getattr(cfg, "n_head", 12)
        hidden_size = getattr(cfg, "n_embd", 768)
        intermediate_size = hidden_size * 4  # GPT-2 uses 4x hidden for MLP
        vocab_size = getattr(cfg, "vocab_size", 50257)
        max_position = getattr(cfg, "n_positions", 1024)

        return ModelInfo(
            model_id=self._model_id,
            display_name=f"GPT-2 ({self._model_id})",
            architecture="GPT2",
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position,
            num_parameters=num_params,
            supports_kv_cache=True,
            supports_attention_output=True,
            positional_encoding_type="learned",
            normalization_type="layer_norm",
            activation_function="gelu_new",
            dtype="float32",
            device=self._device,
            is_loaded=self._is_loaded,
        )

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenise *text* using the loaded GPT-2 BPE tokenizer.

        Args:
            text: Raw input string.

        Returns:
            Rich tokenization metadata dict.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        self._assert_loaded()
        result = TokenizationService.tokenize(text, self._tokenizer)
        return result.to_event_data()

    # ── Instrumented inference ────────────────────────────────────────────────

    async def run_instrumented_inference(
        self,
        prompt: str,
        config: Dict[str, Any],
        event_callback: Callable[[InferenceEvent], Awaitable[None]],
        session_id: str,
    ) -> Dict[str, Any]:
        """Run a fully instrumented GPT-2 generation loop.

        Every computational step emits one or more ``InferenceEvent`` objects
        via *event_callback*.  Large tensors (attention weights, hidden states)
        are stored in ``tensor_store`` and referenced by UUID rather than being
        inlined in the event payload.

        Args:
            prompt:         Raw user prompt text.
            config:         Generation hyper-parameters:
                            ``max_new_tokens``, ``temperature``, ``top_k``,
                            ``top_p``, ``do_sample``, ``repetition_penalty``.
            event_callback: Async callable that receives each event.
            session_id:     Owning session UUID (passed to tensor_store).

        Returns:
            ``{"generated_text": str, "token_count": int, "duration_ms": float}``
        """
        self._assert_loaded()
        loop = asyncio.get_event_loop()

        # Run the CPU/GPU-bound generation in a thread pool so we do not
        # block the FastAPI event loop.
        result = await loop.run_in_executor(
            None,
            lambda: asyncio.run(
                self._run_inference_sync(prompt, config, event_callback, session_id)
            ),
        )
        return result

    async def _run_inference_sync(
        self,
        prompt: str,
        config: Dict[str, Any],
        event_callback: Callable[[InferenceEvent], Awaitable[None]],
        session_id: str,
    ) -> Dict[str, Any]:
        """The actual async generation loop (runs inside the executor thread).

        We use a fresh asyncio event loop inside the executor thread so we can
        still await event_callback (which sends data over the WebSocket).
        """
        torch = _require_torch()

        profiler = InferenceProfiler()
        seq_counter = _SequenceCounter()

        # Generation parameters with defaults
        max_new_tokens: int = int(config.get("max_new_tokens", 100))
        temperature: float = float(config.get("temperature", 1.0))
        top_k: int = int(config.get("top_k", 50))
        top_p: float = float(config.get("top_p", 0.95))
        do_sample: bool = bool(config.get("do_sample", True))

        level = self._instrumentation_level

        try:
            # ── 1. Session started ────────────────────────────────────────────
            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.SESSION_STARTED,
                    seq=seq_counter.next(),
                    stage="session",
                    data={
                        "model_id": self._model_id,
                        "device": self._device,
                        "instrumentation_level": level,
                        "config": config,
                    },
                )
            )

            # ── 2. Prompt received ────────────────────────────────────────────
            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.PROMPT_RECEIVED,
                    seq=seq_counter.next(),
                    stage="prompt",
                    data={"prompt": prompt, "length": len(prompt)},
                )
            )

            # ── 3. Tokenization ───────────────────────────────────────────────
            t_tok = profiler.start_stage("tokenization")
            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.TOKENIZATION_STARTED,
                    seq=seq_counter.next(),
                    stage="tokenization",
                )
            )

            tok_result = TokenizationService.tokenize(prompt, self._tokenizer)
            tok_dur = profiler.end_stage("tokenization", t_tok)

            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.TOKENIZATION_COMPLETED,
                    seq=seq_counter.next(),
                    stage="tokenization",
                    data={
                        **tok_result.to_event_data(),
                        "duration_ms": tok_dur,
                    },
                )
            )

            # ── 4. Prepare input tensors ──────────────────────────────────────
            t_prep = profiler.start_stage("input_prep")
            input_ids = torch.tensor(
                [tok_result.input_ids], dtype=torch.long, device=self._device
            )
            prep_dur = profiler.end_stage("input_prep", t_prep)

            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.INPUT_TENSOR_PREPARED,
                    seq=seq_counter.next(),
                    stage="input_prep",
                    data={
                        "shape": list(input_ids.shape),
                        "dtype": str(input_ids.dtype),
                        "device": self._device,
                        "duration_ms": prep_dur,
                    },
                )
            )

            # ── 5. Token embeddings (prefill step) ────────────────────────────
            if level in ("standard", "deep"):
                t_emb = profiler.start_stage("embeddings")
                with torch.no_grad():
                    token_embeddings = self._model.transformer.wte(input_ids)
                emb_arr = tensor_to_numpy(token_embeddings)
                emb_ref = tensor_store.store(
                    session_id=session_id,
                    name="token_embeddings",
                    tensor=emb_arr,
                    data_source=DataSource.ACTUAL,
                    description="Token embedding matrix for the prompt [1, seq, hidden]",
                )
                emb_dur = profiler.end_stage("embeddings", t_emb)
                await event_callback(
                    self._make_event(
                        session_id=session_id,
                        event_type=EventType.EMBEDDINGS_GENERATED,
                        seq=seq_counter.next(),
                        stage="embeddings",
                        tensor_references=[emb_ref],
                        data={
                            "shape": list(token_embeddings.shape),
                            "duration_ms": emb_dur,
                        },
                    )
                )

            # ── 6. Autoregressive generation loop ─────────────────────────────
            past_key_values: Optional[Any] = None
            current_input_ids = input_ids
            generated_ids: List[int] = []
            full_sequence_ids = tok_result.input_ids.copy()

            gen_start = time.perf_counter()

            for step in range(max_new_tokens):
                t_step = profiler.start_stage(f"step_{step}")

                # ── Forward pass with attention output ────────────────────────
                hook_data: Dict[str, Any] = {}
                hook_manager = HookManager()

                if level in ("standard", "deep"):
                    # Register hooks on all transformer layers
                    for layer_idx, block in enumerate(self._model.transformer.h):
                        hook_manager.register(
                            block.attn,
                            capture_attention_weights(f"layer_{layer_idx}_attn", hook_data),
                        )
                        if level == "deep":
                            hook_manager.register(
                                block.mlp,
                                capture_module_io(f"layer_{layer_idx}_mlp", hook_data),
                            )
                            hook_manager.register(
                                block.ln_1,
                                capture_module_io(f"layer_{layer_idx}_ln1", hook_data),
                            )
                            hook_manager.register(
                                block.ln_2,
                                capture_module_io(f"layer_{layer_idx}_ln2", hook_data),
                            )

                with torch.no_grad():
                    outputs = self._model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        output_attentions=(level in ("standard", "deep")),
                        use_cache=True,
                    )

                hook_manager.remove_all()

                logits = outputs.logits  # [1, seq, vocab]
                past_key_values = outputs.past_key_values
                # Attention weights: tuple of (batch, heads, seq_q, seq_k) per layer
                all_attentions = outputs.attentions  # None or tuple[Tensor, ...]

                # We care about the last token's logits
                next_token_logits = logits[0, -1, :]  # [vocab]
                next_token_logits_np = tensor_to_numpy(next_token_logits)

                # ── 7. Per-layer events (prefill or decode step) ───────────────
                num_layers = len(self._model.transformer.h)
                info = self.get_info()

                for layer_idx in range(num_layers):
                    if level == "lightweight":
                        break  # skip layer events in lightweight mode

                    t_layer = profiler.start_stage(f"layer_{layer_idx}")
                    await event_callback(
                        self._make_event(
                            session_id=session_id,
                            event_type=EventType.TRANSFORMER_LAYER_STARTED,
                            seq=seq_counter.next(),
                            stage=f"layer_{layer_idx}",
                            layer_index=layer_idx,
                            generation_step=step,
                            data={"layer_index": layer_idx, "step": step},
                        )
                    )

                    # ── Attention ─────────────────────────────────────────────
                    attn_refs: List[TensorReference] = []
                    if all_attentions is not None and layer_idx < len(all_attentions):
                        attn_tensor = all_attentions[layer_idx]  # [1, heads, seq_q, seq_k]
                        attn_np = tensor_to_numpy(attn_tensor[0])  # [heads, seq_q, seq_k]
                        attn_ref = tensor_store.store(
                            session_id=session_id,
                            name=f"step{step}_layer{layer_idx}_attention",
                            tensor=attn_np,
                            data_source=DataSource.ACTUAL,
                            description=f"Attention weights step={step} layer={layer_idx} [heads, seq_q, seq_k]",
                        )
                        attn_refs.append(attn_ref)

                        # Summarise attention patterns
                        attn_mean = float(attn_np.mean())
                        attn_max = float(attn_np.max())
                        attn_entropy = float(
                            -np.sum(
                                np.where(attn_np > 0, attn_np * np.log(attn_np + 1e-9), 0.0)
                            ) / (attn_np.shape[0] * attn_np.shape[1])
                        )

                        await event_callback(
                            self._make_event(
                                session_id=session_id,
                                event_type=EventType.ATTENTION_COMPUTED,
                                seq=seq_counter.next(),
                                stage=f"layer_{layer_idx}",
                                layer_index=layer_idx,
                                generation_step=step,
                                tensor_references=attn_refs,
                                data={
                                    "layer_index": layer_idx,
                                    "step": step,
                                    "num_heads": info.num_heads,
                                    "seq_len_q": attn_np.shape[1],
                                    "seq_len_k": attn_np.shape[2],
                                    "attention_mean": attn_mean,
                                    "attention_max": attn_max,
                                    "attention_entropy": attn_entropy,
                                },
                            )
                        )

                    # ── MLP (deep only via hooks) ──────────────────────────────
                    mlp_key = f"layer_{layer_idx}_mlp"
                    if level == "deep" and mlp_key in hook_data:
                        mlp_out = hook_data[mlp_key].get("output")
                        if mlp_out is not None:
                            mlp_np = tensor_to_numpy(mlp_out)
                            mlp_ref = tensor_store.store(
                                session_id=session_id,
                                name=f"step{step}_layer{layer_idx}_mlp_output",
                                tensor=mlp_np,
                                data_source=DataSource.ACTUAL,
                                description=f"MLP output step={step} layer={layer_idx}",
                            )
                            await event_callback(
                                self._make_event(
                                    session_id=session_id,
                                    event_type=EventType.MLP_COMPUTED,
                                    seq=seq_counter.next(),
                                    stage=f"layer_{layer_idx}",
                                    layer_index=layer_idx,
                                    generation_step=step,
                                    tensor_references=[mlp_ref],
                                    data={
                                        "layer_index": layer_idx,
                                        "step": step,
                                        "output_shape": list(mlp_np.shape),
                                    },
                                )
                            )

                    layer_dur = profiler.end_stage(f"layer_{layer_idx}", t_layer)
                    await event_callback(
                        self._make_event(
                            session_id=session_id,
                            event_type=EventType.TRANSFORMER_LAYER_COMPLETED,
                            seq=seq_counter.next(),
                            stage=f"layer_{layer_idx}",
                            layer_index=layer_idx,
                            generation_step=step,
                            data={
                                "layer_index": layer_idx,
                                "step": step,
                                "duration_ms": layer_dur,
                            },
                        )
                    )

                # ── 8. Logits event ───────────────────────────────────────────
                top50_indices = np.argpartition(next_token_logits_np, -50)[-50:]
                top50_indices = top50_indices[np.argsort(next_token_logits_np[top50_indices])[::-1]]
                top50_logit_values = next_token_logits_np[top50_indices].tolist()
                top50_tokens = [
                    self._tokenizer.decode([int(idx)]) for idx in top50_indices
                ]

                await event_callback(
                    self._make_event(
                        session_id=session_id,
                        event_type=EventType.LOGITS_GENERATED,
                        seq=seq_counter.next(),
                        stage="logits",
                        generation_step=step,
                        data={
                            "step": step,
                            "vocab_size": int(next_token_logits_np.shape[0]),
                            "top50_tokens": top50_tokens,
                            "top50_logits": top50_logit_values,
                            "top50_indices": top50_indices.tolist(),
                        },
                    )
                )

                # ── 9. Probabilities ──────────────────────────────────────────
                probs_np = softmax(next_token_logits_np, temperature=temperature)
                top50_probs = probs_np[top50_indices].tolist()

                await event_callback(
                    self._make_event(
                        session_id=session_id,
                        event_type=EventType.PROBABILITIES_CALCULATED,
                        seq=seq_counter.next(),
                        stage="probabilities",
                        generation_step=step,
                        data={
                            "step": step,
                            "temperature": temperature,
                            "top50_tokens": top50_tokens,
                            "top50_probs": top50_probs,
                            "top50_indices": top50_indices.tolist(),
                        },
                        data_source=DataSource.DERIVED,
                    )
                )

                # ── 10. Token selection ───────────────────────────────────────
                if do_sample:
                    filtered_probs = top_k_filter(probs_np.copy(), top_k)
                    filtered_probs = top_p_filter(filtered_probs, top_p)
                    if filtered_probs.sum() > 0:
                        next_token_id = int(np.random.choice(len(filtered_probs), p=filtered_probs))
                    else:
                        next_token_id = int(np.argmax(probs_np))
                    sampling_method = "top_k_top_p"
                else:
                    next_token_id = int(np.argmax(probs_np))
                    sampling_method = "greedy"

                selected_prob = float(probs_np[next_token_id])
                selected_token_str = self._tokenizer.decode([next_token_id])

                await event_callback(
                    self._make_event(
                        session_id=session_id,
                        event_type=EventType.TOKEN_SELECTED,
                        seq=seq_counter.next(),
                        stage="token_selection",
                        generation_step=step,
                        data={
                            "step": step,
                            "token_id": next_token_id,
                            "token": selected_token_str,
                            "probability": selected_prob,
                            "log_probability": float(np.log(selected_prob + 1e-10)),
                            "sampling_method": sampling_method,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                        },
                        data_source=DataSource.ACTUAL,
                    )
                )

                # ── 11. Accumulate generated tokens ───────────────────────────
                generated_ids.append(next_token_id)
                full_sequence_ids.append(next_token_id)
                generated_text_so_far = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                await event_callback(
                    self._make_event(
                        session_id=session_id,
                        event_type=EventType.TOKEN_GENERATED,
                        seq=seq_counter.next(),
                        stage="generation",
                        generation_step=step,
                        data={
                            "step": step,
                            "token_id": next_token_id,
                            "token": selected_token_str,
                            "token_string": selected_token_str,
                            "probability": float(selected_prob),
                            "tokens_generated_so_far": step + 1,
                            "generated_text_so_far": generated_text_so_far,
                        },
                    )
                )

                # ── 12. KV cache update ───────────────────────────────────────
                if past_key_values is not None and level in ("standard", "deep"):
                    # Report KV cache dimensions (first layer as representative)
                    try:
                        if hasattr(past_key_values, "key_cache"):
                            # DynamicCache object in modern HuggingFace transformers
                            key_tensor = past_key_values.key_cache[0]
                            val_tensor = past_key_values.value_cache[0]
                            key_shape = list(key_tensor.shape)
                            val_shape = list(val_tensor.shape)
                            num_cache_layers = len(past_key_values.key_cache)
                        elif isinstance(past_key_values, (list, tuple)):
                            kv_layer_0 = past_key_values[0]
                            key_shape = list(kv_layer_0[0].shape)
                            val_shape = list(kv_layer_0[1].shape)
                            num_cache_layers = len(past_key_values)
                        else:
                            key_shape = [1, 12, step + 1, 64]
                            val_shape = [1, 12, step + 1, 64]
                            num_cache_layers = 12
                    except Exception:
                        key_shape = [1, 12, step + 1, 64]
                        val_shape = [1, 12, step + 1, 64]
                        num_cache_layers = 12

                    await event_callback(
                        self._make_event(
                            session_id=session_id,
                            event_type=EventType.KV_CACHE_UPDATED,
                            seq=seq_counter.next(),
                            stage="kv_cache",
                            generation_step=step,
                            data={
                                "step": step,
                                "num_layers": num_cache_layers,
                                "key_shape": key_shape,
                                "value_shape": val_shape,
                                "cache_tokens": key_shape[2] if len(key_shape) >= 3 else (step + 1),
                            },
                        )
                    )

                # ── Prepare next input ────────────────────────────────────────
                import torch  # noqa: PLC0415 (already imported above, cheap reference)
                current_input_ids = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=self._device
                )
                profiler.end_stage(f"step_{step}", t_step)

                # ── Stop if EOS ───────────────────────────────────────────────
                if next_token_id == self._tokenizer.eos_token_id:
                    logger.debug("EOS token generated at step %d", step)
                    break

            # ── 13. Generation completed ──────────────────────────────────────
            gen_duration_ms = (time.perf_counter() - gen_start) * 1_000.0
            final_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            summary = profiler.get_summary()

            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.GENERATION_COMPLETED,
                    seq=seq_counter.next(),
                    stage="completed",
                    data={
                        "generated_text": final_text,
                        "prompt": prompt,
                        "full_text": prompt + final_text,
                        "tokens_generated": len(generated_ids),
                        "prompt_tokens": tok_result.token_count,
                        "total_tokens": tok_result.token_count + len(generated_ids),
                        "duration_ms": gen_duration_ms,
                        "tokens_per_second": len(generated_ids) / (gen_duration_ms / 1000.0 + 1e-9),
                        "profiling": summary,
                    },
                )
            )

            return {
                "generated_text": final_text,
                "token_count": len(generated_ids),
                "duration_ms": gen_duration_ms,
            }

        except Exception as exc:
            logger.exception("Inference error in session %s", session_id)
            await event_callback(
                self._make_event(
                    session_id=session_id,
                    event_type=EventType.INFERENCE_ERROR,
                    seq=seq_counter.next(),
                    stage="error",
                    data={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            )
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        """Raise ``RuntimeError`` if the model has not been loaded."""
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                f"Model '{self._model_id}' is not loaded. Call load() first."
            )

    @staticmethod
    def _make_event(
        *,
        session_id: str,
        event_type: EventType,
        seq: int,
        stage: str,
        layer_index: Optional[int] = None,
        generation_step: Optional[int] = None,
        tensor_references: Optional[List[TensorReference]] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        data_source: DataSource = DataSource.ACTUAL,
    ) -> InferenceEvent:
        """Construct an ``InferenceEvent`` with all required boilerplate fields."""
        return InferenceEvent(
            session_id=session_id,
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            sequence_number=seq,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            stage=stage,
            layer_index=layer_index,
            generation_step=generation_step,
            tensor_references=tensor_references or [],
            data=data or {},
            metadata=metadata or {},
            data_source=data_source,
        )


# ── Sequence counter ──────────────────────────────────────────────────────────


class _SequenceCounter:
    """Thread-local monotonically increasing integer counter."""

    def __init__(self) -> None:
        self._value: int = 0

    def next(self) -> int:
        """Return the next sequence number and advance the counter."""
        val = self._value
        self._value += 1
        return val
