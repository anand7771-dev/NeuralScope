"""
app/instrumentation/hooks.py
─────────────────────────────
PyTorch forward-hook utilities for capturing intermediate tensor values.

``HookManager`` installs and removes hooks in a managed, exception-safe
fashion. Hook factories produce callables that append captured data to a
shared dictionary keyed by layer name.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "HookManager",
    "capture_module_io",
    "capture_attention_weights",
]

logger = logging.getLogger(__name__)


# ── Hook factories ────────────────────────────────────────────────────────────


def capture_module_io(name: str, store: Dict[str, Any]) -> Callable:
    """Return a forward hook that records module inputs and output.

    The hook stores data under ``store[name]`` as::

        {
            "input":  tuple of tensors (detached, cpu),
            "output": tensor or tuple of tensors (detached, cpu),
        }

    Args:
        name:  Unique key for the entry in *store*.
        store: Mutable dict into which the hook writes.

    Returns:
        A PyTorch-compatible forward hook ``(module, input, output) -> None``.
    """

    def hook(module: Any, input: Tuple, output: Any) -> None:  # noqa: A002
        try:
            import torch  # noqa: PLC0415

            def _detach(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return t.detach().cpu()
                return t

            store[name] = {
                "input": tuple(_detach(x) for x in input),
                "output": _detach(output) if not isinstance(output, tuple) else tuple(_detach(x) for x in output),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hook '%s' failed to capture: %s", name, exc)

    return hook


def capture_attention_weights(name: str, store: Dict[str, Any]) -> Callable:
    """Return a forward hook specialised for GPT-2 attention modules.

    GPT-2's ``GPT2Attention.forward()`` returns a tuple whose second element
    is the attention weight tensor when ``output_attentions=True``.

    The hook stores under ``store[name]``::

        {
            "attn_weights": Tensor[batch, heads, seq_q, seq_k]  (or None),
            "output":       Tensor[batch, seq, hidden]
        }

    Args:
        name:  Unique key for the entry in *store*.
        store: Mutable dict into which the hook writes.

    Returns:
        A PyTorch-compatible forward hook ``(module, input, output) -> None``.
    """

    def hook(module: Any, input: Tuple, output: Any) -> None:  # noqa: A002
        try:
            import torch  # noqa: PLC0415

            attn_output = None
            attn_weights = None

            # GPT2Attention returns: (attn_output, present, [attn_weights])
            if isinstance(output, tuple):
                attn_output = output[0]
                # attn_weights is the last non-None tensor in the tuple
                # when output_attentions=True it is at index 2
                for item in reversed(output):
                    if isinstance(item, torch.Tensor) and item is not attn_output:
                        attn_weights = item
                        break

            else:
                attn_output = output

            store[name] = {
                "attn_weights": attn_weights.detach().cpu() if attn_weights is not None else None,
                "output": attn_output.detach().cpu() if attn_output is not None else None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attention hook '%s' failed: %s", name, exc)

    return hook


# ── Hook manager ──────────────────────────────────────────────────────────────


class HookManager:
    """Context-manager that installs and removes PyTorch forward hooks.

    Usage::

        store: dict = {}
        with HookManager() as hm:
            hm.register(model.transformer.h[0].attn,
                        capture_attention_weights("layer_0_attn", store))
            model(input_ids)

        # After `with` block all hooks have been removed
        attn_data = store["layer_0_attn"]

    Hooks are also removed automatically if an exception is raised inside the
    ``with`` block.
    """

    def __init__(self) -> None:
        self._handles: List[Any] = []

    def register(self, module: Any, hook_fn: Callable) -> None:
        """Install *hook_fn* on *module* and track the handle for removal.

        Args:
            module:  Any ``torch.nn.Module`` instance.
            hook_fn: Forward hook callable ``(module, input, output) -> None``.
        """
        handle = module.register_forward_hook(hook_fn)
        self._handles.append(handle)

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove hook handle: %s", exc)
        self._handles.clear()

    # ── Context manager protocol ───────────────────────────────────────────────

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_all()
