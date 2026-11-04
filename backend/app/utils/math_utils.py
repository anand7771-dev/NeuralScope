"""
app/utils/math_utils.py
────────────────────────
Numerical utilities used throughout the inference pipeline.

All functions operate on NumPy arrays and are intentionally framework-agnostic
so they can be tested without PyTorch installed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "softmax",
    "top_k_filter",
    "top_p_filter",
    "compute_attention",
]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature scaling.

    The computation uses the log-sum-exp trick to prevent overflow::

        softmax(x / T) = exp(x/T - max(x/T)) / sum(exp(x/T - max(x/T)))

    Args:
        logits:      1-D or N-D float array. Softmax is applied along the
                     **last** axis.
        temperature: Scaling factor. Values < 1.0 sharpen the distribution;
                     values > 1.0 flatten it.

    Returns:
        Array of the same shape with values in ``[0, 1]`` summing to 1 along
        the last axis.

    Raises:
        ValueError: If ``temperature`` is non-positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")

    scaled = logits / temperature
    # Subtract row-wise max for numerical stability
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero-out all but the top-*k* probability entries.

    Probabilities outside the top-*k* are set to zero; the array is then
    **renormalised** so values still sum to 1.

    Args:
        probs: 1-D probability array (non-negative, ideally summing to 1).
        k:     Number of top entries to keep.  ``k <= 0`` disables filtering.

    Returns:
        Renormalised 1-D array of the same shape.
    """
    if k <= 0 or k >= len(probs):
        return probs

    filtered = np.zeros_like(probs)
    top_indices = np.argpartition(probs, -k)[-k:]
    filtered[top_indices] = probs[top_indices]

    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus (top-*p*) filtering: keep the smallest set whose cumulative
    probability ≥ *p* and zero out the rest.

    Args:
        probs: 1-D probability array (non-negative, summing to ~1).
        p:     Nucleus probability mass in ``(0, 1]``.
                ``p = 1.0`` keeps all tokens.

    Returns:
        Renormalised 1-D array of the same shape.
    """
    if p >= 1.0:
        return probs

    # Sort descending to find nucleus boundary
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    # Include all tokens until cumulative mass exceeds p
    cutoff_mask = cumulative - sorted_probs > p  # shift so that the token that pushes us over is kept
    sorted_probs[cutoff_mask] = 0.0

    # Scatter back to original ordering
    filtered = np.zeros_like(probs)
    filtered[sorted_indices] = sorted_probs

    total = filtered.sum()
    if total > 0:
        filtered /= total
    return filtered


def compute_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention (reference implementation).

    Computes::

        scores = Q @ K^T / sqrt(d_k)   [+ mask if provided]
        weights = softmax(scores)
        output  = weights @ V

    Args:
        Q:     Query matrix  ``[..., seq_q, d_k]``
        K:     Key matrix    ``[..., seq_k, d_k]``
        V:     Value matrix  ``[..., seq_k, d_v]``
        mask:  Optional additive mask ``[..., seq_q, seq_k]``.
               Typically ``-1e9`` for masked positions.
        scale: Custom scale factor. Defaults to ``1 / sqrt(d_k)``.

    Returns:
        Tuple of ``(attention_weights, output)`` where:
        - ``attention_weights`` has shape ``[..., seq_q, seq_k]``
        - ``output`` has shape ``[..., seq_q, d_v]``
    """
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # Scaled dot product: [..., seq_q, seq_k]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) * scale

    if mask is not None:
        scores = scores + mask

    # Softmax over last axis (key dimension)
    weights = softmax(scores, temperature=1.0)

    # Weighted sum over values: [..., seq_q, d_v]
    output = np.matmul(weights, V)

    return weights, output
