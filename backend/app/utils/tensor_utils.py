"""
app/utils/tensor_utils.py
──────────────────────────
Utilities for converting, slicing, summarising, and serialising tensors.

All functions operate on plain NumPy arrays to remain framework-agnostic
at the utility layer. PyTorch tensors are converted at the boundary via
``tensor_to_numpy()``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "tensor_to_numpy",
    "numpy_to_list",
    "parse_slice_spec",
    "tensor_summary",
    "serialize_tensor_response",
]


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a ``torch.Tensor`` to a CPU NumPy array.

    The tensor is detached from the autograd graph, moved to CPU, and
    converted without copying when possible.

    Args:
        tensor: A ``torch.Tensor``. Passing a plain ``np.ndarray`` is a
                no-op (returned as-is).

    Returns:
        A NumPy array with the same dtype and data.

    Raises:
        ImportError: If PyTorch is not installed.
        TypeError:   If *tensor* is neither a Tensor nor an ndarray.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. Install it with: pip install torch"
        ) from exc

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor or np.ndarray, got {type(tensor).__name__}"
        )

    return tensor.detach().cpu().numpy()


def numpy_to_list(arr: np.ndarray, max_elements: int = 1000) -> List[Any]:
    """Flatten a NumPy array to a plain Python list, truncating if necessary.

    Large arrays are flattened to a 1-D list and truncated at ``max_elements``
    to keep JSON responses manageable.

    Args:
        arr:          Source array.
        max_elements: Maximum number of scalar values to include.

    Returns:
        A Python list of native Python scalars. When truncated, the list has
        exactly ``max_elements`` items.
    """
    flat = arr.flatten()
    if flat.size > max_elements:
        flat = flat[:max_elements]
    return flat.tolist()


def parse_slice_spec(spec: str, shape: Tuple[int, ...]) -> Tuple:
    """Parse a slice specification string into a tuple of Python slice objects.

    Accepts comma-separated slice notation matching NumPy's indexing syntax::

        "0"         → (0,)
        "0,:,0:5"   → (0, slice(None), slice(0, 5))
        ":,::2"     → (slice(None), slice(None, None, 2))

    Args:
        spec:  Slice specification string.
        shape: Shape of the target array (used for validation, not clamping).

    Returns:
        A tuple suitable for use as ``array[result]``.

    Raises:
        ValueError: If the number of dimensions in *spec* exceeds ``len(shape)``.
    """
    parts = [s.strip() for s in spec.split(",")]
    if len(parts) > len(shape):
        raise ValueError(
            f"Slice has {len(parts)} dimensions but array has {len(shape)}."
        )

    indices: List[Any] = []
    for part in parts:
        if part == "" or part == ":":
            indices.append(slice(None))
        elif re.fullmatch(r"-?\d+", part):
            indices.append(int(part))
        else:
            # Parse start:stop or start:stop:step
            components = part.split(":")
            if len(components) == 2:
                start_str, stop_str = components
                start = int(start_str) if start_str else None
                stop = int(stop_str) if stop_str else None
                indices.append(slice(start, stop))
            elif len(components) == 3:
                start_str, stop_str, step_str = components
                start = int(start_str) if start_str else None
                stop = int(stop_str) if stop_str else None
                step = int(step_str) if step_str else None
                indices.append(slice(start, stop, step))
            else:
                raise ValueError(f"Cannot parse slice component: '{part}'")

    return tuple(indices)


def tensor_summary(arr: np.ndarray) -> Dict[str, Any]:
    """Compute descriptive statistics for a NumPy array.

    Args:
        arr: Source array (any dtype, any shape).

    Returns:
        A dict with keys: ``min``, ``max``, ``mean``, ``std``, ``shape``,
        ``dtype``, ``memory_bytes``, ``num_elements``.
    """
    # Cast to float64 for stable statistics on integer arrays
    float_arr = arr.astype(np.float64)
    return {
        "min": float(np.min(float_arr)),
        "max": float(np.max(float_arr)),
        "mean": float(np.mean(float_arr)),
        "std": float(np.std(float_arr)),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "memory_bytes": arr.nbytes,
        "num_elements": int(arr.size),
    }


def serialize_tensor_response(
    arr: np.ndarray,
    max_elements: int = 1000,
    include_data: bool = True,
) -> Dict[str, Any]:
    """Build a JSON-serialisable dict from a NumPy array.

    Args:
        arr:           Source array.
        max_elements:  Max scalars to include in ``data``.
        include_data:  Set ``False`` to omit the ``data`` field entirely
                       (useful when only summary statistics are needed).

    Returns:
        Dict with keys: ``shape``, ``dtype``, ``summary``, and optionally
        ``data`` (truncated flat list) plus ``truncated`` bool flag.
    """
    summary = tensor_summary(arr)
    response: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "summary": summary,
    }

    if include_data:
        total = int(arr.size)
        response["truncated"] = total > max_elements
        response["total_elements"] = total
        response["data"] = numpy_to_list(arr, max_elements=max_elements)

    return response
