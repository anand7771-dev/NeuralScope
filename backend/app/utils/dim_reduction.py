"""
app/utils/dim_reduction.py
──────────────────────────
Dimensionality reduction and vector similarity algorithms for NeuralScope.
Provides PCA projections, cosine similarity matrices, and vector distance metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA

__all__ = ["compute_pca", "compute_cosine_similarity_matrix", "compute_token_similarities"]


def compute_pca(
    vectors: np.ndarray,
    n_components: int = 2,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute PCA projection of high-dimensional vectors (e.g. 768-dim embeddings).

    Args:
        vectors: 2D array of shape [num_tokens, hidden_dim].
        n_components: 2 or 3 for 2D or 3D scatter plots.
        labels: Optional token strings corresponding to rows.

    Returns:
        Dict with points, explained_variance_ratio, and coordinates.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array for PCA, got shape {vectors.shape}")

    n_samples, n_features = vectors.shape
    if n_samples < 2:
        # Cannot fit PCA on less than 2 points; return zero centered coordinates
        coords = np.zeros((n_samples, n_components), dtype=float)
        explained_variance = [1.0] + [0.0] * (n_components - 1)
    else:
        actual_components = min(n_components, n_samples, n_features)
        pca = PCA(n_components=actual_components)
        transformed = pca.fit_transform(vectors)

        # Pad with zeros if fewer components were fitted than requested
        if actual_components < n_components:
            pad = np.zeros((n_samples, n_components - actual_components))
            coords = np.hstack([transformed, pad])
            explained_variance = list(pca.explained_variance_ratio_) + [0.0] * (n_components - actual_components)
        else:
            coords = transformed
            explained_variance = [float(v) for v in pca.explained_variance_ratio_]

    points = []
    for i in range(n_samples):
        pt: Dict[str, Any] = {
            "index": i,
            "label": labels[i] if labels and i < len(labels) else f"Token {i}",
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        }
        if n_components >= 3:
            pt["z"] = float(coords[i, 2])
        points.append(pt)

    return {
        "n_components": n_components,
        "n_samples": n_samples,
        "explained_variance_ratio": explained_variance,
        "points": points,
    }


def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix between row vectors.

    Cosine similarity: (u · v) / (||u|| * ||v||)

    Args:
        vectors: 2D array of shape [N, D].

    Returns:
        Square symmetric matrix [N, N] with values in [-1, 1].
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Prevent divide by zero for zero vectors
    norms = np.where(norms == 0, 1e-12, norms)
    normalized = vectors / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    # Clip numerical precision artifacts
    return np.clip(similarity_matrix, -1.0, 1.0)


def compute_token_similarities(
    embeddings: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    """Compute complete similarity payload including matrix and pairwise rankings."""
    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    n = len(labels)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "source_idx": i,
                "source_label": labels[i],
                "target_idx": j,
                "target_label": labels[j],
                "similarity": float(sim_matrix[i, j]),
            })

    # Sort pairs by similarity descending
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    return {
        "matrix": sim_matrix.tolist(),
        "labels": labels,
        "pairwise_ranked": pairs,
    }

