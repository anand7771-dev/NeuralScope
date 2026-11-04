"""
backend/tests/test_dim_reduction.py
───────────────────────────────────
Unit tests for PCA and Cosine Similarity calculations.
"""
import numpy as np
from app.utils.dim_reduction import compute_pca, compute_cosine_similarity_matrix, compute_token_similarities


def test_pca_computation():
    # 5 sample vectors in 10-dimensional space
    np.random.seed(42)
    vectors = np.random.randn(5, 10)
    labels = ["TokenA", "TokenB", "TokenC", "TokenD", "TokenE"]

    pca_result = compute_pca(vectors, n_components=2, labels=labels)
    assert pca_result["n_components"] == 2
    assert pca_result["n_samples"] == 5
    assert len(pca_result["points"]) == 5
    assert len(pca_result["explained_variance_ratio"]) == 2
    assert pca_result["points"][0]["label"] == "TokenA"
    assert "x" in pca_result["points"][0]
    assert "y" in pca_result["points"][0]


def test_cosine_similarity():
    # Identical vectors should have similarity 1.0
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])
    # Orthogonal vector
    v3 = np.array([-2.0, 1.0, 0.0])

    mat = np.stack([v1, v2, v3])
    sim = compute_cosine_similarity_matrix(mat)

    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[0, 1], 1.0)
    assert np.isclose(sim[0, 2], 0.0)


def test_token_similarities_payload():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = ["dog", "cat", "puppy"]

    payload = compute_token_similarities(vectors, labels)
    assert len(payload["labels"]) == 3
    assert len(payload["pairwise_ranked"]) == 3
    # First ranked pair should have highest cosine similarity
    assert payload["pairwise_ranked"][0]["similarity"] >= payload["pairwise_ranked"][1]["similarity"]

