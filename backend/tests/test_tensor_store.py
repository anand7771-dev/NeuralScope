"""
backend/tests/test_tensor_store.py
──────────────────────────────────
Unit tests for TensorStore caching, slicing, and memory tracking.
"""
import numpy as np
from app.core.events import DataSource
from app.core.tensor_store import TensorStore


def test_store_and_retrieve():
    store = TensorStore(max_mb=10)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    ref = store.store("sess-1", "test_tensor", arr, DataSource.ACTUAL)
    assert ref.ref_id is not None
    assert ref.shape == [2, 2]
    assert ref.num_elements == 4

    retrieved = store.get(ref.ref_id)
    assert retrieved is not None
    assert np.allclose(arr, retrieved)


def test_slice_retrieval():
    store = TensorStore(max_mb=10)
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    ref = store.store("sess-1", "3d_tensor", arr, DataSource.ACTUAL)
    sliced = store.get_slice(ref.ref_id, "0,:,0:2")
    assert sliced is not None
    assert sliced.shape == (3, 2)
    assert np.allclose(sliced, arr[0, :, 0:2])


def test_session_cleanup():
    store = TensorStore(max_mb=10)
    arr = np.ones((100, 100), dtype=np.float32)

    ref1 = store.store("sess-cleanup", "t1", arr, DataSource.ACTUAL)
    ref2 = store.store("sess-cleanup", "t2", arr, DataSource.ACTUAL)

    assert store.get(ref1.ref_id) is not None
    assert store.get(ref2.ref_id) is not None

    store.clear_session("sess-cleanup")
    assert store.get(ref1.ref_id) is None
    assert store.get(ref2.ref_id) is None

