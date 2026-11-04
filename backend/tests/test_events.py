"""
backend/tests/test_events.py
────────────────────────────
Unit tests for event schemas and validation.
"""
import json
from app.core.events import DataSource, EventType, InferenceEvent, TensorReference


def test_event_serialization():
    tensor_ref = TensorReference(
        ref_id="t-123",
        name="attention_weights",
        shape=[1, 12, 10, 10],
        dtype="float32",
        device="cpu",
        data_source=DataSource.ACTUAL,
        description="Head attention weights",
        num_elements=1200,
        memory_bytes=4800,
    )

    event = InferenceEvent(
        session_id="session-test",
        event_id="evt-1",
        event_type=EventType.ATTENTION_COMPUTED,
        sequence_number=1,
        timestamp="2026-11-05T00:00:00Z",
        stage="transformer",
        layer_index=2,
        data={"num_heads": 12},
        tensor_references=[tensor_ref],
        data_source=DataSource.ACTUAL,
    )

    dumped = event.model_dump_json()
    assert "session-test" in dumped
    assert "attention_computed" in dumped

    parsed = InferenceEvent.model_validate_json(dumped)
    assert parsed.session_id == "session-test"
    assert parsed.event_type == EventType.ATTENTION_COMPUTED
    assert len(parsed.tensor_references) == 1
    assert parsed.tensor_references[0].shape == [1, 12, 10, 10]

