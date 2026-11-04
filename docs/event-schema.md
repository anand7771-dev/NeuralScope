# Event Schema — NeuralScope v1.0

This document defines the WebSocket event protocol used between the NeuralScope backend and frontend.

---

## Overview

Events are JSON objects emitted over a WebSocket connection at `/ws/{session_id}`.

Events are:
- **Ordered** by `sequence_number` (monotonically increasing per session)
- **Typed** by `event_type` (see `EventType` enum)
- **Annotated** with `data_source` to indicate the provenance of displayed data
- **Referenced** for large tensors (tensors are not embedded inline)

---

## Base Event Schema

```json
{
  "schema_version": "1.0",
  "session_id": "uuid-string",
  "event_id": "uuid-string",
  "event_type": "attention_computed",
  "sequence_number": 42,
  "timestamp": "2024-01-15T10:30:00.123456Z",
  "stage": "transformer",
  "layer_index": 3,
  "generation_step": 1,
  "data_source": "actual",
  "metadata": {},
  "tensor_references": [],
  "data": {}
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Protocol version. Currently `"1.0"` |
| `session_id` | UUID string | Identifies the inference session |
| `event_id` | UUID string | Unique event identifier |
| `event_type` | EventType | Category of this event (see below) |
| `sequence_number` | int | Monotonically increasing. Use for ordering |
| `timestamp` | ISO 8601 | Server-side timestamp when event was emitted |
| `stage` | string | High-level pipeline stage |
| `layer_index` | int? | Transformer layer index (0-based), if applicable |
| `generation_step` | int? | Token generation step (0-based), if applicable |
| `data_source` | DataSource | Provenance of data in this event |
| `metadata` | object | Arbitrary key-value metadata for this event |
| `tensor_references` | TensorReference[] | References to large tensors stored server-side |
| `data` | object | Event-specific payload (see per-event schemas below) |

---

## DataSource Values

| Value | Meaning |
|-------|---------|
| `"actual"` | Direct output of the model computation |
| `"derived"` | Computed from actual model data (e.g., statistics, rankings) |
| `"conceptual"` | Educational visualization, not tied to this specific inference |
| `"simulated"` | Demonstration only. Must be clearly labeled in UI |

---

## TensorReference Schema

Large tensors (attention weights, hidden states, etc.) are stored server-side and referenced by ID.

```json
{
  "ref_id": "uuid-string",
  "name": "attention_weights_layer_3",
  "shape": [1, 12, 15, 15],
  "dtype": "float32",
  "device": "cpu",
  "data_source": "actual",
  "description": "Attention weight matrix for layer 3, all heads",
  "num_elements": 2700,
  "memory_bytes": 10800
}
```

### Retrieving a Tensor

```
GET /api/tensors/{session_id}/{ref_id}
GET /api/tensors/{session_id}/{ref_id}?slice_spec=0,0,:,:   # Head 0 only
GET /api/tensors/{session_id}/{ref_id}/summary              # Stats only
```

---

## Event Types

### Session Events

#### `session_started`
```json
{
  "event_type": "session_started",
  "stage": "session",
  "data": {
    "session_id": "...",
    "model_id": "gpt2",
    "config": { "max_new_tokens": 100, "temperature": 1.0 }
  }
}
```

#### `prompt_received`
```json
{
  "event_type": "prompt_received",
  "stage": "prompt",
  "data": {
    "prompt": "The capital of France is",
    "char_count": 24
  }
}
```

---

### Tokenization Events

#### `tokenization_started`
```json
{
  "event_type": "tokenization_started",
  "stage": "tokenization",
  "data": { "tokenizer_type": "BPE", "encoding": "GPT-2 BPE" }
}
```

#### `tokenization_completed`
```json
{
  "event_type": "tokenization_completed",
  "stage": "tokenization",
  "data_source": "actual",
  "data": {
    "input_ids": [464, 3139, 286, 4881, 318],
    "tokens": ["The", " capital", " of", " France", " is"],
    "token_count": 5,
    "vocab_size": 50257,
    "tokenizer_type": "BPE",
    "bos_token": "<|endoftext|>",
    "bos_token_id": 50256,
    "eos_token": "<|endoftext|>",
    "eos_token_id": 50256,
    "special_tokens": []
  }
}
```

---

### Embedding Events

#### `embeddings_generated`
```json
{
  "event_type": "embeddings_generated",
  "stage": "embeddings",
  "data_source": "actual",
  "tensor_references": [
    {
      "ref_id": "...",
      "name": "token_embeddings",
      "shape": [1, 5, 768],
      "dtype": "float32",
      "description": "Token embedding matrix before positional encoding"
    }
  ],
  "data": {
    "embedding_dim": 768,
    "sequence_length": 5,
    "positional_encoding_type": "learned"
  }
}
```

---

### Input Tensor Events

#### `input_tensor_prepared`
```json
{
  "event_type": "input_tensor_prepared",
  "stage": "input",
  "data_source": "actual",
  "data": {
    "shape": [1, 5],
    "dtype": "int64",
    "device": "cpu",
    "batch_size": 1,
    "sequence_length": 5,
    "attention_mask_shape": [1, 5],
    "has_causal_mask": true
  }
}
```

---

### Transformer Layer Events

#### `transformer_layer_started`
```json
{
  "event_type": "transformer_layer_started",
  "stage": "transformer",
  "layer_index": 0,
  "data": {
    "layer_index": 0,
    "total_layers": 12,
    "input_shape": [1, 5, 768]
  }
}
```

#### `attention_computed`
```json
{
  "event_type": "attention_computed",
  "stage": "transformer",
  "layer_index": 3,
  "data_source": "actual",
  "tensor_references": [
    {
      "ref_id": "...",
      "name": "attention_weights_layer_3",
      "shape": [1, 12, 5, 5],
      "dtype": "float32",
      "description": "Attention probability matrices for all 12 heads in layer 3"
    },
    {
      "ref_id": "...",
      "name": "attention_output_layer_3",
      "shape": [1, 5, 768],
      "dtype": "float32",
      "description": "Attention output after value aggregation and projection"
    }
  ],
  "data": {
    "layer_index": 3,
    "num_heads": 12,
    "head_dim": 64,
    "sequence_length": 5,
    "attention_type": "causal_self_attention",
    "scaling_factor": 0.125,
    "duration_ms": 2.3
  }
}
```

#### `mlp_computed`
```json
{
  "event_type": "mlp_computed",
  "stage": "transformer",
  "layer_index": 3,
  "data_source": "actual",
  "data": {
    "layer_index": 3,
    "input_shape": [1, 5, 768],
    "intermediate_shape": [1, 5, 3072],
    "output_shape": [1, 5, 768],
    "activation_function": "gelu",
    "duration_ms": 1.8
  }
}
```

---

### Output Events

#### `logits_generated`
```json
{
  "event_type": "logits_generated",
  "stage": "output",
  "data_source": "actual",
  "generation_step": 0,
  "data": {
    "vocab_size": 50257,
    "top_tokens": [
      {"rank": 1, "token_id": 6342, "token_string": " Paris", "logit": 15.234},
      {"rank": 2, "token_id": 257, "token_string": " a", "logit": 10.891}
    ]
  }
}
```

#### `probabilities_calculated`
```json
{
  "event_type": "probabilities_calculated",
  "stage": "output",
  "data_source": "derived",
  "generation_step": 0,
  "data": {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "top_tokens": [
      {"rank": 1, "token_id": 6342, "token_string": " Paris", "logit": 15.234, "probability": 0.9823},
      {"rank": 2, "token_id": 257, "token_string": " a", "logit": 10.891, "probability": 0.0097}
    ]
  }
}
```

#### `token_selected`
```json
{
  "event_type": "token_selected",
  "stage": "output",
  "data_source": "actual",
  "generation_step": 0,
  "data": {
    "token_id": 6342,
    "token_string": " Paris",
    "probability": 0.9823,
    "logit": 15.234,
    "rank": 1,
    "sampling_method": "top_p",
    "temperature_applied": 1.0
  }
}
```

#### `token_generated`
```json
{
  "event_type": "token_generated",
  "stage": "generation",
  "data_source": "actual",
  "generation_step": 0,
  "data": {
    "step": 0,
    "token_id": 6342,
    "token_string": " Paris",
    "current_sequence": "The capital of France is Paris",
    "sequence_length": 6,
    "is_eos": false,
    "cumulative_tokens_generated": 1,
    "step_duration_ms": 85.3
  }
}
```

---

### Completion Events

#### `generation_completed`
```json
{
  "event_type": "generation_completed",
  "stage": "generation",
  "data": {
    "generated_text": " Paris, the city of lights.",
    "total_tokens_generated": 7,
    "prompt_tokens": 5,
    "total_tokens": 12,
    "total_duration_ms": 1250.0,
    "time_to_first_token_ms": 320.0,
    "tokens_per_second": 5.6,
    "stop_reason": "eos_token",
    "device": "cpu",
    "model_id": "gpt2"
  }
}
```

#### `inference_error`
```json
{
  "event_type": "inference_error",
  "stage": "error",
  "data": {
    "error_type": "RuntimeError",
    "message": "CUDA out of memory",
    "stage_at_error": "transformer",
    "layer_at_error": 5,
    "recoverable": false
  }
}
```

---

## Backpressure Handling

The WebSocket consumer (frontend) is expected to handle events faster than they are produced. If the frontend disconnects, the session remains in `"paused"` state server-side. Events are recorded in the session store and can be replayed via `GET /api/sessions/{session_id}/events`.

---

## Versioning

The `schema_version` field is included in every event. Future breaking changes will increment the major version. Additive changes increment the minor version. Clients should ignore unknown fields.

