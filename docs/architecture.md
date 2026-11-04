# NeuralScope System Architecture

## Overview

NeuralScope is a full-stack application for visualizing the internal computational processes of Large Language Models during inference. It consists of:

1. **React Frontend** — Interactive visualization environment
2. **FastAPI Backend** — Model execution, instrumentation, and event streaming
3. **Model Adapter Layer** — Architecture-specific model integrations
4. **Computational Visibility Engine (Phase 2)** — Dimensionality reduction (PCA), vector geometry, multidimensional tensor slicing, and multi-head attention comparative analysis.

---

## Design Principles

### 1. Data Accuracy First
Every visualization is grounded in actual computational data. The application categorizes all displayed data as:
- **Actual** — Direct model computation output (e.g., token embeddings, attention weights, raw logits)
- **Derived** — Statistically derived metrics (e.g., PCA coordinates, cosine similarity matrix, softmax probability distribution, value histograms)
- **Conceptual** — Educational explanations and mathematical formulations
- **Simulated** — Clearly-labeled demonstrations

### 2. Event-Driven Architecture
The backend emits structured events over WebSocket as inference runs. The frontend consumes these events and updates visualizations reactively. This decoupling means:
- Visualizations are synchronized with actual execution
- Sessions can be recorded and replayed
- Multiple frontends can observe the same session

### 3. Tensor Reference System
Large tensors are never transmitted inline through WebSocket. Instead:
1. Backend stores tensor in `TensorStore` with a UUID reference
2. Event contains the `TensorReference` metadata (shape, dtype, stats) only
3. Frontend requests specific tensor slices on demand via REST:
   - Slicing: `GET /api/tensors/{session_id}/{ref_id}?slice_spec=0,0,:,:`
   - Paginated 2D view: `GET /api/tensors/{session_id}/{ref_id}/matrix2d?row_offset=0&col_offset=0`
   - Histogram distribution: `GET /api/tensors/{session_id}/{ref_id}/histogram?bins=20`

---

## Module Map

```
backend/app/
├── main.py              — FastAPI app, lifespan management, router registration
├── config.py            — Pydantic settings with environment variable support
│
├── api/
│   ├── routes/
│   │   ├── health.py    — /health hardware telemetry endpoint
│   │   ├── models.py    — /api/models/* endpoints
│   │   ├── sessions.py  — /api/sessions/* endpoints
│   │   ├── tensors.py   — /api/tensors/* endpoints (slices, matrix2d, histograms)
│   │   └── embeddings.py— /api/embeddings/* endpoints (PCA, Cosine similarity)
│   └── websocket.py     — /ws/{session_id} WebSocket endpoint
│
├── core/
│   ├── events.py        — Pydantic schemas for all event types
│   ├── session.py       — InferenceSession, SessionManager
│   └── tensor_store.py  — In-memory tensor storage with reference IDs
│
├── adapters/
│   ├── base.py          — AbstractModelAdapter, ModelInfo
│   └── gpt2.py          — GPT-2 implementation with PyTorch forward hooks
│
├── inference/
│   └── engine.py        — InferenceEngine (adapter registry, run orchestration)
│
├── instrumentation/
│   ├── hooks.py         — PyTorch forward hook utilities
│   └── profiler.py      — Timing and memory profiling
│
├── tokenization/
│   └── service.py       — Tokenization with metadata extraction
│
└── utils/
    ├── dim_reduction.py — PCA projection & Cosine similarity algorithms
    ├── tensor_utils.py  — Tensor serialization, slicing, conversion
    └── math_utils.py    — Softmax, attention math, sampling utilities
```

---

## Computational Visibility Modules (Phase 2)

### 1. Embedding Space Explorer
- **2D PCA Projection**: Projects 768-dimensional token embedding vectors down to 2 principal components to visualize semantic clustering of prompt tokens.
- **Pairwise Cosine Similarity**: Computes the exact dot product similarity between token vectors, with full matrix and ranked proximity pairs.

### 2. Universal Multidimensional Tensor Inspector
- Slices arbitrary N-dimensional tensors (e.g. $[1, 12, \text{seq}, \text{seq}]$) along chosen dimension indices.
- Paginated 2D matrix grid with cell heatmap intensity scaling.
- Live distribution histogram for weights and intermediate activations.

### 3. Multi-Head Attention Explorer
- Inspects all 12 heads side-by-side in a comparative grid.
- Aggregates attention maps into a layer-wide mean attention matrix.
- Single-head focused view for deep query-key affinity inspection.

### 4. Mathematical Operation Decomposition
- Detailed formulations and tensor shapes for Layer Normalization, Scaled Dot-Product Attention, Two-Layer Position-Wise MLP (GELU), and Residual Stream additions.
