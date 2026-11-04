# NeuralScope

**An Interactive LLM Inference Visualization and Exploration Platform**

NeuralScope makes the internal computational process of Large Language Models (LLMs) observable, understandable, and interactive. Submit a prompt, watch the actual inference process, explore attention heads, inspect intermediate tensors, and understand how tokens are generated — step by step.

---

## What NeuralScope Is

NeuralScope is a combination of:

- **LLM Inference Visualization Platform** — observe the full pipeline in real time
- **Transformer Architecture Explorer** — navigate layers, heads, and mathematical operations
- **Tensor Inspector** — examine actual intermediate tensor values, slices, and distribution histograms
- **Multi-Head Attention Visualizer** — explore all 12 heads side-by-side or aggregated
- **Embedding Space Explorer** — PCA 2D projections and token-to-token cosine similarity matrices
- **Probability Distribution Explorer** — understand token sampling and softmax dynamics
- **Inference Performance Profiler** — measure actual timing and hardware resource usage
- **Educational Platform** — learn how modern language models actually compute representations

---

## Quick Start

### 1. Install Backend Dependencies

```powershell
cd c:\Users\KIIT0001\Desktop\project\neural-scope\backend

# Install PyTorch (CPU — works on all machines)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

> **GPU users**: For NVIDIA CUDA acceleration, use:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Install Frontend Dependencies

```powershell
cd c:\Users\KIIT0001\Desktop\project\neural-scope\frontend
npm install
```

### 3. Run the Backend

```powershell
cd c:\Users\KIIT0001\Desktop\project\neural-scope\backend
python run.py
```
*API: `http://localhost:8000` | Swagger Docs: `http://localhost:8000/docs`*

### 4. Run the Frontend

```powershell
cd c:\Users\KIIT0001\Desktop\project\neural-scope\frontend
npm run dev
```
*Frontend: `http://localhost:5173`*

---

## Features & Visualization Tabs

| Tab | Capability |
|-----|------------|
| 🟢 **Decoding Stream** | Live token-by-token generation with throughput, latency, and duration metrics |
| 🔤 **Tokenization** | BPE subword segmentation, token IDs, vocab positions, and special token badges |
| 🧱 **Layer Hierarchy** | 12-block progression bar, sublayer checks (LN1 → Attn → LN2 → MLP) |
| 🌐 **Multi-Head Attention** | 12 heads side-by-side comparative grid, single head zoom, aggregated mean head |
| 🧭 **Embedding Space** | 2D PCA vector scatter plot, pairwise cosine similarity matrix, ranked semantic proximity pairs |
| 🎛️ **Tensor Inspector** | Universal N-dimensional tensor slicer, paginated 2D cell grid, value distribution histogram |
| 📐 **Math Operations** | Mathematical formulations and live parameters for LayerNorm, Attention, MLP (GELU), Residuals |
| 📊 **Sampling & Logits** | Softmax formula display with temperature, Top-K, Top-P, and top candidate bars |

---

## Roadmap

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| **Phase 1: Foundation** | ✅ Completed | WebSocket streaming, GPT-2 adapter, tokenization, autoregressive generation |
| **Phase 2: Computational Visibility** | ✅ Completed | PCA embedding explorer, cosine similarity, 2D matrix slicer, multi-head grid, math breakdown |
| **Phase 3: Interactive Controls** | ⏳ Next | Pause, step-forward, step-backward, interactive attention head masking |
| **Phase 4: KV Cache & Memory** | ⏳ Planned | Incremental KV-cache growth visualization, prefill vs. decode memory profiling |
| **Phase 5: Multi-Model Expansion** | ⏳ Planned | LLaMA-2 / Mistral model adapters with RoPE and RMSNorm |

---

## License

MIT
