# Model Adapters — NeuralScope

This document describes the Model Adapter pattern used in NeuralScope and how to implement adapters for new model architectures.

---

## Overview

A **Model Adapter** is a class that wraps a specific HuggingFace model and exposes a standardized interface for:

1. Loading and unloading the model
2. Extracting model architecture information
3. Running instrumented inference
4. Tokenizing text
5. Capturing intermediate computational states

The adapter abstraction isolates architecture-specific details from the rest of the application, enabling NeuralScope to support multiple model families without changes to the visualization engine.

---

## AbstractModelAdapter Interface

```python
class AbstractModelAdapter(ABC):
    
    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""
    
    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
    
    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return architecture metadata derived from model config."""
    
    @abstractmethod
    async def run_instrumented_inference(
        self,
        prompt: str,
        config: dict,
        event_callback: Callable[[InferenceEvent], Awaitable[None]],
        session_id: str,
    ) -> dict:
        """
        Run full instrumented inference.
        
        Must emit events via event_callback in sequence order.
        Large tensors must be stored via tensor_store, not emitted inline.
        Returns generation result dict.
        """
    
    @abstractmethod
    def tokenize(self, text: str) -> dict:
        """
        Tokenize text and return full metadata.
        Must use the actual tokenizer for this model.
        """
```

---

## ModelInfo Schema

```python
class ModelInfo(BaseModel):
    model_id: str              # e.g., "gpt2"
    display_name: str          # e.g., "GPT-2 (124M)"
    architecture: str          # e.g., "GPT2LMHeadModel"
    num_layers: int            # Number of transformer blocks
    num_heads: int             # Attention heads per layer
    hidden_size: int           # Hidden state dimension
    intermediate_size: int     # FFN intermediate dimension
    vocab_size: int
    max_position_embeddings: int
    num_parameters: int
    supports_kv_cache: bool
    supports_attention_output: bool
    positional_encoding_type: str  # "learned" | "rope" | "sinusoidal" | "alibi"
    normalization_type: str        # "layer_norm" | "rms_norm"
    activation_function: str       # "gelu" | "silu" | "relu" | "swiglu"
    dtype: str                     # "float32" | "float16" | "bfloat16"
    device: str                    # "cpu" | "cuda:0"
    is_loaded: bool
```

---

## GPT-2 Adapter

The reference implementation is `app/adapters/gpt2.py`.

### Architecture-Specific Instrumentation

GPT-2 uses:
- **Positional encoding**: Learned positional embeddings (`wpe`)
- **Normalization**: Layer Normalization (`nn.LayerNorm`)
- **Attention**: Multi-head self-attention with causal mask
- **FFN**: Two-layer MLP with GELU activation
- **Embedding tied to LM head**: `wte` weights are shared with the output projection

GPT-2 exposes attention weights when called with `output_attentions=True`. This returns:
```python
outputs = model(input_ids, output_attentions=True, use_cache=True)
# outputs.attentions: tuple of [batch, heads, seq, seq] tensors, one per layer
# outputs.past_key_values: KV cache state
# outputs.logits: [batch, seq, vocab] logits
```

### Forward Hook Points

The adapter registers hooks at these GPT-2 module paths:
- `model.transformer.h[i]` — full block input/output
- `model.transformer.h[i].attn` — attention output
- `model.transformer.h[i].mlp` — FFN output
- `model.transformer.h[i].ln_1` — pre-attention layer norm
- `model.transformer.h[i].ln_2` — pre-FFN layer norm

---

## Implementing a New Adapter

### 1. Create the Adapter File

```python
# app/adapters/llama.py

from .base import AbstractModelAdapter, ModelInfo
from app.core.events import InferenceEvent, EventType, DataSource

class LlamaAdapter(AbstractModelAdapter):
    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-hf"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
    
    def load(self) -> None:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
        self.model = LlamaForCausalLM.from_pretrained(self.model_id)
        self.model.eval()
    
    def get_info(self) -> ModelInfo:
        cfg = self.model.config
        return ModelInfo(
            model_id=self.model_id,
            display_name=f"Llama-2 ({cfg.num_hidden_layers}L)",
            architecture="LlamaForCausalLM",
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_parameters=sum(p.numel() for p in self.model.parameters()),
            supports_kv_cache=True,
            supports_attention_output=True,
            positional_encoding_type="rope",          # Llama uses RoPE
            normalization_type="rms_norm",             # Llama uses RMSNorm
            activation_function="silu",                # Llama uses SiLU/SwiGLU
            dtype=str(next(self.model.parameters()).dtype),
            device=str(next(self.model.parameters()).device),
            is_loaded=True,
        )
```

### 2. Architecture-Specific Considerations

| Architecture | Positional Encoding | Norm | Attention | FFN |
|-------------|--------------------|----|-----------|-----|
| GPT-2 | Learned embeddings | LayerNorm | Standard MHA | GELU MLP |
| Llama 2 | RoPE | RMSNorm | GQA | SwiGLU |
| Mistral | RoPE | RMSNorm | GQA + Sliding Window | SwiGLU |
| Qwen2 | RoPE | RMSNorm | GQA | SwiGLU |

### 3. Register the Adapter

In `app/main.py`:
```python
from app.adapters.llama import LlamaAdapter

inference_engine.register_adapter("llama-2-7b", LlamaAdapter("meta-llama/Llama-2-7b-hf"))
```

---

## Instrumentation Limitations by Architecture

| Feature | GPT-2 | Llama | Notes |
|---------|-------|-------|-------|
| Attention weights | ✅ Via `output_attentions=True` | ✅ | Standard HF API |
| Hidden states | ✅ Via forward hooks | ✅ | Forward hook |
| KV cache | ✅ `past_key_values` | ✅ | Standard HF API |
| MoE routing | N/A | N/A | Mixtral-specific |
| Flash Attention | ❌ No weight export | ❌ | FlashAttn doesn't expose weights |

> **Important**: When Flash Attention is enabled, attention weight capture is not possible through standard APIs. The adapter must detect this and emit a `conceptual` data source event instead, with a clear explanation.

---

## Testing Adapters

Each adapter should have tests in `tests/test_adapters/test_{name}.py`:

```python
def test_adapter_loads():
    adapter = GPT2Adapter()
    adapter.load()
    assert adapter.get_info().is_loaded

def test_adapter_tokenize():
    adapter = GPT2Adapter()
    adapter.load()
    result = adapter.tokenize("Hello world")
    assert len(result["input_ids"]) > 0
    assert len(result["tokens"]) == len(result["input_ids"])

def test_adapter_inference_emits_events():
    events = []
    async def collect(event):
        events.append(event)
    
    adapter = GPT2Adapter()
    adapter.load()
    asyncio.run(adapter.run_instrumented_inference(
        "Hello", {"max_new_tokens": 5}, collect, "test-session"
    ))
    
    event_types = {e.event_type for e in events}
    assert EventType.TOKENIZATION_COMPLETED in event_types
    assert EventType.GENERATION_COMPLETED in event_types
```

