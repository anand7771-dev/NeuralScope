# Inference Pipeline Documentation

## Overview

This document describes the complete inference pipeline as implemented in NeuralScope. Each stage corresponds to visualization modules in the frontend and event emitters in the backend.

---

## Pipeline Stages

### Stage 1: Prompt Ingestion

**What happens:**
The user's raw text prompt is received by the backend. For Phase 1 (GPT-2), this is a single user message with no system prompt or conversation history. The raw prompt is stored in the session and passed to the tokenizer.

**Events emitted:**
- `prompt_received` — contains raw prompt text and character count

**Future expansion:**
Chat template formatting, system message injection, conversation history concatenation, and special control tokens will be added when multi-turn conversation support is implemented.

---

### Stage 2: Tokenization

**What happens:**
The raw text is converted to a sequence of integer token IDs using the GPT-2 BPE tokenizer.

GPT-2 uses Byte-Pair Encoding (BPE) with a vocabulary of 50,257 tokens. The tokenizer:
1. Encodes the text as UTF-8 bytes
2. Converts bytes to Unicode characters using a fixed byte-to-unicode map
3. Pre-tokenizes by splitting on spaces and punctuation
4. Applies BPE merge rules iteratively to find the optimal subword segmentation

**Actual data captured:**
- Input IDs (the integer token sequence)
- Token strings (decoded individually, e.g., " capital" for token 3139)
- Token count
- Special tokens (BOS/EOS, if present)
- Vocabulary size

**Events emitted:**
- `tokenization_started`
- `tokenization_completed` with full token metadata

**Data source:** `actual` — uses the real GPT-2 tokenizer

---

### Stage 3: Embedding Generation

**What happens:**
Each token ID is mapped to a 768-dimensional embedding vector via a lookup in the token embedding matrix `wte` (shape: [50257, 768]). Additionally, a learned positional embedding is retrieved from `wpe` (shape: [1024, 768]) for each position. The final input representation is the sum of token and positional embeddings.

```python
# Actual GPT-2 embedding computation:
token_embeds = model.transformer.wte(input_ids)       # [batch, seq, 768]
position_ids = torch.arange(seq_len).unsqueeze(0)
pos_embeds = model.transformer.wpe(position_ids)       # [batch, seq, 768]
hidden_states = token_embeds + pos_embeds              # [batch, seq, 768]
hidden_states = model.transformer.drop(hidden_states)  # Dropout (eval: identity)
```

**Actual data captured:**
- Token embedding matrix slice for the input tokens: shape [1, seq, 768]
- Positional embedding slice: shape [1, seq, 768]
- Combined input embedding: shape [1, seq, 768]

**Events emitted:**
- `embeddings_generated` with TensorReferences to actual embedding vectors

**Data source:** `actual` — directly read from `model.transformer.wte` and `model.transformer.wpe`

---

### Stage 4: Input Tensor Preparation

**What happens:**
The input embedding tensor is prepared for the transformer blocks. This includes:
- Verifying shape: [batch=1, seq, hidden=768]
- Creating the attention mask (all 1s for input tokens)
- Setting up causal masking (GPT-2 uses lower-triangular causal masking built into the attention implementation)

**Events emitted:**
- `input_tensor_prepared` with shape, dtype, device metadata

---

### Stage 5: Transformer Blocks (×12 for GPT-2)

Each of GPT-2's 12 transformer blocks contains:

#### 5a. Pre-Attention Layer Normalization (`ln_1`)
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```
Where γ and β are learned parameters, μ is the mean, σ² is the variance, ε = 1e-5.

#### 5b. Self-Attention
```
Q = x · W_Q + b_Q     [seq, 768]
K = x · W_K + b_K     [seq, 768]  
V = x · W_V + b_V     [seq, 768]
```
Split into 12 heads of dimension 64 each:
```
Attention(Q, K, V) = softmax(QK^T / √64 + M) · V
```
Where M is the causal mask (−∞ for future positions).

GPT-2 computes attention for all 12 heads in parallel. The output is concatenated and projected:
```
output = concat(head_1, ..., head_12) · W_O + b_O   [seq, 768]
```

#### 5c. Residual Connection
```
h = x + attention_output
```

#### 5d. Pre-FFN Layer Normalization (`ln_2`)
Same as `ln_1`, but with different learned parameters.

#### 5e. Feed-Forward Network (MLP)
```
fc1_out = h · W_fc1 + b_fc1   [seq, 3072]  (4× expansion)
gelu_out = GELU(fc1_out)
fc2_out = gelu_out · W_fc2 + b_fc2   [seq, 768]  (projection back)
```

GPT-2 uses the original GELU approximation:
```
GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

#### 5f. Residual Connection
```
output = h + mlp_output
```

**Events emitted per layer:**
- `transformer_layer_started`
- `attention_computed` (with TensorReference to attention weights)
- `mlp_computed`
- `normalization_completed`
- `residual_computed`
- `transformer_layer_completed`

---

### Stage 6: Final Layer Normalization

After all 12 transformer blocks:
```python
hidden_states = model.transformer.ln_f(hidden_states)   # [1, seq, 768]
```

---

### Stage 7: Language Model Head (Logits)

The final hidden state at the last token position is projected to vocabulary logits:
```python
logits = model.lm_head(hidden_states[:, -1, :])   # [1, 50257]
```

In GPT-2, `lm_head` shares weights with `wte` (the token embedding matrix). This is called weight tying.

The logit for token `v` represents the unnormalized log-probability of token `v` being the next token.

**Events emitted:**
- `logits_generated` with top-50 actual logit values and token strings

---

### Stage 8: Probability Calculation

```python
# Apply temperature
scaled_logits = logits / temperature

# Softmax to probabilities
probs = torch.softmax(scaled_logits, dim=-1)   # [1, 50257]
```

**Optionally:**
- Top-K: Set all probabilities outside top-K to 0, renormalize
- Top-P (nucleus): Sort by probability, take cumulative sum, truncate at P, renormalize

**Events emitted:**
- `probabilities_calculated` with top-50 actual probabilities after sampling filters applied

**Data source:** `derived` — computed from actual logits

---

### Stage 9: Token Selection

```python
if do_sample:
    next_token = torch.multinomial(probs, num_samples=1)
else:
    next_token = torch.argmax(probs, dim=-1)  # Greedy
```

**Events emitted:**
- `token_selected` with token ID, string, probability, rank, sampling method used

**Data source:** `actual` — this is the actual token the model selected

---

### Stage 10: Autoregressive Continuation

The selected token is appended to the input sequence:
```python
input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
```

The KV cache is updated from the previous forward pass:
```python
past_key_values = outputs.past_key_values
# Shape per layer: [batch, heads, seq, head_dim] for both K and V
```

The model then runs another forward pass with:
- `input_ids` = just the new token
- `past_key_values` = accumulated KV cache

This is the key efficiency of autoregressive generation: instead of re-processing all previous tokens, the KV cache stores their key and value representations from every layer.

**Events emitted per token:**
- `token_generated`
- `kv_cache_updated`
- (All layer events repeated for the new token's forward pass)

---

### Stage 11: Stop Condition

Generation stops when:
1. The EOS token (`<|endoftext|>`, ID 50256) is generated
2. `max_new_tokens` is reached
3. The user disconnects the WebSocket

**Events emitted:**
- `generation_completed` with full statistics

---

### Stage 12: Response Reconstruction

The sequence of generated token IDs is decoded back to text using the tokenizer:
```python
generated_ids = output_ids[:, input_length:]
response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

The final response text is stored in the session and included in the `generation_completed` event.

---

## KV Cache Mechanics

Without caching, processing a sequence of N tokens requires O(N²) attention computations (each token attends to all previous tokens). The KV cache stores key and value tensors from all previous positions, so at step T, only the new token needs to be processed:

```
Step 0: Process [t₀, t₁, t₂, t₃, t₄] → cache K,V for all 5 positions
Step 1: Process [t₅] only, attending to cached K,V[0:5] → cache K,V[5]
Step 2: Process [t₆] only, attending to cached K,V[0:6] → cache K,V[6]
```

GPT-2 KV cache shape per layer:
- Key: [batch=1, heads=12, seq_so_far, head_dim=64]
- Value: [batch=1, heads=12, seq_so_far, head_dim=64]

Memory per layer at seq length N:
```
2 × 1 × 12 × N × 64 × 4 bytes = 6,144 × N bytes
Per all 12 layers: 73,728 × N bytes ≈ 72 KB × N tokens
```

At 100 tokens: ~7.2 MB for GPT-2 KV cache.

