import { useState } from 'react'
import { Plus, Equal } from 'lucide-react'
import { Card } from '@/components/ui/Card'

export function OperationInspector() {
  const [activeTab, setActiveTab] = useState<'layernorm' | 'attention' | 'mlp' | 'residual'>('layernorm')

  return (
    <Card
      title="Mathematical Computation & Sublayer Explorer"
      dataSource="actual"
      subtitle="Inspecting Transformer Mathematical Formulations & Tensor Operations"
      headerRight={
        <div className="flex items-center gap-1 font-mono text-xs">
          <button
            onClick={() => setActiveTab('layernorm')}
            className={`px-2.5 py-1 rounded ${activeTab === 'layernorm' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            LayerNorm
          </button>
          <button
            onClick={() => setActiveTab('attention')}
            className={`px-2.5 py-1 rounded ${activeTab === 'attention' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            Attention Equation
          </button>
          <button
            onClick={() => setActiveTab('mlp')}
            className={`px-2.5 py-1 rounded ${activeTab === 'mlp' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            MLP & GELU
          </button>
          <button
            onClick={() => setActiveTab('residual')}
            className={`px-2.5 py-1 rounded ${activeTab === 'residual' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            Residual Pathways
          </button>
        </div>
      }
    >
      <div className="space-y-4 font-mono text-xs">
        {/* LAYERNORM BREAKDOWN */}
        {activeTab === 'layernorm' && (
          <div className="space-y-4">
            <div className="p-4 bg-ns-elevated/70 rounded-xl border border-ns-border space-y-2">
              <span className="text-ns-blue font-semibold text-sm">Layer Normalization (Ba et al., 2016)</span>
              <div className="p-3 bg-ns-surface rounded-lg border border-ns-border text-white text-sm">
                {'y = ((x - μ) / √(σ² + ε)) ⊙ γ + β'}
              </div>
              <p className="text-ns-textMuted text-[11px] leading-relaxed">
                Layer Normalization normalizes activations across the hidden dimension (d = 768) independently for each token position,
                stabilizing gradient dynamics without batch dependency.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Mean & Variance</span>
                <div className="text-white mt-1">{'μ = (1/d) Σ x_i'}</div>
                <div className="text-white mt-0.5">{'σ² = (1/d) Σ (x_i - μ)²'}</div>
                <span className="text-[10px] text-ns-textMuted">Computed per token across 768 features</span>
              </div>
              <div className="p-3 rounded-lg bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Epsilon Parameter</span>
                <div className="text-ns-blue font-bold text-base mt-1">1e-5 (0.00001)</div>
                <span className="text-[10px] text-ns-textMuted">Numerical stabilization constant</span>
              </div>
              <div className="p-3 rounded-lg bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Learnable Parameters</span>
                <div className="text-ns-purple font-bold text-base mt-1">γ, β ∈ ℝ⁷⁶⁸</div>
                <span className="text-[10px] text-ns-textMuted">Scale (weight) and shift (bias) vectors</span>
              </div>
            </div>
          </div>
        )}

        {/* ATTENTION EQUATION */}
        {activeTab === 'attention' && (
          <div className="space-y-4">
            <div className="p-4 bg-ns-elevated/70 rounded-xl border border-ns-border space-y-2">
              <span className="text-ns-purple font-semibold text-sm">Scaled Dot-Product Multi-Head Attention</span>
              <div className="p-3 bg-ns-surface rounded-lg border border-ns-border text-white text-sm">
                {'Attention(Q, K, V) = softmax((Q · Kᵀ / √d_k) + M) · V'}
              </div>
              <p className="text-ns-textMuted text-[11px] leading-relaxed">
                Queries and Keys are projected into 12 heads of dimension d_k = 64. The scaling factor 1/√64 = 0.125 prevents dot-products
                from growing into regions with vanishingly small softmax gradients.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px]">Projection</span>
                <div className="text-white font-bold mt-1">{'Q = x · W_Q'}</div>
                <span className="text-ns-textMuted text-[9px]">[seq, 768] → [12, seq, 64]</span>
              </div>
              <div className="p-3 rounded bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px]">Affinity Score</span>
                <div className="text-ns-blue font-bold mt-1">{'S = Q · Kᵀ / 8'}</div>
                <span className="text-ns-textMuted text-[9px]">Token-to-token similarity</span>
              </div>
              <div className="p-3 rounded bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px]">Causal Mask (M)</span>
                <div className="text-ns-purple font-bold mt-1">{'M_ij = -∞ (i < j)'}</div>
                <span className="text-ns-textMuted text-[9px]">Prevents looking ahead</span>
              </div>
              <div className="p-3 rounded bg-ns-surface border border-ns-border">
                <span className="text-ns-textMuted text-[10px]">Weighted Value</span>
                <div className="text-ns-green font-bold mt-1">{'O = Softmax(S) · V'}</div>
                <span className="text-ns-textMuted text-[9px]">Synthesized context vector</span>
              </div>
            </div>
          </div>
        )}

        {/* FEED-FORWARD MLP & GELU */}
        {activeTab === 'mlp' && (
          <div className="space-y-4">
            <div className="p-4 bg-ns-elevated/70 rounded-xl border border-ns-border space-y-2">
              <span className="text-ns-green font-semibold text-sm">Two-Layer Position-Wise Feed-Forward Network</span>
              <div className="p-3 bg-ns-surface rounded-lg border border-ns-border text-white text-sm">
                {'FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂'}
              </div>
              <p className="text-ns-textMuted text-[11px] leading-relaxed">
                The MLP expands the 768-dimensional residual stream by 4× up to 3,072 dimensions, introduces non-linear activations
                via Gaussian Error Linear Units (GELU), and projects back to 768 dimensions.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-ns-surface border border-ns-border space-y-1">
                <span className="text-ns-textMuted text-[10px] uppercase">GELU Non-Linearity Approximation</span>
                <div className="text-ns-green font-mono text-xs">
                  {'GELU(x) = 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))'}
                </div>
                <p className="text-ns-textMuted text-[10px] pt-1">
                  Unlike ReLU, GELU weighs inputs by their percentile value, providing smooth non-zero gradients for negative values.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-ns-surface border border-ns-border space-y-1">
                <span className="text-ns-textMuted text-[10px] uppercase">Dimensional Transformation</span>
                <div className="text-white font-mono text-xs">
                  {'Input: [seq, 768] → W₁ [768, 3072] → Intermediate [seq, 3072] → W₂ [3072, 768] → Output [seq, 768]'}
                </div>
                <p className="text-ns-textMuted text-[10px] pt-1">
                  Total MLP parameter footprint: ~4.7 million weights per transformer layer.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* RESIDUAL CONNECTIONS */}
        {activeTab === 'residual' && (
          <div className="space-y-4">
            <div className="p-4 bg-ns-elevated/70 rounded-xl border border-ns-border space-y-2">
              <span className="text-ns-blue font-semibold text-sm">Residual Stream Highway</span>
              <div className="p-3 bg-ns-surface rounded-lg border border-ns-border text-white text-sm">
                {'h_{l+1} = h_l + Sublayer(LayerNorm(h_l))'}
              </div>
              <p className="text-ns-textMuted text-[11px] leading-relaxed">
                GPT-2 uses Pre-LayerNorm residual connections. Information flows uninterrupted along the residual stream,
                allowing shallow and deep layers to read and write features additively without vanishing gradients.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-ns-surface border border-ns-border flex items-center justify-around text-center">
              <div>
                <span className="text-ns-textMuted text-[10px]">Residual Stream (In)</span>
                <div className="text-white font-bold text-sm mt-1">{'h_l ∈ ℝ⁷⁶⁸'}</div>
              </div>
              <Plus className="w-5 h-5 text-ns-blue" />
              <div>
                <span className="text-ns-textMuted text-[10px]">Sublayer Output</span>
                <div className="text-ns-purple font-bold text-sm mt-1">{'Δh ∈ ℝ⁷⁶⁸'}</div>
              </div>
              <Equal className="w-5 h-5 text-ns-green" />
              <div>
                <span className="text-ns-textMuted text-[10px]">Updated Stream (Out)</span>
                <div className="text-ns-green font-bold text-sm mt-1">{'h_{l+1} ∈ ℝ⁷⁶⁸'}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
