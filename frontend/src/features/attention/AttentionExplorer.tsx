import { useState } from 'react'
import { Eye, Network, Layers, Sparkles, Filter } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { AttentionHeatmap } from '@/components/charts/AttentionHeatmap'
import { useInferenceStore } from '@/stores/inferenceStore'

export function AttentionExplorer() {
  const layers = useInferenceStore(s => s.layers)
  const tokens = useInferenceStore(s => s.tokens)
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number>(0)
  const [focusedHead, setFocusedHead] = useState<number>(0)
  const [compareMode, setCompareMode] = useState<boolean>(true)

  const activeLayer = layers[selectedLayerIndex]
  const tokenLabels = tokens.map(t => t.token_string)
  const numHeads = 12

  if (!activeLayer || !activeLayer.attentionWeights) {
    return (
      <Card title="Multi-Head Attention Explorer" dataSource="actual" subtitle="Head-by-Head Comparative Visualizer">
        <div className="h-64 flex items-center justify-center text-ns-textMuted font-mono text-xs">
          Attention weights for Layer {selectedLayerIndex} are not yet captured. Run inference to view attention patterns.
        </div>
      </Card>
    )
  }

  // Compute average attention head across all 12 heads
  const avgWeights: number[][] = []
  const seqLen = tokenLabels.length
  if (seqLen > 0 && activeLayer.attentionWeights.length > 0) {
    for (let i = 0; i < seqLen; i++) {
      avgWeights[i] = []
      for (let j = 0; j < seqLen; j++) {
        let sum = 0
        for (let h = 0; h < numHeads; h++) {
          sum += (activeLayer.attentionWeights[h]?.[i]?.[j] || 0)
        }
        avgWeights[i][j] = sum / numHeads
      }
    }
  }

  return (
    <Card
      title="Multi-Head Attention Explorer"
      dataSource="actual"
      subtitle={`Layer ${selectedLayerIndex} • 12 Independent Heads (d_k = 64) • Attention(Q,K,V) = softmax(QKᵀ / √64)V`}
      headerRight={
        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="flex items-center gap-1">
            <span className="text-ns-textMuted">Layer:</span>
            <select
              value={selectedLayerIndex}
              onChange={e => setSelectedLayerIndex(Number(e.target.value))}
              className="bg-ns-elevated border border-ns-border text-white px-2 py-1 rounded text-xs outline-none"
            >
              {Array.from({ length: 12 }, (_, i) => (
                <option key={i} value={i}>
                  Block {i} {layers[i]?.hasAttention ? '•' : ''}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`px-2.5 py-1 rounded ${compareMode ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            {compareMode ? 'Multi-Head Grid' : 'Single Head Focus'}
          </button>
        </div>
      }
    >
      <div className="space-y-4 font-mono text-xs">
        {compareMode ? (
          /* ALL 12 HEADS SIDE-BY-SIDE GRID */
          <div className="space-y-3">
            <div className="text-ns-textMuted text-[11px] flex justify-between">
              <span>All 12 Heads (Parallel Query-Key Attention Maps)</span>
              <span>Click a head to inspect details</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Array.from({ length: numHeads }, (_, hIdx) => {
                const headMatrix = activeLayer.attentionWeights?.[hIdx] || [[]]
                const isSelected = focusedHead === hIdx

                return (
                  <div
                    key={hIdx}
                    onClick={() => {
                      setFocusedHead(hIdx)
                      setCompareMode(false)
                    }}
                    className={`p-2 rounded-lg border cursor-pointer transition-all hover:scale-102 ${
                      isSelected
                        ? 'border-ns-blue bg-ns-blue/10 ring-1 ring-ns-blue'
                        : 'border-ns-border bg-ns-surface/80 hover:border-ns-border/80'
                    }`}
                  >
                    <div className="flex items-center justify-between pb-1 border-b border-ns-border/40 text-[10px]">
                      <span className="font-bold text-white">Head #{hIdx}</span>
                      <span className="text-ns-textMuted text-[9px]">64-dim</span>
                    </div>

                    <div className="mt-2 aspect-square flex items-center justify-center overflow-hidden">
                      {headMatrix.length > 0 && headMatrix[0].length > 0 ? (
                        <div
                          className="grid gap-[1px] w-full h-full p-0.5 bg-ns-elevated rounded"
                          style={{
                            gridTemplateColumns: `repeat(${headMatrix[0].length}, minmax(0, 1fr))`,
                          }}
                        >
                          {headMatrix.map((row, r) =>
                            row.map((val, c) => (
                              <div
                                key={`${r}-${c}`}
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${Math.min(1, val * 1.5)})`,
                                }}
                                className="w-full h-full rounded-[0.5px]"
                              />
                            ))
                          )}
                        </div>
                      ) : (
                        <span className="text-[10px] text-ns-textMuted">Empty</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          /* SINGLE HEAD DEEP FOCUS */
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-white">Focused Head #{focusedHead} Detail</span>
              <div className="flex items-center gap-1">
                {Array.from({ length: numHeads }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => setFocusedHead(i)}
                    className={`px-2 py-0.5 rounded text-[10px] ${
                      focusedHead === i ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'
                    }`}
                  >
                    H{i}
                  </button>
                ))}
              </div>
            </div>

            <div className="p-4 bg-ns-elevated rounded-lg border border-ns-border flex justify-center">
              <AttentionHeatmap
                weights={activeLayer.attentionWeights[focusedHead] || [[]]}
                rowLabels={tokenLabels}
                colLabels={tokenLabels}
                head={focusedHead}
              />
            </div>
          </div>
        )}

        {/* LAYER AVERAGE ATTENTION HEAD */}
        {avgWeights.length > 0 && (
          <div className="pt-3 border-t border-ns-border/50 space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-semibold text-ns-textCode flex items-center gap-1.5">
                <Network className="w-3.5 h-3.5" />
                Aggregated Multi-Head Mean Pattern
              </span>
              <Badge variant="derived" size="sm">Mean across 12 heads</Badge>
            </div>
            <div className="p-3 bg-ns-surface/80 rounded-lg border border-ns-border flex justify-center">
              <AttentionHeatmap
                weights={avgWeights}
                rowLabels={tokenLabels}
                colLabels={tokenLabels}
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

