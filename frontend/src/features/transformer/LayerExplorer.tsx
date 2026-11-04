import { useState } from 'react'
import { Layers, ChevronDown, ChevronRight, CheckCircle2, Loader2, Sparkles } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { LayerProgress } from '@/components/charts/LayerProgress'
import { AttentionHeatmap } from '@/components/charts/AttentionHeatmap'
import { useInferenceStore } from '@/stores/inferenceStore'
import { formatNumber } from '@/utils/format'

export function LayerExplorer() {
  const currentLayer = useInferenceStore(s => s.currentLayer)
  const totalLayers = useInferenceStore(s => s.totalLayers) || 12
  const layers = useInferenceStore(s => s.layers)
  const tokens = useInferenceStore(s => s.tokens)
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number>(0)
  const [selectedHead, setSelectedHead] = useState<number>(0)

  const activeLayer = layers[selectedLayerIndex]
  const tokenLabels = tokens.map(t => t.token_string)

  return (
    <Card
      title="Transformer Layer Progression"
      dataSource="actual"
      subtitle={`Architecture: 12 Blocks • 12 Heads • 768 Hidden Dim (GPT-2)`}
      headerRight={
        <div className="flex items-center gap-2">
          <Badge variant="actual" size="sm">
            Active: Layer {currentLayer}/{totalLayers}
          </Badge>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Layer Blocks Progression Bar */}
        <div className="p-3 bg-ns-elevated border border-ns-border rounded-lg">
          <LayerProgress
            totalLayers={totalLayers}
            currentLayer={currentLayer}
            layerStatuses={layers.map(l => l.status)}
            onLayerClick={idx => setSelectedLayerIndex(idx)}
          />
        </div>

        {/* Selected Layer Deep Dive */}
        <div className="border border-ns-border rounded-lg p-4 bg-ns-surface/50 space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-ns-border/60 pb-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-ns-purple" />
              <span className="font-bold text-white text-sm">Transformer Block {selectedLayerIndex}</span>
              {activeLayer?.status === 'completed' ? (
                <Badge variant="success" size="sm">Completed</Badge>
              ) : activeLayer?.status === 'processing' ? (
                <Badge variant="info" size="sm"><Loader2 className="w-3 h-3 animate-spin mr-1 inline" />Computing</Badge>
              ) : (
                <Badge variant="default" size="sm">Pending</Badge>
              )}
            </div>
            {activeLayer?.durationMs && (
              <span className="text-ns-textMuted text-[11px]">{activeLayer.durationMs.toFixed(1)} ms</span>
            )}
          </div>

          {/* Sub-operations Flow Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-2 text-center text-[11px]">
            <div className={`p-2.5 rounded border ${activeLayer?.hasLn1 ? 'bg-ns-blue/15 border-ns-blue/30 text-ns-blue' : 'bg-ns-elevated border-ns-border text-ns-textMuted'}`}>
              <div className="font-semibold">LayerNorm 1</div>
              <div className="text-[10px] mt-0.5 opacity-80">Pre-Attention Norm</div>
            </div>
            <div className={`p-2.5 rounded border ${activeLayer?.hasAttention ? 'bg-ns-purple/15 border-ns-purple/30 text-ns-purple' : 'bg-ns-elevated border-ns-border text-ns-textMuted'}`}>
              <div className="font-semibold">Self-Attention</div>
              <div className="text-[10px] mt-0.5 opacity-80">12 Multi-Heads (Q,K,V)</div>
            </div>
            <div className={`p-2.5 rounded border ${activeLayer?.hasLn2 ? 'bg-ns-blue/15 border-ns-blue/30 text-ns-blue' : 'bg-ns-elevated border-ns-border text-ns-textMuted'}`}>
              <div className="font-semibold">LayerNorm 2</div>
              <div className="text-[10px] mt-0.5 opacity-80">Pre-MLP Norm</div>
            </div>
            <div className={`p-2.5 rounded border ${activeLayer?.hasMlp ? 'bg-ns-green/15 border-ns-green/30 text-ns-green' : 'bg-ns-elevated border-ns-border text-ns-textMuted'}`}>
              <div className="font-semibold">Feed-Forward MLP</div>
              <div className="text-[10px] mt-0.5 opacity-80">768 → 3072 → 768 (GELU)</div>
            </div>
          </div>

          {/* Attention Weights Matrix Exploration */}
          {activeLayer?.attentionWeights ? (
            <div className="mt-4 pt-3 border-t border-ns-border/60 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-white">Attention Head Heatmap</span>
                <div className="flex items-center gap-1 text-[11px]">
                  <span className="text-ns-textMuted mr-1">Head:</span>
                  {Array.from({ length: 12 }, (_, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedHead(i)}
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        selectedHead === i ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted hover:text-white'
                      }`}
                    >
                      {i}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-ns-elevated border border-ns-border overflow-x-auto flex justify-center">
                <AttentionHeatmap
                  weights={activeLayer.attentionWeights[selectedHead] || [[]]}
                  rowLabels={tokenLabels}
                  colLabels={tokenLabels}
                  head={selectedHead}
                />
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-ns-textMuted text-[11px]">
              Attention weights for Block {selectedLayerIndex} will appear once computed by forward hooks.
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

