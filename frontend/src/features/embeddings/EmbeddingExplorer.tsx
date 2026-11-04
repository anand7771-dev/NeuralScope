import { useEffect, useState } from 'react'
import { Compass, ScatterPlot, GitCommit, Search, Sparkles } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatNumber } from '@/utils/format'
import { api } from '@/services/api'
import { useInferenceStore } from '@/stores/inferenceStore'

interface Point {
  index: number
  label: string
  x: number
  y: number
}

interface PCAResult {
  explained_variance_ratio: number[]
  points: Point[]
}

interface SimilarityResult {
  matrix: number[][]
  labels: string[]
  pairwise_ranked: Array<{
    source_idx: number
    source_label: string
    target_idx: number
    target_label: string
    similarity: number
  }>
}

export function EmbeddingExplorer() {
  const currentSessionId = useInferenceStore(s => s.currentSessionId)
  const tokens = useInferenceStore(s => s.tokens)
  const [pcaData, setPcaData] = useState<PCAResult | null>(null)
  const [simData, setSimData] = useState<SimilarityResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedPoint, setSelectedPoint] = useState<Point | null>(null)
  const [viewMode, setViewMode] = useState<'pca' | 'matrix' | 'pairs'>('pca')

  useEffect(() => {
    if (!currentSessionId) return
    setLoading(true)

    Promise.all([
      api.embeddings.pca<PCAResult>(currentSessionId).catch(() => null),
      api.embeddings.similarity<SimilarityResult>(currentSessionId).catch(() => null),
    ]).then(([pca, sim]) => {
      if (pca) setPcaData(pca)
      if (sim) setSimData(sim)
      setLoading(false)
    })
  }, [currentSessionId, tokens.length])

  if (!currentSessionId) {
    return (
      <Card title="Embedding Space Explorer" dataSource="actual" subtitle="High-Dimensional Vector Analysis">
        <div className="h-64 flex items-center justify-center text-ns-textMuted font-mono text-xs">
          Start an inference session to inspect token embedding projections and cosine similarities.
        </div>
      </Card>
    )
  }

  // Calculate coordinates bounds for SVG scaling
  const points = pcaData?.points || []
  let minX = 0, maxX = 1, minY = 0, maxY = 1
  if (points.length > 0) {
    minX = Math.min(...points.map(p => p.x))
    maxX = Math.max(...points.map(p => p.x))
    minY = Math.min(...points.map(p => p.y))
    maxY = Math.max(...points.map(p => p.y))
  }
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1

  return (
    <Card
      title="Embedding Space & Semantic Geometry"
      dataSource="derived"
      subtitle={`768-Dimensional Hidden Vector Analysis • PCA 2D Projection • Cosine Metric`}
      headerRight={
        <div className="flex items-center gap-1 font-mono text-xs">
          <button
            onClick={() => setViewMode('pca')}
            className={`px-2.5 py-1 rounded ${viewMode === 'pca' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            2D PCA Scatter
          </button>
          <button
            onClick={() => setViewMode('matrix')}
            className={`px-2.5 py-1 rounded ${viewMode === 'matrix' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            Cosine Matrix
          </button>
          <button
            onClick={() => setViewMode('pairs')}
            className={`px-2.5 py-1 rounded ${viewMode === 'pairs' ? 'bg-ns-blue text-white' : 'bg-ns-elevated text-ns-textMuted'}`}
          >
            Top Pairs
          </button>
        </div>
      }
    >
      {loading ? (
        <div className="h-64 flex items-center justify-center gap-2 text-ns-textMuted font-mono text-xs">
          <Spinner size="md" />
          <span>Computing Principal Component Analysis on embedding tensors...</span>
        </div>
      ) : (
        <div className="space-y-4">
          {/* VIEW: PCA 2D SCATTER PLOT */}
          {viewMode === 'pca' && (
            <div className="space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between text-ns-textMuted text-[11px] pb-1 border-b border-ns-border/50">
                <span>
                  Explained Variance: PC1: {( (pcaData?.explained_variance_ratio[0] || 0) * 100 ).toFixed(1)}% | PC2: {( (pcaData?.explained_variance_ratio[1] || 0) * 100 ).toFixed(1)}%
                </span>
                <span>Click any point to inspect token vector</span>
              </div>

              <div className="relative w-full h-80 bg-ns-elevated/70 rounded-xl border border-ns-border p-4 flex items-center justify-center overflow-hidden">
                {/* SVG Coordinate Grid */}
                <svg className="w-full h-full" viewBox="0 0 600 300">
                  {/* Axis lines */}
                  <line x1="50" y1="150" x2="550" y2="150" stroke="#2a2a3a" strokeDasharray="3 3" />
                  <line x1="300" y1="20" x2="300" y2="280" stroke="#2a2a3a" strokeDasharray="3 3" />

                  {/* Scatter Points */}
                  {points.map((pt) => {
                    const normX = 80 + ((pt.x - minX) / spanX) * 440
                    const normY = 250 - ((pt.y - minY) / spanY) * 200
                    const isSelected = selectedPoint?.index === pt.index

                    return (
                      <g
                        key={pt.index}
                        onClick={() => setSelectedPoint(pt)}
                        className="cursor-pointer group"
                      >
                        <circle
                          cx={normX}
                          cy={normY}
                          r={isSelected ? 7 : 5}
                          className={`${
                            isSelected
                              ? 'fill-ns-green stroke-white stroke-2'
                              : 'fill-ns-blue group-hover:fill-ns-purple transition-colors'
                          }`}
                        />
                        <text
                          x={normX + 8}
                          y={normY + 4}
                          className="fill-ns-textMuted group-hover:fill-white text-[10px] font-mono pointer-events-none transition-colors"
                        >
                          "{pt.label}"
                        </text>
                      </g>
                    )
                  })}
                </svg>
              </div>

              {selectedPoint && (
                <div className="p-3 bg-ns-surface border border-ns-border rounded-lg flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-ns-textMuted">Token:</span>
                    <span className="text-white font-bold font-mono">"{selectedPoint.label}"</span>
                    <span className="text-ns-textMuted">Position: #{selectedPoint.index}</span>
                  </div>
                  <div className="text-ns-blue font-mono">
                    Coord: ({selectedPoint.x.toFixed(3)}, {selectedPoint.y.toFixed(3)})
                  </div>
                </div>
              )}
            </div>
          )}

          {/* VIEW: COSINE SIMILARITY MATRIX */}
          {viewMode === 'matrix' && simData && (
            <div className="space-y-3 font-mono text-xs">
              <div className="text-ns-textMuted text-[11px]">
                Pairwise cosine similarity matrix S<sub>ij</sub> ∈ [-1, 1]. Diagonal is 1.0 (self-similarity).
              </div>
              <div className="overflow-x-auto border border-ns-border rounded-lg max-h-80">
                <table className="w-full text-center text-[10px]">
                  <thead className="bg-ns-surface text-ns-textMuted border-b border-ns-border sticky top-0">
                    <tr>
                      <th className="p-2 text-left">Token</th>
                      {simData.labels.map((lbl, idx) => (
                        <th key={idx} className="p-2 max-w-[60px] truncate">"{lbl}"</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ns-border/30">
                    {simData.matrix.map((row, rIdx) => (
                      <tr key={rIdx} className="hover:bg-ns-elevated/40">
                        <td className="p-2 text-left font-semibold text-white bg-ns-surface/80">
                          "{simData.labels[rIdx]}"
                        </td>
                        {row.map((val, cIdx) => {
                          const intensity = Math.max(0, val)
                          const bg = rIdx === cIdx ? '#3b82f644' : `rgba(59, 130, 246, ${intensity * 0.6})`
                          return (
                            <td
                              key={cIdx}
                              style={{ backgroundColor: bg }}
                              title={`${simData.labels[rIdx]} ↔ ${simData.labels[cIdx]}: ${val.toFixed(4)}`}
                              className="p-1.5 font-mono text-white"
                            >
                              {val.toFixed(2)}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* VIEW: TOP SIMILAR PAIRS */}
          {viewMode === 'pairs' && simData && (
            <div className="space-y-2 font-mono text-xs max-h-80 overflow-y-auto">
              <div className="text-ns-textMuted text-[11px] pb-1">
                Ranked token pairs by vector proximity:
              </div>
              <div className="divide-y divide-ns-border/40 border border-ns-border rounded-lg">
                {simData.pairwise_ranked.map((pair, idx) => (
                  <div key={idx} className="p-2.5 flex items-center justify-between hover:bg-ns-elevated/50">
                    <div className="flex items-center gap-3">
                      <span className="text-ns-textMuted text-[10px]">#{idx + 1}</span>
                      <span className="text-white font-semibold">"{pair.source_label}"</span>
                      <span className="text-ns-textMuted">↔</span>
                      <span className="text-white font-semibold">"{pair.target_label}"</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-ns-elevated rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-ns-blue h-full"
                          style={{ width: `${Math.max(0, pair.similarity) * 100}%` }}
                        />
                      </div>
                      <span className="text-ns-blue font-bold min-w-[50px] text-right">
                        {pair.similarity.toFixed(4)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

