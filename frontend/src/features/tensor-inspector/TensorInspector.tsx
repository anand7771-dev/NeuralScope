import { useEffect, useState } from 'react'
import { Database, Grid, BarChart3, ChevronLeft, ChevronRight, Hash, Layers } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { formatBytes, formatShape, formatNumber } from '@/utils/format'
import { api } from '@/services/api'
import { useInferenceStore } from '@/stores/inferenceStore'
import type { TensorReference } from '@/types'

interface HistogramData {
  counts: number[]
  bin_edges: number[]
  min: number
  max: number
  mean: number
  std: number
  total_elements: number
}

interface Matrix2DData {
  total_rows: number
  total_cols: number
  row_offset: number
  col_offset: number
  rows_returned: number
  cols_returned: number
  values: number[][]
}

export function TensorInspector() {
  const currentSessionId = useInferenceStore(s => s.currentSessionId)
  const events = useInferenceStore(s => s.events)

  // Collect all unique tensor references from events
  const tensorRefs: TensorReference[] = []
  const seenRefs = new Set<string>()
  events.forEach(evt => {
    evt.tensor_references?.forEach(ref => {
      if (!seenRefs.has(ref.ref_id)) {
        seenRefs.add(ref.ref_id)
        tensorRefs.push(ref)
      }
    })
  })

  const [selectedRefId, setSelectedRefId] = useState<string>(tensorRefs[0]?.ref_id || '')
  const [histogram, setHistogram] = useState<HistogramData | null>(null)
  const [matrixData, setMatrixData] = useState<Matrix2DData | null>(null)
  const [loading, setLoading] = useState(false)
  const [pageRow, setPageRow] = useState(0)
  const [pageCol, setPageCol] = useState(0)
  const [dimIndices, setDimIndices] = useState('0,0')

  const pageSize = 12

  // Update selectedRefId when new tensors arrive if none selected
  useEffect(() => {
    if (!selectedRefId && tensorRefs.length > 0) {
      setSelectedRefId(tensorRefs[0].ref_id)
    }
  }, [tensorRefs.length, selectedRefId])

  // Fetch histogram and 2D matrix slice
  useEffect(() => {
    if (!currentSessionId || !selectedRefId) return
    setLoading(true)

    Promise.all([
      api.tensors.getHistogram<HistogramData>(currentSessionId, selectedRefId, 16).catch(() => null),
      api.tensors.getMatrix2D<Matrix2DData>(currentSessionId, selectedRefId, {
        row_offset: pageRow,
        col_offset: pageCol,
        max_rows: pageSize,
        max_cols: pageSize,
        dim_indices: dimIndices,
      }).catch(() => null),
    ]).then(([hist, mat]) => {
      if (hist) setHistogram(hist)
      if (mat) setMatrixData(mat)
      setLoading(false)
    })
  }, [currentSessionId, selectedRefId, pageRow, pageCol, dimIndices])

  const selectedRef = tensorRefs.find(r => r.ref_id === selectedRefId)

  if (tensorRefs.length === 0) {
    return (
      <Card title="Universal Tensor Inspector" dataSource="actual" subtitle="Multidimensional Matrix & Value Inspection">
        <div className="h-64 flex items-center justify-center text-ns-textMuted font-mono text-xs">
          No tensor references captured yet. Run an inference session to inspect intermediate tensors.
        </div>
      </Card>
    )
  }

  return (
    <Card
      title="Universal Tensor Inspector"
      dataSource="actual"
      subtitle="Examine multidimensional tensor shapes, values, and memory profiles"
      headerRight={
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-ns-textMuted">Tensors ({tensorRefs.length}):</span>
          <select
            value={selectedRefId}
            onChange={e => {
              setSelectedRefId(e.target.value)
              setPageRow(0)
              setPageCol(0)
            }}
            className="bg-ns-elevated border border-ns-border text-white px-2 py-1 rounded text-xs outline-none"
          >
            {tensorRefs.map(t => (
              <option key={t.ref_id} value={t.ref_id}>
                {t.name} — {formatShape(t.shape)}
              </option>
            ))}
          </select>
        </div>
      }
    >
      <div className="space-y-5 font-mono text-xs">
        {/* Tensor Header Metadata */}
        {selectedRef && (
          <div className="p-3 bg-ns-elevated/70 border border-ns-border rounded-lg grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div>
              <span className="text-ns-textMuted text-[10px] uppercase">Tensor Shape</span>
              <div className="text-ns-blue font-bold text-sm mt-0.5">{formatShape(selectedRef.shape)}</div>
            </div>
            <div>
              <span className="text-ns-textMuted text-[10px] uppercase">Elements</span>
              <div className="text-white font-bold text-sm mt-0.5">{selectedRef.num_elements.toLocaleString()}</div>
            </div>
            <div>
              <span className="text-ns-textMuted text-[10px] uppercase">Data Type</span>
              <div className="text-ns-purple font-bold text-sm mt-0.5">{selectedRef.dtype}</div>
            </div>
            <div>
              <span className="text-ns-textMuted text-[10px] uppercase">Memory Footprint</span>
              <div className="text-ns-green font-bold text-sm mt-0.5">{formatBytes(selectedRef.memory_bytes)}</div>
            </div>
          </div>
        )}

        {/* Value Distribution Histogram */}
        {histogram && histogram.counts.length > 0 && (
          <div className="p-3 bg-ns-surface border border-ns-border rounded-lg space-y-2">
            <div className="flex items-center justify-between text-[11px] pb-1 border-b border-ns-border/40">
              <span className="font-semibold text-white flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-ns-blue" />
                Value Distribution Histogram
              </span>
              <div className="text-ns-textMuted flex items-center gap-3">
                <span>Min: {histogram.min.toFixed(4)}</span>
                <span>Max: {histogram.max.toFixed(4)}</span>
                <span>Mean: {histogram.mean.toFixed(4)}</span>
                <span>Std: {histogram.std.toFixed(4)}</span>
              </div>
            </div>

            <div className="h-24 flex items-end gap-1 pt-2">
              {histogram.counts.map((count, i) => {
                const maxCount = Math.max(...histogram.counts) || 1
                const heightPct = (count / maxCount) * 100
                return (
                  <div
                    key={i}
                    title={`Bin [${histogram.bin_edges[i]} to ${histogram.bin_edges[i + 1]}]: ${count} elements`}
                    className="flex-1 bg-ns-blue/40 hover:bg-ns-blue transition-colors rounded-t cursor-help relative group"
                    style={{ height: `${Math.max(4, heightPct)}%` }}
                  />
                )
              })}
            </div>
          </div>
        )}

        {/* 2D Slice Matrix Grid */}
        <div className="p-3 bg-ns-surface border border-ns-border rounded-lg space-y-3">
          <div className="flex items-center justify-between pb-1 border-b border-ns-border/50 text-[11px]">
            <span className="font-semibold text-white flex items-center gap-1.5">
              <Grid className="w-3.5 h-3.5 text-ns-purple" />
              2D Matrix Slice Navigation
            </span>

            {/* Pagination Controls */}
            {matrixData && (
              <div className="flex items-center gap-2">
                <span className="text-ns-textMuted">
                  Rows {matrixData.row_offset + 1}–{matrixData.row_offset + matrixData.rows_returned} of {matrixData.total_rows}
                </span>
                <button
                  disabled={pageRow === 0}
                  onClick={() => setPageRow(Math.max(0, pageRow - pageSize))}
                  className="p-1 rounded bg-ns-elevated border border-ns-border disabled:opacity-30"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <button
                  disabled={pageRow + pageSize >= matrixData.total_rows}
                  onClick={() => setPageRow(pageRow + pageSize)}
                  className="p-1 rounded bg-ns-elevated border border-ns-border disabled:opacity-30"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* Matrix Cells */}
          {loading ? (
            <div className="h-48 flex items-center justify-center gap-2 text-ns-textMuted">
              <Spinner size="sm" /> Loading slice...
            </div>
          ) : matrixData && matrixData.values.length > 0 ? (
            <div className="overflow-x-auto border border-ns-border/60 rounded">
              <table className="w-full text-center text-[10px] font-mono">
                <thead className="bg-ns-elevated/80 text-ns-textMuted">
                  <tr>
                    <th className="p-1.5 border-r border-ns-border/50">r\c</th>
                    {Array.from({ length: matrixData.cols_returned }, (_, i) => (
                      <th key={i} className="p-1.5 border-r border-ns-border/40">
                        {matrixData.col_offset + i}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-ns-border/30">
                  {matrixData.values.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-ns-elevated/30">
                      <td className="p-1.5 bg-ns-elevated/50 text-ns-textMuted border-r border-ns-border/50 font-bold">
                        {matrixData.row_offset + rIdx}
                      </td>
                      {row.map((val, cIdx) => {
                        const absVal = Math.abs(val)
                        const heat = Math.min(0.7, absVal)
                        return (
                          <td
                            key={cIdx}
                            style={{
                              backgroundColor: val > 0 ? `rgba(59, 130, 246, ${heat})` : `rgba(239, 68, 68, ${heat})`,
                            }}
                            className="p-1.5 border-r border-ns-border/20 text-white truncate max-w-[65px]"
                            title={`Row ${matrixData.row_offset + rIdx}, Col ${matrixData.col_offset + cIdx}: ${val}`}
                          >
                            {val.toFixed(3)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-6 text-center text-ns-textMuted">
              Select tensor to inspect matrix values.
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

