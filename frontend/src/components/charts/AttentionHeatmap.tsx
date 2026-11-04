import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { clsx } from 'clsx'
import { blueScale, formatNumber } from '@/utils/format'

// ── Types ─────────────────────────────────────────────────────────────────

interface AttentionHeatmapProps {
  weights: number[][]          // [rows][cols] — attention weights
  rowLabels: string[]          // token labels for rows (queries)
  colLabels: string[]          // token labels for cols (keys)
  selectedRow?: number         // highlight this row index
  onCellClick?: (row: number, col: number, value: number) => void
  className?: string
  title?: string
  head?: number                // which attention head this represents
}

interface TooltipState {
  visible: boolean
  x: number
  y: number
  row: number
  col: number
  value: number
}

// ── Constants ─────────────────────────────────────────────────────────────

const CELL_SIZE    = 28
const LABEL_WIDTH  = 72
const LABEL_HEIGHT = 60
const MARGIN       = { top: 8, right: 8, bottom: 8, left: 8 }

// ── Component ─────────────────────────────────────────────────────────────

export function AttentionHeatmap({
  weights,
  rowLabels,
  colLabels,
  selectedRow,
  onCellClick,
  className,
  title,
  head,
}: AttentionHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false, x: 0, y: 0, row: 0, col: 0, value: 0,
  })

  const numRows = weights.length
  const numCols = weights[0]?.length ?? 0

  const svgWidth  = MARGIN.left + LABEL_WIDTH + numCols * CELL_SIZE + MARGIN.right
  const svgHeight = MARGIN.top  + LABEL_HEIGHT + numRows * CELL_SIZE + MARGIN.bottom

  const renderHeatmap = useCallback(() => {
    if (!svgRef.current || numRows === 0 || numCols === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left + LABEL_WIDTH},${MARGIN.top + LABEL_HEIGHT})`)

    // ── Cells ───────────────────────────────────────────────────────────
    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        const val = weights[r][c] ?? 0
        const isHighlightRow = selectedRow === r

        g.append('rect')
          .attr('x', c * CELL_SIZE)
          .attr('y', r * CELL_SIZE)
          .attr('width',  CELL_SIZE - 1)
          .attr('height', CELL_SIZE - 1)
          .attr('rx', 2)
          .attr('fill', blueScale(val))
          .attr('stroke', isHighlightRow ? '#f59e0b' : 'transparent')
          .attr('stroke-width', isHighlightRow ? 1.5 : 0)
          .style('cursor', onCellClick ? 'pointer' : 'default')
          .on('mouseenter', function (event: MouseEvent) {
            d3.select(this).attr('stroke', '#e2e8f0').attr('stroke-width', 1.5)
            const rect = svgRef.current!.getBoundingClientRect()
            setTooltip({
              visible: true,
              x: event.clientX - rect.left + 8,
              y: event.clientY - rect.top - 32,
              row: r,
              col: c,
              value: val,
            })
          })
          .on('mouseleave', function () {
            d3.select(this)
              .attr('stroke', isHighlightRow ? '#f59e0b' : 'transparent')
              .attr('stroke-width', isHighlightRow ? 1.5 : 0)
            setTooltip(prev => ({ ...prev, visible: false }))
          })
          .on('click', () => onCellClick?.(r, c, val))
      }
    }

    // ── Column labels (rotated) ──────────────────────────────────────────
    colLabels.forEach((label, c) => {
      g.append('text')
        .attr('x', c * CELL_SIZE + CELL_SIZE / 2)
        .attr('y', -4)
        .attr('text-anchor', 'start')
        .attr('transform', `rotate(-45, ${c * CELL_SIZE + CELL_SIZE / 2}, -4)`)
        .attr('fill', '#64748b')
        .attr('font-size', '10')
        .attr('font-family', '"JetBrains Mono", monospace')
        .text(label.length > 6 ? label.slice(0, 6) + '…' : label)
    })

    // ── Row labels ───────────────────────────────────────────────────────
    rowLabels.forEach((label, r) => {
      const isHighlight = selectedRow === r
      g.append('text')
        .attr('x', -6)
        .attr('y', r * CELL_SIZE + CELL_SIZE / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', isHighlight ? '#f59e0b' : '#64748b')
        .attr('font-size', '10')
        .attr('font-family', '"JetBrains Mono", monospace')
        .attr('font-weight', isHighlight ? '600' : '400')
        .text(label.length > 8 ? label.slice(0, 8) + '…' : label)
    })
  }, [weights, rowLabels, colLabels, selectedRow, numRows, numCols, onCellClick])

  useEffect(() => {
    renderHeatmap()
  }, [renderHeatmap])

  if (numRows === 0 || numCols === 0) {
    return (
      <div className={clsx(
        'flex items-center justify-center h-32 text-ns-textMuted text-sm',
        className
      )}>
        No attention data available
      </div>
    )
  }

  return (
    <div className={clsx('relative overflow-auto', className)}>
      {title && (
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-semibold text-ns-textPrimary">{title}</span>
          {head !== undefined && (
            <span className="text-xs text-ns-textMuted font-mono">Head {head}</span>
          )}
        </div>
      )}

      <svg
        ref={svgRef}
        width={svgWidth}
        height={svgHeight}
        className="block"
      />

      {/* Hover tooltip */}
      {tooltip.visible && (
        <div
          className={clsx(
            'absolute z-20 pointer-events-none',
            'bg-ns-elevated border border-ns-border rounded-md px-2.5 py-1.5',
            'text-xs font-mono text-ns-textPrimary shadow-lg'
          )}
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <span className="text-ns-textMuted">
            [{tooltip.row}, {tooltip.col}]
          </span>{' '}
          <span className="text-ns-blue font-semibold">
            {formatNumber(tooltip.value, 4)}
          </span>
          <div className="text-ns-textMuted">
            {rowLabels[tooltip.row]} → {colLabels[tooltip.col]}
          </div>
        </div>
      )}
    </div>
  )
}
