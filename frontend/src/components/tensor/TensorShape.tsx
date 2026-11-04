import { clsx } from 'clsx'
import { formatShape } from '@/utils/format'
import { Tooltip } from '@/components/ui/Tooltip'

// ── Types ─────────────────────────────────────────────────────────────────

interface TensorShapeProps {
  shape: number[]
  /** Optional dimension labels, same length as shape */
  dimLabels?: string[]
  className?: string
}

// ── Common dimension label presets ────────────────────────────────────────

/**
 * Heuristically assign dimension labels based on common tensor shapes.
 * Falls back to generic d0, d1, … labels.
 */
function inferDimLabels(shape: number[], provided?: string[]): string[] {
  if (provided && provided.length === shape.length) return provided

  // Common patterns
  if (shape.length === 4) return ['batch', 'heads', 'seq', 'head_dim']
  if (shape.length === 3) return ['batch', 'seq', 'hidden']
  if (shape.length === 2) return ['seq', 'hidden']
  if (shape.length === 1) return ['vocab/seq']
  return shape.map((_, i) => `d${i}`)
}

// ── Component ─────────────────────────────────────────────────────────────

export function TensorShape({ shape, dimLabels, className }: TensorShapeProps) {
  const labels = inferDimLabels(shape, dimLabels)

  return (
    <div className={clsx('inline-flex items-center gap-0.5', className)}>
      <span className="text-ns-textMuted font-mono text-sm">[</span>
      {shape.map((dim, i) => (
        <span key={i} className="inline-flex items-center">
          <Tooltip content={labels[i]} placement="top">
            <span className="font-mono text-sm text-ns-textCode cursor-default hover:text-ns-purple transition-colors">
              {dim.toLocaleString()}
            </span>
          </Tooltip>
          {i < shape.length - 1 && (
            <span className="text-ns-textMuted font-mono text-sm">,&nbsp;</span>
          )}
        </span>
      ))}
      <span className="text-ns-textMuted font-mono text-sm">]</span>
    </div>
  )
}

// ── Compact labeled variant ───────────────────────────────────────────────

interface TensorShapeLabeledProps {
  shape: number[]
  dimLabels?: string[]
  className?: string
}

export function TensorShapeLabeled({ shape, dimLabels, className }: TensorShapeLabeledProps) {
  const labels = inferDimLabels(shape, dimLabels)

  return (
    <div className={clsx('flex flex-wrap gap-2', className)}>
      {shape.map((dim, i) => (
        <div key={i} className="flex flex-col items-center">
          <span className="text-[10px] text-ns-textMuted font-mono uppercase tracking-wider">
            {labels[i]}
          </span>
          <span className="text-sm font-mono font-semibold text-ns-textCode">
            {dim.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Raw string ────────────────────────────────────────────────────────────

export function formatTensorShape(shape: number[]): string {
  return formatShape(shape)
}
