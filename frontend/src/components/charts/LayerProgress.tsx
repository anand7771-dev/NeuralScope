import { clsx } from 'clsx'
import { motion } from 'framer-motion'
import { Tooltip } from '@/components/ui/Tooltip'
import type { LayerStatus } from '@/types'

// ── Types ─────────────────────────────────────────────────────────────────

interface LayerProgressProps {
  totalLayers: number
  currentLayer: number
  layerStatuses?: LayerStatus[]    // one per layer
  className?: string
  onLayerClick?: (index: number) => void
}

// ── Color helpers ─────────────────────────────────────────────────────────

const statusBg = (status: LayerStatus, isCurrent: boolean): string => {
  if (isCurrent) return 'bg-ns-purple ring-1 ring-ns-purple/60'
  if (status === 'completed') return 'bg-ns-purple/30'
  return 'bg-ns-border/50'
}

// ── Component ─────────────────────────────────────────────────────────────

export function LayerProgress({
  totalLayers,
  currentLayer,
  layerStatuses,
  className,
  onLayerClick,
}: LayerProgressProps) {
  if (totalLayers === 0) {
    return (
      <div className={clsx('text-ns-textMuted text-xs', className)}>
        Layer count unknown
      </div>
    )
  }

  const completedCount = layerStatuses
    ? layerStatuses.filter(s => s === 'completed').length
    : Math.max(currentLayer, 0)

  return (
    <div className={clsx('', className)}>
      {/* Summary row */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-ns-textMuted">
          Layer{' '}
          <span className="text-ns-textPrimary font-mono font-semibold">
            {currentLayer >= 0 ? currentLayer + 1 : '—'}
          </span>
          {' / '}
          <span className="font-mono">{totalLayers}</span>
        </span>
        <span className="text-xs text-ns-textMuted font-mono">
          {((completedCount / totalLayers) * 100).toFixed(0)}%
        </span>
      </div>

      {/* Layer block strip */}
      <div className="flex flex-wrap gap-0.5">
        {Array.from({ length: totalLayers }, (_, i) => {
          const status: LayerStatus = layerStatuses?.[i] ?? (
            i < currentLayer ? 'completed' : i === currentLayer ? 'processing' : 'pending'
          )
          const isCurrent = i === currentLayer

          return (
            <Tooltip key={i} content={`Layer ${i} — ${status}`}>
              <motion.button
                className={clsx(
                  'w-4 h-4 rounded-sm transition-colors duration-200 cursor-pointer',
                  statusBg(status, isCurrent),
                  onLayerClick && 'hover:ring-1 hover:ring-ns-blue/60'
                )}
                onClick={() => onLayerClick?.(i)}
                whileHover={{ scale: 1.2 }}
                animate={
                  isCurrent
                    ? { opacity: [1, 0.5, 1], transition: { repeat: Infinity, duration: 1.2 } }
                    : { opacity: 1 }
                }
                aria-label={`Layer ${i}: ${status}`}
              />
            </Tooltip>
          )
        })}
      </div>

      {/* Overall progress bar */}
      <div className="mt-2 h-0.5 bg-ns-border rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-ns-purple rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${(completedCount / totalLayers) * 100}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
