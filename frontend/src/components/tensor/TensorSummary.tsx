import { clsx } from 'clsx'
import { Badge } from '@/components/ui/Badge'
import { formatBytes, formatNumber, formatShape } from '@/utils/format'
import type { DataSource } from '@/types'

interface TensorSummaryProps {
  name: string
  shape: number[]
  dtype: string
  device?: string
  dataSource?: DataSource
  min?: number
  max?: number
  mean?: number
  std?: number
  memoryBytes?: number
  className?: string
}

export function TensorSummary({
  name,
  shape,
  dtype,
  device = 'cpu',
  dataSource = 'actual',
  min,
  max,
  mean,
  std,
  memoryBytes,
  className,
}: TensorSummaryProps) {
  return (
    <div
      className={clsx(
        'rounded-lg border border-ns-border bg-ns-surface/80 p-3 font-mono text-xs text-ns-textPrimary',
        className
      )}
    >
      <div className="flex items-center justify-between pb-2 border-b border-ns-border/60">
        <span className="font-semibold text-ns-textCode">{name}</span>
        <Badge variant={dataSource} size="sm">
          {dataSource}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-2 text-[11px]">
        <div>
          <span className="text-ns-textMuted">Shape: </span>
          <span className="text-ns-blue font-semibold">{formatShape(shape)}</span>
        </div>
        <div>
          <span className="text-ns-textMuted">Dtype: </span>
          <span>{dtype}</span>
        </div>
        <div>
          <span className="text-ns-textMuted">Device: </span>
          <span>{device}</span>
        </div>
        {memoryBytes !== undefined && (
          <div>
            <span className="text-ns-textMuted">Memory: </span>
            <span>{formatBytes(memoryBytes)}</span>
          </div>
        )}
      </div>

      {(min !== undefined || max !== undefined || mean !== undefined || std !== undefined) && (
        <div className="mt-2 pt-2 border-t border-ns-border/40 grid grid-cols-4 gap-1 text-[10px] text-center">
          <div className="bg-ns-elevated/60 rounded p-1">
            <div className="text-ns-textMuted">MIN</div>
            <div className="font-mono text-ns-textPrimary">{min !== undefined ? formatNumber(min, 3) : '—'}</div>
          </div>
          <div className="bg-ns-elevated/60 rounded p-1">
            <div className="text-ns-textMuted">MAX</div>
            <div className="font-mono text-ns-textPrimary">{max !== undefined ? formatNumber(max, 3) : '—'}</div>
          </div>
          <div className="bg-ns-elevated/60 rounded p-1">
            <div className="text-ns-textMuted">MEAN</div>
            <div className="font-mono text-ns-textPrimary">{mean !== undefined ? formatNumber(mean, 3) : '—'}</div>
          </div>
          <div className="bg-ns-elevated/60 rounded p-1">
            <div className="text-ns-textMuted">STD</div>
            <div className="font-mono text-ns-textPrimary">{std !== undefined ? formatNumber(std, 3) : '—'}</div>
          </div>
        </div>
      )}
    </div>
  )
}

