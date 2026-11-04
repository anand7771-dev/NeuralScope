import { clsx } from 'clsx'
import { Tooltip } from '@/components/ui/Tooltip'
import { formatNumber } from '@/utils/format'
import type { TopToken } from '@/types'

// ── Types ─────────────────────────────────────────────────────────────────

interface ProbabilityBarProps {
  tokens: TopToken[]
  maxItems?: number
  selectedTokenId?: number
  className?: string
  showLogits?: boolean
}

// ── Color by rank ─────────────────────────────────────────────────────────

const rankColor = (rank: number, selected: boolean): string => {
  if (selected) return 'bg-ns-green'
  if (rank === 1) return 'bg-ns-blue'
  if (rank === 2) return 'bg-ns-purple'
  if (rank <= 5) return 'bg-purple-700'
  return 'bg-ns-border'
}

const rankTextColor = (rank: number, selected: boolean): string => {
  if (selected) return 'text-ns-green'
  if (rank === 1) return 'text-ns-blue'
  if (rank === 2) return 'text-ns-purple'
  if (rank <= 5) return 'text-purple-400'
  return 'text-ns-textMuted'
}

// ── Component ─────────────────────────────────────────────────────────────

export function ProbabilityBar({
  tokens,
  maxItems = 20,
  selectedTokenId,
  className,
  showLogits = false,
}: ProbabilityBarProps) {
  const displayed = tokens.slice(0, maxItems)
  const maxProb = Math.max(...displayed.map(t => t.probability), 0.001)

  if (displayed.length === 0) {
    return (
      <div className={clsx('text-ns-textMuted text-sm text-center py-6', className)}>
        No token data available
      </div>
    )
  }

  return (
    <div className={clsx('space-y-1', className)}>
      {displayed.map(token => {
        const isSelected = token.token_id === selectedTokenId
        const pct = (token.probability / maxProb) * 100

        return (
          <Tooltip
            key={token.token_id}
            content={
              <span className="font-mono text-xs">
                ID: {token.token_id} | p={formatNumber(token.probability, 4)}
                {showLogits && ` | logit=${formatNumber(token.logit, 3)}`}
              </span>
            }
          >
            <div
              className={clsx(
                'flex items-center gap-2 group cursor-default',
                isSelected && 'opacity-100',
                !isSelected && 'opacity-80 hover:opacity-100'
              )}
            >
              {/* Rank */}
              <span
                className={clsx(
                  'w-5 text-right font-mono text-[10px] shrink-0',
                  rankTextColor(token.rank, isSelected)
                )}
              >
                {token.rank}
              </span>

              {/* Token label */}
              <span
                className={clsx(
                  'w-24 text-xs font-mono truncate shrink-0',
                  isSelected
                    ? 'text-ns-green font-semibold'
                    : 'text-ns-textPrimary'
                )}
                title={token.token_string}
              >
                {JSON.stringify(token.token_string)}
              </span>

              {/* Bar */}
              <div className="flex-1 h-4 bg-ns-elevated rounded-sm overflow-hidden relative">
                <div
                  className={clsx(
                    'h-full rounded-sm transition-all duration-300',
                    rankColor(token.rank, isSelected)
                  )}
                  style={{ width: `${pct}%` }}
                />
                {isSelected && (
                  <div className="absolute inset-y-0 right-1 flex items-center">
                    <span className="text-[9px] text-ns-green font-mono">✓ selected</span>
                  </div>
                )}
              </div>

              {/* Probability */}
              <span
                className={clsx(
                  'w-14 text-right font-mono text-xs shrink-0',
                  isSelected ? 'text-ns-green font-semibold' : 'text-ns-textMuted'
                )}
              >
                {(token.probability * 100).toFixed(2)}%
              </span>
            </div>
          </Tooltip>
        )
      })}
    </div>
  )
}
