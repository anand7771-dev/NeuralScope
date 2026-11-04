import { clsx } from 'clsx'
import { DataSourceBadge } from './Badge'
import type { DataSource } from '@/types'

// ── Types ─────────────────────────────────────────────────────────────────

interface CardProps {
  title?: string
  subtitle?: string
  dataSource?: DataSource
  className?: string
  headerClassName?: string
  bodyClassName?: string
  children: React.ReactNode
  /** Extra content to render in the header's right slot */
  headerRight?: React.ReactNode
  /** Use a more visually elevated surface */
  elevated?: boolean
  noPadding?: boolean
}

// ── Component ─────────────────────────────────────────────────────────────

export function Card({
  title,
  subtitle,
  dataSource,
  className,
  headerClassName,
  bodyClassName,
  children,
  headerRight,
  elevated = false,
  noPadding = false,
}: CardProps) {
  const hasHeader = title || subtitle || dataSource || headerRight

  return (
    <div
      className={clsx(
        'rounded-lg border border-ns-border',
        elevated ? 'bg-ns-elevated' : 'bg-ns-surface',
        className
      )}
    >
      {hasHeader && (
        <div
          className={clsx(
            'flex items-center justify-between gap-3 px-4 py-3',
            'border-b border-ns-border',
            headerClassName
          )}
        >
          <div className="flex items-center gap-3 min-w-0">
            {(title || subtitle) && (
              <div className="min-w-0">
                {title && (
                  <h3 className="text-sm font-semibold text-ns-textPrimary truncate">
                    {title}
                  </h3>
                )}
                {subtitle && (
                  <p className="text-xs text-ns-textMuted truncate">{subtitle}</p>
                )}
              </div>
            )}
            {dataSource && <DataSourceBadge source={dataSource} />}
          </div>
          {headerRight && (
            <div className="flex items-center gap-2 shrink-0">{headerRight}</div>
          )}
        </div>
      )}

      <div className={clsx(!noPadding && 'p-4', bodyClassName)}>
        {children}
      </div>
    </div>
  )
}
