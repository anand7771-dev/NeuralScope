import { clsx } from 'clsx'
import type { DataSource } from '@/types'

// ── Types ─────────────────────────────────────────────────────────────────

interface BadgeProps {
  variant: DataSource | 'default' | 'info' | 'error' | 'warning' | 'success'
  children: React.ReactNode
  className?: string
  size?: 'sm' | 'md'
}

// ── Variant config ────────────────────────────────────────────────────────

const VARIANT_STYLES: Record<BadgeProps['variant'], string> = {
  actual:     'bg-ns-green/15 text-ns-green border border-ns-green/30',
  derived:    'bg-ns-blue/15 text-ns-blue border border-ns-blue/30',
  conceptual: 'bg-ns-purple/15 text-ns-purple border border-ns-purple/30',
  simulated:  'bg-ns-orange/15 text-ns-orange border border-ns-orange/30',
  default:    'bg-ns-border/50 text-ns-textMuted border border-ns-border',
  info:       'bg-ns-blue/15 text-ns-blue border border-ns-blue/30',
  error:      'bg-red-500/15 text-red-400 border border-red-500/30',
  warning:    'bg-ns-orange/15 text-ns-orange border border-ns-orange/30',
  success:    'bg-ns-green/15 text-ns-green border border-ns-green/30',
}

const SIZE_STYLES: Record<NonNullable<BadgeProps['size']>, string> = {
  sm: 'text-[10px] px-1.5 py-0.5 tracking-wide',
  md: 'text-xs px-2 py-0.5 tracking-wide',
}

// ── Component ─────────────────────────────────────────────────────────────

export function Badge({
  variant,
  children,
  className,
  size = 'sm',
}: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded font-mono font-medium uppercase',
        VARIANT_STYLES[variant],
        SIZE_STYLES[size],
        className
      )}
    >
      {children}
    </span>
  )
}

// ── DataSource badge shorthand ────────────────────────────────────────────

interface DataSourceBadgeProps {
  source: DataSource
  className?: string
}

const DATA_SOURCE_LABELS: Record<DataSource, string> = {
  actual:     'actual',
  derived:    'derived',
  conceptual: 'conceptual',
  simulated:  'simulated',
}

export function DataSourceBadge({ source, className }: DataSourceBadgeProps) {
  return (
    <Badge variant={source} className={className}>
      {DATA_SOURCE_LABELS[source]}
    </Badge>
  )
}
