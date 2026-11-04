import { clsx } from 'clsx'
import { Tooltip } from './Tooltip'

// ── Types ─────────────────────────────────────────────────────────────────

type DotColor = 'green' | 'yellow' | 'red' | 'gray' | 'blue' | 'purple'

interface StatusDotProps {
  color: DotColor
  pulse?: boolean
  label?: string
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

// ── Color map ─────────────────────────────────────────────────────────────

const COLOR_MAP: Record<DotColor, string> = {
  green:  'bg-ns-green shadow-[0_0_6px_#10b981]',
  yellow: 'bg-ns-orange shadow-[0_0_6px_#f59e0b]',
  red:    'bg-red-500 shadow-[0_0_6px_#ef4444]',
  gray:   'bg-ns-textMuted',
  blue:   'bg-ns-blue shadow-[0_0_6px_#3b82f6]',
  purple: 'bg-ns-purple shadow-[0_0_6px_#8b5cf6]',
}

const SIZE_MAP: Record<NonNullable<StatusDotProps['size']>, string> = {
  sm: 'w-1.5 h-1.5',
  md: 'w-2 h-2',
  lg: 'w-2.5 h-2.5',
}

// ── Component ─────────────────────────────────────────────────────────────

export function StatusDot({
  color,
  pulse = true,
  label,
  className,
  size = 'md',
}: StatusDotProps) {
  const dot = (
    <span
      className={clsx(
        'inline-block rounded-full shrink-0',
        COLOR_MAP[color],
        SIZE_MAP[size],
        pulse && color !== 'gray' && 'animate-pulse-slow',
        className
      )}
      aria-label={label}
    />
  )

  if (label) {
    return (
      <Tooltip content={label}>
        {dot}
      </Tooltip>
    )
  }

  return dot
}
