import { clsx } from 'clsx'

// ── Types ─────────────────────────────────────────────────────────────────

interface SpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg'
  className?: string
  label?: string
}

const SIZE_MAP: Record<NonNullable<SpinnerProps['size']>, number> = {
  xs: 12,
  sm: 16,
  md: 24,
  lg: 36,
}

// ── Component ─────────────────────────────────────────────────────────────

export function Spinner({ size = 'md', className, label }: SpinnerProps) {
  const px = SIZE_MAP[size]

  return (
    <span
      role="status"
      aria-label={label ?? 'Loading'}
      className={clsx('inline-flex items-center justify-center', className)}
    >
      <svg
        width={px}
        height={px}
        viewBox="0 0 24 24"
        fill="none"
        className="animate-spin"
        aria-hidden="true"
      >
        {/* Background track */}
        <circle
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="2.5"
          className="text-ns-border"
        />
        {/* Spinning arc */}
        <path
          d="M12 2a10 10 0 0 1 10 10"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          className="text-ns-blue"
        />
      </svg>
      {label && (
        <span className="ml-2 text-sm text-ns-textMuted">{label}</span>
      )}
    </span>
  )
}
