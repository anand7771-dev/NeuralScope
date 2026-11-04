import { clsx } from 'clsx'
import { useState, useRef, type ReactNode } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────

interface TooltipProps {
  content: ReactNode
  children: ReactNode
  placement?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  className?: string
}

// ── Component ─────────────────────────────────────────────────────────────

export function Tooltip({
  content,
  children,
  placement = 'top',
  delay = 400,
  className,
}: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const show = () => {
    timerRef.current = setTimeout(() => setVisible(true), delay)
  }

  const hide = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setVisible(false)
  }

  const placementClasses: Record<NonNullable<TooltipProps['placement']>, string> = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left:   'right-full top-1/2 -translate-y-1/2 mr-2',
    right:  'left-full top-1/2 -translate-y-1/2 ml-2',
  }

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {visible && content != null && (
        <span
          role="tooltip"
          className={clsx(
            'absolute z-50 pointer-events-none',
            'rounded-md px-2.5 py-1.5',
            'bg-ns-elevated border border-ns-border',
            'text-xs text-ns-textPrimary whitespace-nowrap',
            'shadow-lg',
            'animate-fade-in',
            placementClasses[placement],
            className
          )}
        >
          {content}
        </span>
      )}
    </span>
  )
}
