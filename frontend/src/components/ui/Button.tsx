import { clsx } from 'clsx'
import { Loader2 } from 'lucide-react'
import type { ButtonHTMLAttributes } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  fullWidth?: boolean
}

// ── Variant config ────────────────────────────────────────────────────────

const VARIANT_STYLES: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary:
    'bg-ns-blue hover:bg-blue-500 text-white border border-ns-blue/50 ' +
    'shadow-[0_0_12px_rgba(59,130,246,0.25)] hover:shadow-[0_0_20px_rgba(59,130,246,0.4)]',
  secondary:
    'bg-ns-elevated hover:bg-ns-border/50 text-ns-textPrimary border border-ns-border ' +
    'hover:border-ns-blue/40',
  danger:
    'bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/40 ' +
    'hover:border-red-500',
  ghost:
    'bg-transparent hover:bg-ns-elevated text-ns-textMuted hover:text-ns-textPrimary ' +
    'border border-transparent',
  outline:
    'bg-transparent hover:bg-ns-elevated text-ns-textPrimary border border-ns-border ' +
    'hover:border-ns-blue/50',
}

const SIZE_STYLES: Record<NonNullable<ButtonProps['size']>, string> = {
  sm: 'h-7 px-3 text-xs gap-1.5',
  md: 'h-9 px-4 text-sm gap-2',
  lg: 'h-11 px-6 text-base gap-2.5',
}

// ── Component ─────────────────────────────────────────────────────────────

export function Button({
  variant = 'secondary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center rounded-md font-medium',
        'transition-all duration-150 cursor-pointer select-none',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        VARIANT_STYLES[variant],
        SIZE_STYLES[size],
        fullWidth && 'w-full',
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="animate-spin" size={14} />
      ) : (
        leftIcon && <span className="shrink-0">{leftIcon}</span>
      )}
      {children && <span>{children}</span>}
      {!isLoading && rightIcon && (
        <span className="shrink-0">{rightIcon}</span>
      )}
    </button>
  )
}
