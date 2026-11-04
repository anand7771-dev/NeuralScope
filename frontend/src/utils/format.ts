// ── Utility helpers ───────────────────────────────────────────────────────

/**
 * Format a number to a fixed number of decimal places with commas.
 */
export const formatNumber = (n: number, decimals = 4): string => {
  if (!isFinite(n)) return 'N/A'
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/**
 * Human-readable byte count (B → KB → MB → GB).
 */
export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

/**
 * Human-readable duration from milliseconds.
 */
export const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`
  const mins = Math.floor(ms / 60_000)
  const secs = ((ms % 60_000) / 1000).toFixed(0)
  return `${mins}m ${secs}s`
}

/**
 * Pretty-print a tensor shape array.
 * @example formatShape([1, 12, 768]) → "[1, 12, 768]"
 */
export const formatShape = (shape: number[]): string =>
  `[${shape.join(', ')}]`

/**
 * Tokens per second, e.g. "23.4 tok/s".
 */
export const formatTokensPerSecond = (tps: number): string =>
  `${tps.toFixed(1)} tok/s`

/**
 * Deterministic color string from a token ID.
 * Returns a Tailwind-compatible hex color string.
 */
export const tokenToColor = (tokenId: number): string => {
  // Palette of visually distinct, dark-theme-friendly colors
  const palette = [
    '#1e3a5f', // dark blue
    '#2d1b69', // dark purple
    '#1a4731', // dark green
    '#4a1942', // dark magenta
    '#3d2b00', // dark amber
    '#1f3a3a', // dark teal
    '#3a1f1f', // dark red
    '#1a2a3a', // dark navy
    '#2a3a1a', // dark olive
    '#3a2a1a', // dark brown
    '#1a3a2a', // dark emerald
    '#2a1a3a', // dark indigo
  ]
  return palette[tokenId % palette.length]
}

/**
 * Deterministic foreground color for a token chip background.
 */
export const tokenToTextColor = (tokenId: number): string => {
  const palette = [
    '#93c5fd', // blue
    '#c4b5fd', // purple
    '#6ee7b7', // green
    '#f9a8d4', // pink
    '#fcd34d', // amber
    '#99f6e4', // teal
    '#fca5a5', // red
    '#93c5fd', // navy → blue
    '#d9f99d', // olive → lime
    '#fdba74', // orange
    '#6ee7b7', // emerald
    '#a5b4fc', // indigo
  ]
  return palette[tokenId % palette.length]
}

/**
 * Abbreviate large parameter counts: 1.5B, 7B, 125M, etc.
 */
export const formatParamCount = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return `${n}`
}

/**
 * Clamp a value between min and max.
 */
export const clamp = (val: number, min: number, max: number): number =>
  Math.min(Math.max(val, min), max)

/**
 * Linear interpolation.
 */
export const lerp = (a: number, b: number, t: number): number =>
  a + (b - a) * t

/**
 * Map a 0–1 value to a blue-scale hex color for heatmaps.
 */
export const blueScale = (t: number): string => {
  const v = clamp(t, 0, 1)
  const r = Math.round(lerp(10, 59, v))
  const g = Math.round(lerp(10, 130, v))
  const b = Math.round(lerp(20, 246, v))
  return `rgb(${r},${g},${b})`
}

/**
 * Rough token count estimate from character count.
 */
export const estimateTokenCount = (text: string): number =>
  Math.ceil(text.length / 4)
