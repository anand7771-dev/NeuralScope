import { Sliders, HelpCircle } from 'lucide-react'
import { Tooltip } from '@/components/ui/Tooltip'
import type { InferenceConfig } from '@/types'

interface InferenceConfigProps {
  config: InferenceConfig
  onChange: (updated: Partial<InferenceConfig>) => void
  disabled?: boolean
}

export function InferenceConfigPanel({ config, onChange, disabled }: InferenceConfigProps) {
  return (
    <div className="space-y-4 text-xs font-mono">
      <div className="flex items-center gap-1.5 text-ns-textPrimary font-semibold pb-1 border-b border-ns-border/60">
        <Sliders className="w-3.5 h-3.5 text-ns-blue" />
        <span>Inference Parameters</span>
      </div>

      {/* Max New Tokens */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-ns-textMuted flex items-center gap-1">
            Max Tokens
            <Tooltip content="Maximum number of tokens to generate">
              <HelpCircle className="w-3 h-3 text-ns-textMuted cursor-help" />
            </Tooltip>
          </span>
          <span className="text-ns-textPrimary font-semibold">{config.max_new_tokens}</span>
        </div>
        <input
          type="range"
          min="5"
          max="200"
          step="5"
          value={config.max_new_tokens}
          disabled={disabled}
          onChange={e => onChange({ max_new_tokens: Number(e.target.value) })}
          className="w-full accent-ns-blue bg-ns-surface h-1.5 rounded cursor-pointer disabled:opacity-50"
        />
      </div>

      {/* Temperature */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-ns-textMuted flex items-center gap-1">
            Temperature
            <Tooltip content="Softmax temperature. Lower is deterministic, higher is creative.">
              <HelpCircle className="w-3 h-3 text-ns-textMuted cursor-help" />
            </Tooltip>
          </span>
          <span className="text-ns-textPrimary font-semibold">{config.temperature.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.05"
          value={config.temperature}
          disabled={disabled}
          onChange={e => onChange({ temperature: Number(e.target.value) })}
          className="w-full accent-ns-blue bg-ns-surface h-1.5 rounded cursor-pointer disabled:opacity-50"
        />
      </div>

      {/* Top-K */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-ns-textMuted flex items-center gap-1">
            Top-K
            <Tooltip content="Keep only the K highest probability tokens before sampling. 0 disables.">
              <HelpCircle className="w-3 h-3 text-ns-textMuted cursor-help" />
            </Tooltip>
          </span>
          <span className="text-ns-textPrimary font-semibold">{config.top_k}</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={config.top_k}
          disabled={disabled}
          onChange={e => onChange({ top_k: Number(e.target.value) })}
          className="w-full accent-ns-blue bg-ns-surface h-1.5 rounded cursor-pointer disabled:opacity-50"
        />
      </div>

      {/* Top-P (Nucleus) */}
      <div className="space-y-1.5">
        <div className="flex justify-between items-center">
          <span className="text-ns-textMuted flex items-center gap-1">
            Top-P (Nucleus)
            <Tooltip content="Cumulative probability cutoff for candidate tokens.">
              <HelpCircle className="w-3 h-3 text-ns-textMuted cursor-help" />
            </Tooltip>
          </span>
          <span className="text-ns-textPrimary font-semibold">{config.top_p.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.05"
          value={config.top_p}
          disabled={disabled}
          onChange={e => onChange({ top_p: Number(e.target.value) })}
          className="w-full accent-ns-blue bg-ns-surface h-1.5 rounded cursor-pointer disabled:opacity-50"
        />
      </div>

      {/* Sampling Toggle */}
      <div className="flex items-center justify-between pt-2 border-t border-ns-border/40">
        <span className="text-ns-textMuted">Sample (vs Greedy)</span>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange({ do_sample: !config.do_sample })}
          className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
            config.do_sample
              ? 'bg-ns-blue/20 text-ns-blue border border-ns-blue/40'
              : 'bg-ns-surface text-ns-textMuted border border-ns-border'
          }`}
        >
          {config.do_sample ? 'Multinomial' : 'Greedy (Argmax)'}
        </button>
      </div>
    </div>
  )
}

