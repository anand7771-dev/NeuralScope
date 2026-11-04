import { useState } from 'react'
import { Play, Sparkles, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { InferenceConfigPanel } from './InferenceConfig'
import { useModelStore } from '@/stores/modelStore'
import type { InferenceConfig } from '@/types'

interface PromptWorkspaceProps {
  onRunInference: (config: InferenceConfig) => void
  isRunning: boolean
}

export function PromptWorkspace({ onRunInference, isRunning }: PromptWorkspaceProps) {
  const models = useModelStore(s => s.models)
  const selectedModelId = useModelStore(s => s.selectedModelId) || 'gpt2'
  const selectModel = useModelStore(s => s.selectModel)

  const [prompt, setPrompt] = useState('In a distant future, artificial intelligence and humanity')
  const [config, setConfig] = useState<InferenceConfig>({
    model_id: selectedModelId,
    prompt: '',
    max_new_tokens: 30,
    temperature: 0.8,
    top_k: 50,
    top_p: 0.9,
    do_sample: true,
    repetition_penalty: 1.0,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || isRunning) return
    onRunInference({
      ...config,
      model_id: selectedModelId,
      prompt: prompt.trim(),
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 h-full">
      {/* Model Selector */}
      <div className="flex flex-col gap-1.5 font-mono text-xs">
        <label className="text-ns-textMuted">Target Model</label>
        <select
          value={selectedModelId}
          disabled={isRunning}
          onChange={e => {
            selectModel(e.target.value)
            setConfig(c => ({ ...c, model_id: e.target.value }))
          }}
          className="bg-ns-elevated border border-ns-border text-white px-3 py-2 rounded-lg outline-none focus:border-ns-blue transition-colors"
        >
          {models.length > 0 ? (
            models.map(m => (
              <option key={m.model_id} value={m.model_id}>
                {m.display_name} {m.is_loaded ? '• (Loaded)' : ''}
              </option>
            ))
          ) : (
            <option value="gpt2">GPT-2 (124M Parameters)</option>
          )}
        </select>
      </div>

      {/* Prompt Editor */}
      <div className="flex flex-col gap-1.5 flex-1 min-h-[140px] font-mono text-xs">
        <div className="flex justify-between items-center text-ns-textMuted">
          <label className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-ns-purple" />
            Input Prompt
          </label>
          <span className="text-[11px]">{prompt.length} chars (approx ~{Math.ceil(prompt.length / 4)} tokens)</span>
        </div>
        <textarea
          value={prompt}
          disabled={isRunning}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter prompt to observe inference pipeline..."
          className="w-full flex-1 p-3.5 rounded-lg bg-ns-elevated/70 border border-ns-border text-white placeholder-ns-textMuted/60 resize-none font-mono text-sm leading-relaxed outline-none focus:border-ns-blue transition-colors"
        />
        <span className="text-[10px] text-ns-textMuted text-right">Press Ctrl + Enter to run</span>
      </div>

      {/* Inference Parameters */}
      <div className="p-3.5 rounded-lg bg-ns-surface/60 border border-ns-border">
        <InferenceConfigPanel
          config={config}
          onChange={updated => setConfig(prev => ({ ...prev, ...updated }))}
          disabled={isRunning}
        />
      </div>

      {/* Action Button */}
      <Button
        type="submit"
        variant="primary"
        disabled={!prompt.trim() || isRunning}
        className="w-full justify-center py-2.5 font-mono text-sm shadow-lg shadow-ns-blue/20"
      >
        <Play className="w-4 h-4 fill-current" />
        {isRunning ? 'Running Instrumented Inference...' : 'Run Instrumented Inference'}
      </Button>
    </form>
  )
}

