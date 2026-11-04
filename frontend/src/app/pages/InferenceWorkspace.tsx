import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Layers,
  Terminal,
  Sparkles,
  BarChart2,
  Compass,
  Grid,
  FunctionSquare,
  Network,
} from 'lucide-react'
import { PromptWorkspace } from '@/features/prompt-workspace/PromptWorkspace'
import { TokenizationView } from '@/features/tokenizer/TokenizationView'
import { GenerationView } from '@/features/generation/GenerationView'
import { LayerExplorer } from '@/features/transformer/LayerExplorer'
import { OperationInspector } from '@/features/transformer/OperationInspector'
import { AttentionExplorer } from '@/features/attention/AttentionExplorer'
import { EmbeddingExplorer } from '@/features/embeddings/EmbeddingExplorer'
import { TensorInspector } from '@/features/tensor-inspector/TensorInspector'
import { SamplingView } from '@/features/sampling/SamplingView'
import { ExecutionTimeline } from '@/features/timeline/ExecutionTimeline'
import { DataSourceLegend } from '@/app/components/DataSourceLegend'
import { useInference } from '@/hooks/useInference'
import { useInferenceStore } from '@/stores/inferenceStore'
import { api } from '@/services/api'
import type { InferenceConfig, InferenceEvent } from '@/types'

export function InferenceWorkspace() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const { startInference, isRunning } = useInference()
  const addEvent = useInferenceStore(s => s.addEvent)
  const reset = useInferenceStore(s => s.reset)

  const [activeTab, setActiveTab] = useState<
    'generation' | 'tokenization' | 'layers' | 'attention' | 'embeddings' | 'tensors' | 'operations' | 'sampling'
  >('generation')

  // Replay historical session if id provided and not 'new'
  useEffect(() => {
    if (sessionId && sessionId !== 'new') {
      reset()
      api.sessions
        .getEvents<InferenceEvent[]>(sessionId)
        .then(events => {
          if (Array.isArray(events)) {
            events.forEach(evt => addEvent(evt))
          }
        })
        .catch(err => {
          console.error('Failed to load session events:', err)
        })
    }
  }, [sessionId, reset, addEvent])

  const handleRunInference = async (config: InferenceConfig) => {
    try {
      const newSessionId = await startInference(config)
      navigate(`/inference/${newSessionId}`, { replace: true })
    } catch (err) {
      console.error('Inference run error:', err)
    }
  }

  const tabs = [
    { id: 'generation', label: 'Decoding Stream', icon: Terminal },
    { id: 'tokenization', label: 'Tokenization', icon: Sparkles },
    { id: 'layers', label: 'Layer Hierarchy', icon: Layers },
    { id: 'attention', label: 'Multi-Head Attention', icon: Network },
    { id: 'embeddings', label: 'Embedding Space', icon: Compass },
    { id: 'tensors', label: 'Tensor Inspector', icon: Grid },
    { id: 'operations', label: 'Math Operations', icon: FunctionSquare },
    { id: 'sampling', label: 'Sampling & Logits', icon: BarChart2 },
  ] as const

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] overflow-hidden">
      {/* Main Workspace Area (Split Panels) */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Side: Prompt & Inference Config Workspace */}
        <div className="w-80 md:w-96 border-r border-ns-border bg-ns-surface/80 p-4 flex flex-col overflow-y-auto shrink-0">
          <PromptWorkspace onRunInference={handleRunInference} isRunning={isRunning} />
        </div>

        {/* Right Side: Visualisation Modules */}
        <div className="flex-1 flex flex-col overflow-hidden bg-ns-bg">
          {/* Top Tabs Bar */}
          <div className="h-12 border-b border-ns-border bg-ns-surface px-4 flex items-center justify-between shrink-0 overflow-x-auto">
            <div className="flex items-center gap-1 font-mono text-xs">
              {tabs.map(({ id, label, icon: Icon }) => {
                const isActive = activeTab === id
                return (
                  <button
                    key={id}
                    onClick={() => setActiveTab(id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                      isActive
                        ? 'bg-ns-blue/20 text-ns-blue font-semibold border border-ns-blue/40'
                        : 'text-ns-textMuted hover:text-white'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {label}
                  </button>
                )
              })}
            </div>

            <div className="hidden xl:block shrink-0 ml-4">
              <DataSourceLegend />
            </div>
          </div>

          {/* Active View Container */}
          <div className="flex-1 p-5 overflow-y-auto space-y-6">
            {activeTab === 'generation' && <GenerationView />}
            {activeTab === 'tokenization' && <TokenizationView />}
            {activeTab === 'layers' && <LayerExplorer />}
            {activeTab === 'attention' && <AttentionExplorer />}
            {activeTab === 'embeddings' && <EmbeddingExplorer />}
            {activeTab === 'tensors' && <TensorInspector />}
            {activeTab === 'operations' && <OperationInspector />}
            {activeTab === 'sampling' && <SamplingView />}
          </div>
        </div>
      </div>

      {/* Bottom Timeline */}
      <ExecutionTimeline />
    </div>
  )
}
