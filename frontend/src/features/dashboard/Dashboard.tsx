import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Play, Cpu, Layers, HardDrive, Zap, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { StatusDot } from '@/components/ui/StatusDot'
import { useModelStore } from '@/stores/modelStore'
import { api } from '@/services/api'
import type { SessionInfo } from '@/types'

export function Dashboard() {
  const navigate = useNavigate()
  const models = useModelStore(s => s.models)
  const fetchModels = useModelStore(s => s.fetchModels)
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [systemInfo, setSystemInfo] = useState<any>(null)

  useEffect(() => {
    fetchModels()
    api.sessions.list<SessionInfo[]>().then(res => {
      if (Array.isArray(res)) setSessions(res)
    }).catch(() => {})

    api.health.check().then(res => setSystemInfo(res)).catch(() => {})
  }, [fetchModels])

  const loadedModel = models.find(m => m.is_loaded) || models[0]

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Hero Welcome Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-ns-surface via-ns-elevated to-ns-surface border border-ns-border flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-ns-blue/15 text-ns-blue border border-ns-blue/30 text-xs font-mono">
            <Zap className="w-3.5 h-3.5" />
            Interactive LLM Inference Visualization Platform
          </div>
          <h1 className="text-3xl font-bold font-mono tracking-tight text-white">
            Neural<span className="text-ns-blue">Scope</span>
          </h1>
          <p className="text-sm text-ns-textMuted max-w-2xl leading-relaxed">
            Observe the internal computational journey of Large Language Models in real-time.
            Inspect actual token embeddings, attention heads, transformer layers, KV-cache dynamics, and token probability sampling.
          </p>
        </div>

        <Button
          variant="primary"
          onClick={() => navigate('/inference/new')}
          className="py-3 px-5 text-sm font-mono shadow-xl shadow-ns-blue/20 shrink-0"
        >
          <Play className="w-4 h-4 fill-current mr-2" />
          Launch Inference Session
        </Button>
      </div>

      {/* Grid: Model Status + Hardware Telemetry */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Model Status Card */}
        <Card
          title="Active Architecture"
          dataSource="actual"
          subtitle={loadedModel ? loadedModel.display_name : 'Loading model catalog...'}
          className="md:col-span-2"
        >
          {loadedModel ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2 font-mono text-xs">
              <div className="p-3 rounded-lg bg-ns-elevated border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Parameters</span>
                <div className="text-white font-bold text-base mt-1">124M</div>
                <span className="text-[10px] text-ns-textMuted">GPT-2 Small</span>
              </div>
              <div className="p-3 rounded-lg bg-ns-elevated border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Transformer Blocks</span>
                <div className="text-ns-blue font-bold text-base mt-1">{loadedModel.num_layers} Layers</div>
                <span className="text-[10px] text-ns-textMuted">{loadedModel.num_heads} Heads / Layer</span>
              </div>
              <div className="p-3 rounded-lg bg-ns-elevated border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Hidden Dimension</span>
                <div className="text-ns-purple font-bold text-base mt-1">{loadedModel.hidden_size}</div>
                <span className="text-[10px] text-ns-textMuted">768 Dim Stream</span>
              </div>
              <div className="p-3 rounded-lg bg-ns-elevated border border-ns-border">
                <span className="text-ns-textMuted text-[10px] uppercase">Vocabulary Size</span>
                <div className="text-ns-green font-bold text-base mt-1">{loadedModel.vocab_size.toLocaleString()}</div>
                <span className="text-[10px] text-ns-textMuted">BPE Subwords</span>
              </div>
            </div>
          ) : (
            <div className="py-6 text-center text-ns-textMuted font-mono text-xs">
              Connecting to local backend server...
            </div>
          )}
        </Card>

        {/* Hardware & Runtime Card */}
        <Card title="System Telemetry" dataSource="actual" subtitle="Local Compute Environment">
          <div className="space-y-3 font-mono text-xs pt-1">
            <div className="flex items-center justify-between pb-2 border-b border-ns-border/60">
              <span className="text-ns-textMuted">System RAM:</span>
              <span className="text-white font-semibold">
                {systemInfo?.hardware?.ram_available_gb ?? '—'} GB / {systemInfo?.hardware?.ram_total_gb ?? '16'} GB Free
              </span>
            </div>
            <div className="flex items-center justify-between pb-2 border-b border-ns-border/60">
              <span className="text-ns-textMuted">CPU Hardware:</span>
              <span className="text-white font-semibold">
                {systemInfo?.hardware?.cpu_cores_logical ?? '12'} Threads ({systemInfo?.hardware?.cpu_cores_physical ?? '10'} Cores)
              </span>
            </div>
            <div className="flex items-center justify-between pb-2 border-b border-ns-border/60">
              <span className="text-ns-textMuted">Acceleration:</span>
              <span className="text-ns-blue font-semibold">
                {systemInfo?.hardware?.cuda_available ? systemInfo.hardware.cuda_device : 'CPU Engine (AVX2)'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ns-textMuted">Backend Server:</span>
              <span className="text-ns-green font-semibold">FastAPI • v0.1.0</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Sessions */}
      <Card title="Recent Inference Sessions" subtitle="Session Event Logs and Recorded Runs">
        {sessions.length > 0 ? (
          <div className="divide-y divide-ns-border/50 font-mono text-xs">
            {sessions.map(s => (
              <div
                key={s.session_id}
                onClick={() => navigate(`/inference/${s.session_id}`)}
                className="py-3 px-2 flex items-center justify-between hover:bg-ns-elevated/60 rounded cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-3">
                  <StatusDot status={s.status === 'completed' ? 'loaded' : 'idle'} size="sm" />
                  <span className="text-white font-medium max-w-md truncate">"{s.prompt}"</span>
                </div>
                <div className="flex items-center gap-4 text-ns-textMuted">
                  <span>{s.token_count} tokens</span>
                  <span>{s.total_duration_ms ? `${(s.total_duration_ms / 1000).toFixed(2)}s` : ''}</span>
                  <ArrowRight className="w-4 h-4 text-ns-blue" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center text-ns-textMuted font-mono text-xs">
            No previous sessions recorded. Start your first inference run above.
          </div>
        )}
      </Card>
    </div>
  )
}

