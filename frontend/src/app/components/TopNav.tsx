import { Link, useLocation } from 'react-router-dom'
import { Activity, Cpu, Layers, Terminal } from 'lucide-react'
import { StatusDot } from '@/components/ui/StatusDot'
import { useInferenceStore } from '@/stores/inferenceStore'
import { useModelStore } from '@/stores/modelStore'

export function TopNav() {
  const location = useLocation()
  const wsStatus = useInferenceStore(s => s.wsStatus)
  const models = useModelStore(s => s.models)
  const loadedModel = models.find(m => m.is_loaded)

  const navLinks = [
    { to: '/', label: 'Dashboard', icon: Activity },
    { to: '/models', label: 'Models', icon: Cpu },
  ]

  const wsDotStatus =
    wsStatus === 'connected' ? 'loaded' : wsStatus === 'connecting' ? 'loading' : wsStatus === 'error' ? 'error' : 'idle'

  return (
    <header className="h-14 border-b border-ns-border bg-ns-surface/90 backdrop-blur px-5 flex items-center justify-between z-40 sticky top-0">
      <div className="flex items-center gap-6">
        <Link to="/" className="flex items-center gap-2.5 font-bold tracking-tight text-white hover:opacity-90">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-ns-blue to-ns-purple flex items-center justify-center text-white shadow-md shadow-ns-purple/20">
            <Layers className="w-4 h-4" />
          </div>
          <span className="text-base font-mono">Neural<span className="text-ns-blue">Scope</span></span>
          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-ns-blue/10 text-ns-blue border border-ns-blue/20">
            v0.1.0
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {navLinks.map(({ to, label, icon: Icon }) => {
            const isActive = location.pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-ns-elevated text-white border border-ns-border'
                    : 'text-ns-textMuted hover:text-white hover:bg-ns-elevated/50'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </Link>
            )
          })}
        </nav>
      </div>

      <div className="flex items-center gap-4 text-xs font-mono">
        {loadedModel && (
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-ns-elevated border border-ns-border text-ns-textMuted">
            <Cpu className="w-3.5 h-3.5 text-ns-purple" />
            <span className="text-ns-textPrimary font-semibold">{loadedModel.display_name}</span>
            <span className="text-[10px] text-ns-textMuted">({loadedModel.num_layers}L / {loadedModel.num_heads}H)</span>
          </div>
        )}

        <div className="flex items-center gap-2 px-2.5 py-1 rounded bg-ns-surface border border-ns-border">
          <StatusDot status={wsDotStatus} size="sm" pulse={wsStatus === 'connecting'} />
          <span className="text-[11px] text-ns-textMuted uppercase">WS: {wsStatus}</span>
        </div>
      </div>
    </header>
  )
}

