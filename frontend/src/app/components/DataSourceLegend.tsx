import { Badge } from '@/components/ui/Badge'

export function DataSourceLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs border border-ns-border/80 bg-ns-surface/60 rounded-lg p-2.5">
      <span className="text-ns-textMuted text-[11px] font-mono uppercase tracking-wider">Data Provenance:</span>
      <div className="flex items-center gap-1.5">
        <Badge variant="actual" size="sm">Actual</Badge>
        <span className="text-[11px] text-ns-textMuted">Real model tensor/compute</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Badge variant="derived" size="sm">Derived</Badge>
        <span className="text-[11px] text-ns-textMuted">Calculated from actual data</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Badge variant="conceptual" size="sm">Conceptual</Badge>
        <span className="text-[11px] text-ns-textMuted">Educational diagram</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Badge variant="simulated" size="sm">Simulated</Badge>
        <span className="text-[11px] text-ns-textMuted">Demonstration only</span>
      </div>
    </div>
  )
}

