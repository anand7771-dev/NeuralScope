import { useRef, useEffect } from 'react'
import { History, Play, Pause, RotateCcw } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { useInferenceStore } from '@/stores/inferenceStore'
import { eventTypeLabel } from '@/utils/eventHandlers'

export function ExecutionTimeline() {
  const events = useInferenceStore(s => s.events)
  const eventIndex = useInferenceStore(s => s.eventIndex)
  const setEventIndex = useInferenceStore(s => s.setEventIndex)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll timeline to end as events arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth
    }
  }, [events.length])

  if (events.length === 0) return null

  return (
    <div className="h-14 border-t border-ns-border bg-ns-surface/95 backdrop-blur px-4 flex items-center gap-4 z-30">
      <div className="flex items-center gap-2 text-xs font-mono text-ns-textMuted shrink-0">
        <History className="w-4 h-4 text-ns-blue" />
        <span className="font-semibold text-white">Timeline</span>
        <span>({events.length} events)</span>
      </div>

      {/* Event Nodes Scrollable Track */}
      <div ref={scrollRef} className="flex-1 flex items-center gap-1.5 overflow-x-auto py-2 no-scrollbar">
        {events.map((evt, idx) => {
          const isSelected = idx === eventIndex
          const label = eventTypeLabel(evt.event_type)

          const nodeColor = evt.event_type.includes('token')
            ? 'bg-ns-blue border-ns-blue/50 text-ns-blue'
            : evt.event_type.includes('layer') || evt.event_type.includes('attention')
            ? 'bg-ns-purple border-ns-purple/50 text-ns-purple'
            : evt.event_type.includes('error')
            ? 'bg-red-500 border-red-500 text-red-400'
            : 'bg-ns-green border-ns-green/50 text-ns-green'

          return (
            <button
              key={evt.event_id || idx}
              onClick={() => setEventIndex(idx)}
              title={`#${evt.sequence_number} • ${label} • ${evt.stage}`}
              className={`shrink-0 flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono border transition-all ${
                isSelected
                  ? 'ring-2 ring-white scale-105 bg-ns-elevated'
                  : 'bg-ns-elevated/60 hover:bg-ns-elevated'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${nodeColor}`} />
              <span className="text-white max-w-[120px] truncate">{label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

