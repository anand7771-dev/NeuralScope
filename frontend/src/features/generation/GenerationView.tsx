import { useMemo } from 'react'
import { Cpu, CheckCircle2, Clock, Zap } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { useInferenceStore } from '@/stores/inferenceStore'
import { formatDuration, formatTokensPerSecond } from '@/utils/format'

export function GenerationView() {
  const generatedTokens = useInferenceStore(s => s.generatedTokens)
  const fullGeneratedText = useInferenceStore(s => s.fullGeneratedText)
  const sessionStatus = useInferenceStore(s => s.sessionStatus)
  const timeToFirstToken = useInferenceStore(s => s.timeToFirstToken)
  const tokensPerSecond = useInferenceStore(s => s.tokensPerSecond)
  const totalDurationMs = useInferenceStore(s => s.totalDurationMs)

  const isGenerating = sessionStatus === 'running'

  return (
    <Card
      title="Autoregressive Generation"
      dataSource="actual"
      subtitle="Live Token-by-Token Decoding Stream"
      headerRight={
        <div className="flex items-center gap-2">
          {isGenerating ? (
            <Badge variant="actual" size="sm">
              <span className="animate-pulse mr-1">●</span> Decoding
            </Badge>
          ) : sessionStatus === 'completed' ? (
            <Badge variant="success" size="sm">
              <CheckCircle2 className="w-3 h-3 mr-1 inline" /> Complete
            </Badge>
          ) : null}
        </div>
      }
    >
      <div className="space-y-4">
        {/* Telemetry Metrics Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
          <div className="p-2.5 rounded bg-ns-elevated border border-ns-border flex flex-col gap-0.5">
            <span className="text-ns-textMuted text-[10px] uppercase">Tokens Generated</span>
            <span className="text-white font-bold text-sm">{generatedTokens.length}</span>
          </div>
          <div className="p-2.5 rounded bg-ns-elevated border border-ns-border flex flex-col gap-0.5">
            <span className="text-ns-textMuted text-[10px] uppercase">Throughput</span>
            <span className="text-ns-blue font-bold text-sm">
              {tokensPerSecond ? formatTokensPerSecond(tokensPerSecond) : '—'}
            </span>
          </div>
          <div className="p-2.5 rounded bg-ns-elevated border border-ns-border flex flex-col gap-0.5">
            <span className="text-ns-textMuted text-[10px] uppercase">Time to 1st Token</span>
            <span className="text-ns-purple font-bold text-sm">
              {timeToFirstToken ? `${timeToFirstToken.toFixed(0)} ms` : '—'}
            </span>
          </div>
          <div className="p-2.5 rounded bg-ns-elevated border border-ns-border flex flex-col gap-0.5">
            <span className="text-ns-textMuted text-[10px] uppercase">Total Duration</span>
            <span className="text-ns-green font-bold text-sm">
              {totalDurationMs ? formatDuration(totalDurationMs) : '—'}
            </span>
          </div>
        </div>

        {/* Live Generation Stream Box */}
        <div className="min-h-[160px] p-4 rounded-lg bg-ns-elevated/70 border border-ns-border font-mono text-sm leading-relaxed text-white">
          {generatedTokens.length > 0 ? (
            <span>
              {generatedTokens.map((item, idx) => (
                <span
                  key={idx}
                  title={`Token: "${item.token}" | Prob: ${(item.probability * 100).toFixed(1)}% | Step: ${item.step}`}
                  className="inline-block hover:bg-ns-blue/20 hover:text-ns-blue rounded px-0.5 transition-colors cursor-help animate-fade-in"
                >
                  {item.token}
                </span>
              ))}
              {isGenerating && (
                <span className="inline-block w-2 h-4 bg-ns-blue ml-0.5 align-middle animate-blink" />
              )}
            </span>
          ) : (
            <span className="text-ns-textMuted/60">
              Awaiting model forward passes. Generated response tokens will appear here dynamically.
            </span>
          )}
        </div>
      </div>
    </Card>
  )
}

