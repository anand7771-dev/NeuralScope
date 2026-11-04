import { BarChart3, Check } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ProbabilityBar } from '@/components/charts/ProbabilityBar'
import { useInferenceStore } from '@/stores/inferenceStore'
import { formatNumber } from '@/utils/format'

export function SamplingView() {
  const topTokens = useInferenceStore(s => s.topTokens)
  const selectedToken = useInferenceStore(s => s.selectedToken)
  const samplingStrategy = useInferenceStore(s => s.samplingStrategy)
  const temperature = useInferenceStore(s => s.temperature)

  if (topTokens.length === 0) {
    return (
      <Card title="Next-Token Probability Distribution" dataSource="derived" subtitle="Softmax over Vocab Logits">
        <div className="h-48 flex items-center justify-center text-ns-textMuted font-mono text-xs">
          Awaiting next-token logits from language model head.
        </div>
      </Card>
    )
  }

  return (
    <Card
      title="Next-Token Probability Distribution"
      dataSource="derived"
      subtitle={`Sampling Strategy: ${samplingStrategy || 'Top-P'} • Temp: ${temperature.toFixed(2)}`}
      headerRight={
        selectedToken && (
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-mono text-ns-textMuted">Sampled:</span>
            <Badge variant="actual" size="sm">
              "{selectedToken.token_string}" ({(selectedToken.probability * 100).toFixed(1)}%)
            </Badge>
          </div>
        )
      }
    >
      <div className="space-y-4">
        {/* Softmax Formula Banner */}
        <div className="p-2.5 rounded-lg bg-ns-elevated border border-ns-border flex items-center justify-between text-xs font-mono">
          <div className="text-ns-textMuted">
            P(w<sub>i</sub>) = exp(z<sub>i</sub> / T) / ∑ exp(z<sub>j</sub> / T)
          </div>
          <div className="text-ns-blue font-semibold">T = {temperature.toFixed(2)}</div>
        </div>

        {/* Horizontal Probability Bars */}
        <ProbabilityBar
          tokens={topTokens}
          maxItems={10}
          selectedTokenId={selectedToken?.token_id}
          showLogits={true}
        />
      </div>
    </Card>
  )
}

