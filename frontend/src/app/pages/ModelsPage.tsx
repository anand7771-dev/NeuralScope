import { useEffect } from 'react'
import { Cpu, CheckCircle2, Download, AlertCircle } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { useModelStore } from '@/stores/modelStore'

export function ModelsPage() {
  const models = useModelStore(s => s.models)
  const fetchModels = useModelStore(s => s.fetchModels)
  const loadModel = useModelStore(s => s.loadModel)
  const isLoadingModel = useModelStore(s => s.isLoadingModel)

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-mono font-bold text-white">Model Architecture Catalog</h1>
        <p className="text-xs font-mono text-ns-textMuted">
          Supported transformer architectures, configurations, and instrumented hooks.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map(m => (
          <Card
            key={m.model_id}
            title={m.display_name}
            dataSource="actual"
            subtitle={`${m.architecture} • ${m.dtype} • ${m.device}`}
            headerRight={
              m.is_loaded ? (
                <Badge variant="success" size="sm">
                  <CheckCircle2 className="w-3 h-3 mr-1 inline" /> Loaded in RAM
                </Badge>
              ) : (
                <Badge variant="default" size="sm">Not Loaded</Badge>
              )
            }
          >
            <div className="space-y-4 font-mono text-xs pt-2">
              <div className="grid grid-cols-2 gap-3 text-ns-textMuted">
                <div>
                  <span>Layers: </span>
                  <span className="text-white font-semibold">{m.num_layers} Blocks</span>
                </div>
                <div>
                  <span>Attention Heads: </span>
                  <span className="text-white font-semibold">{m.num_heads} Heads</span>
                </div>
                <div>
                  <span>Hidden Dimension: </span>
                  <span className="text-white font-semibold">{m.hidden_size}</span>
                </div>
                <div>
                  <span>MLP Intermediate: </span>
                  <span className="text-white font-semibold">{m.intermediate_size}</span>
                </div>
                <div>
                  <span>Max Sequence Length: </span>
                  <span className="text-white font-semibold">{m.max_position_embeddings}</span>
                </div>
                <div>
                  <span>Vocabulary Size: </span>
                  <span className="text-white font-semibold">{m.vocab_size.toLocaleString()}</span>
                </div>
                <div>
                  <span>Positional Encoding: </span>
                  <span className="text-ns-blue font-semibold capitalize">{m.positional_encoding_type}</span>
                </div>
                <div>
                  <span>Normalization: </span>
                  <span className="text-ns-purple font-semibold capitalize">{m.normalization_type}</span>
                </div>
              </div>

              <div className="pt-3 border-t border-ns-border flex justify-end">
                {!m.is_loaded && (
                  <Button
                    variant="primary"
                    disabled={isLoadingModel}
                    onClick={() => loadModel(m.model_id)}
                  >
                    <Download className="w-3.5 h-3.5 mr-1" />
                    Load Weights into Memory
                  </Button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}

