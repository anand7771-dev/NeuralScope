import { useState } from 'react'
import { Hash, Code, ListFilter } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { tokenToColor } from '@/utils/format'
import { useInferenceStore } from '@/stores/inferenceStore'
import type { TokenInfo } from '@/types'

export function TokenizationView() {
  const tokens = useInferenceStore(s => s.tokens)
  const [selectedToken, setSelectedToken] = useState<TokenInfo | null>(null)

  if (tokens.length === 0) {
    return (
      <Card title="Tokenization" dataSource="actual" subtitle="BPE Subword Token Stream">
        <div className="h-48 flex items-center justify-center text-ns-textMuted font-mono text-xs">
          No tokens processed yet. Submit a prompt to view tokenization.
        </div>
      </Card>
    )
  }

  return (
    <Card
      title="Tokenization"
      dataSource="actual"
      subtitle={`Byte-Pair Encoding (BPE) • ${tokens.length} tokens`}
      headerRight={
        <div className="text-xs font-mono text-ns-textMuted">
          Vocab: <span className="text-white">50,257</span>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Token Chips Representation */}
        <div className="p-3.5 rounded-lg bg-ns-elevated border border-ns-border flex flex-wrap gap-1.5">
          {tokens.map(token => {
            const isSelected = selectedToken?.position === token.position
            const bg = tokenToColor(token.token_id)
            return (
              <button
                key={token.position}
                onClick={() => setSelectedToken(token)}
                style={{ backgroundColor: `${bg}33`, borderColor: isSelected ? '#3b82f6' : `${bg}66` }}
                className={`flex flex-col items-center px-2 py-1 rounded border text-xs font-mono transition-transform hover:scale-105 ${
                  isSelected ? 'ring-2 ring-ns-blue' : ''
                }`}
              >
                <span className="text-white font-medium">{token.token_string.replace(/ /g, '␣')}</span>
                <span className="text-[10px] text-ns-textMuted mt-0.5">{token.token_id}</span>
              </button>
            )
          })}
        </div>

        {/* Selected Token Details */}
        {selectedToken && (
          <div className="p-3 rounded-lg bg-ns-surface border border-ns-border text-xs font-mono flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div>
                <span className="text-ns-textMuted">Position: </span>
                <span className="text-white font-semibold">#{selectedToken.position}</span>
              </div>
              <div>
                <span className="text-ns-textMuted">Token String: </span>
                <span className="text-ns-textCode font-semibold font-mono">
                  "{selectedToken.token_string}"
                </span>
              </div>
              <div>
                <span className="text-ns-textMuted">ID: </span>
                <span className="text-ns-blue font-semibold">{selectedToken.token_id}</span>
              </div>
            </div>
            {selectedToken.is_special && (
              <Badge variant="warning" size="sm">
                Control / Special Token
              </Badge>
            )}
          </div>
        )}

        {/* Token Sequence Breakdown Table */}
        <div className="max-h-48 overflow-y-auto border border-ns-border rounded-lg">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-ns-surface text-ns-textMuted border-b border-ns-border text-[11px] sticky top-0">
              <tr>
                <th className="p-2">Pos</th>
                <th className="p-2">Token ID</th>
                <th className="p-2">Token Subword</th>
                <th className="p-2">Bytes / Special</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ns-border/40">
              {tokens.map(token => (
                <tr
                  key={token.position}
                  onClick={() => setSelectedToken(token)}
                  className={`hover:bg-ns-elevated/70 cursor-pointer ${
                    selectedToken?.position === token.position ? 'bg-ns-blue/10' : ''
                  }`}
                >
                  <td className="p-2 text-ns-textMuted">{token.position}</td>
                  <td className="p-2 text-ns-blue">{token.token_id}</td>
                  <td className="p-2 text-white font-medium">"{token.token_string}"</td>
                  <td className="p-2">
                    {token.is_special ? (
                      <Badge variant="warning" size="sm">Special</Badge>
                    ) : (
                      <span className="text-ns-textMuted text-[10px]">Standard subword</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  )
}

