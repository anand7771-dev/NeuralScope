import { useState, useEffect } from 'react'
import { Server, Check, X, ExternalLink, Zap } from 'lucide-react'
import { getBackendBaseUrl } from '@/services/api'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function BackendSettingsModal({ isOpen, onClose }: Props) {
  const [url, setUrl] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setUrl(localStorage.getItem('neuralscope_backend_url') || '')
      setSaved(false)
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleSave = () => {
    if (url.trim()) {
      localStorage.setItem('neuralscope_backend_url', url.trim())
    } else {
      localStorage.removeItem('neuralscope_backend_url')
    }
    setSaved(true)
    setTimeout(() => {
      onClose()
      window.location.reload()
    }, 600)
  }

  const handleResetToDemo = () => {
    localStorage.removeItem('neuralscope_backend_url')
    setUrl('')
    setSaved(true)
    setTimeout(() => {
      onClose()
      window.location.reload()
    }, 600)
  }

  const activeUrl = getBackendBaseUrl()

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 font-sans">
      <div className="bg-ns-surface border border-ns-border rounded-xl shadow-2xl max-w-lg w-full p-6 text-white relative animate-in fade-in zoom-in-95 duration-150">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-ns-textMuted hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-lg bg-ns-blue/20 border border-ns-blue/30 flex items-center justify-center text-ns-blue">
            <Server className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-base font-mono">Backend Connection Settings</h3>
            <p className="text-xs text-ns-textMuted">Configure live PyTorch server or use demo simulation</p>
          </div>
        </div>

        <div className="space-y-4 text-xs font-mono">
          <div className="p-3 bg-ns-elevated border border-ns-border rounded-lg">
            <div className="flex justify-between items-center mb-1 text-[11px]">
              <span className="text-ns-textMuted">Active Target:</span>
              <span className="text-ns-blue font-bold truncate max-w-[260px]">{activeUrl}</span>
            </div>
            <p className="text-[11px] text-ns-textMuted font-sans">
              On Vercel, NeuralScope runs in <b>Built-in Demo Simulation Mode</b> by default. To connect your live GPU/CPU PyTorch backend, enter your public backend URL below.
            </p>
          </div>

          <div>
            <label className="block mb-1 text-ns-textMuted">Custom Remote Backend URL (HTTP/HTTPS):</label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="e.g. https://my-neuralscope.up.railway.app or ngrok url"
              className="w-full bg-ns-base border border-ns-border px-3 py-2 rounded-lg text-white font-mono text-xs focus:border-ns-blue outline-none"
            />
            <span className="text-[10px] text-ns-textMuted block mt-1">
              Supports Railway, Render, Fly.io, Cloudflare Tunnel, or ngrok.
            </span>
          </div>

          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={handleResetToDemo}
              className="px-3 py-1.5 rounded-lg border border-ns-border text-ns-textMuted hover:text-white transition-colors"
            >
              Reset to Demo Mode
            </button>

            <button
              type="button"
              onClick={handleSave}
              className="px-4 py-1.5 rounded-lg bg-ns-blue hover:bg-ns-blue/90 text-white font-semibold transition-colors flex items-center gap-1.5"
            >
              {saved ? (
                <>
                  <Check className="w-3.5 h-3.5" /> Saved!
                </>
              ) : (
                <>
                  <Zap className="w-3.5 h-3.5" /> Save & Reconnect
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
