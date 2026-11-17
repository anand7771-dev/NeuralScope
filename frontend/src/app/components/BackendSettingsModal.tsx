import { useState, useEffect } from 'react'
import { Server, Check, X, Key, Zap } from 'lucide-react'
import { getBackendBaseUrl } from '@/services/api'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function BackendSettingsModal({ isOpen, onClose }: Props) {
  const [url, setUrl] = useState('')
  const [geminiKey, setGeminiKey] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setUrl(localStorage.getItem('neuralscope_backend_url') || '')
      setGeminiKey(localStorage.getItem('gemini_api_key') || '')
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

    if (geminiKey.trim()) {
      localStorage.setItem('gemini_api_key', geminiKey.trim())
    } else {
      localStorage.removeItem('gemini_api_key')
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
            <h3 className="font-semibold text-base font-mono">Backend & API Settings</h3>
            <p className="text-xs text-ns-textMuted">Configure Google Gemini key or remote server</p>
          </div>
        </div>

        <div className="space-y-4 text-xs font-mono">
          {/* Gemini API Key */}
          <div className="p-3 bg-ns-elevated border border-ns-purple/40 rounded-lg">
            <div className="flex items-center gap-2 mb-1.5 text-ns-purple font-semibold">
              <Key className="w-4 h-4" />
              <span>Google Gemini API Key</span>
            </div>
            <p className="text-[11px] text-ns-textMuted font-sans mb-2">
              Enter your Gemini API key from <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-ns-blue underline">Google AI Studio</a>. NeuralScope will use official Gemini 2.0 Flash for real, intelligent inference.
            </p>
            <input
              type="password"
              value={geminiKey}
              onChange={(e) => setGeminiKey(e.target.value)}
              placeholder="AIzaSy..."
              className="w-full bg-ns-base border border-ns-border px-3 py-2 rounded-lg text-white font-mono text-xs focus:border-ns-purple outline-none"
            />
          </div>

          {/* Backend URL */}
          <div>
            <div className="flex justify-between items-center mb-1 text-[11px]">
              <span className="text-ns-textMuted">Custom Remote Backend URL:</span>
              <span className="text-ns-blue font-bold truncate max-w-[200px]">{activeUrl}</span>
            </div>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="e.g. https://my-backend.railway.app or leave blank for local"
              className="w-full bg-ns-base border border-ns-border px-3 py-2 rounded-lg text-white font-mono text-xs focus:border-ns-blue outline-none"
            />
          </div>

          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={handleResetToDemo}
              className="px-3 py-1.5 rounded-lg border border-ns-border text-ns-textMuted hover:text-white transition-colors"
            >
              Reset Defaults
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
                  <Zap className="w-3.5 h-3.5" /> Save & Connect
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
