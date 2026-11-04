import { useCallback, useRef, useState } from 'react'
import { api } from '@/services/api'
import { InferenceWebSocket } from '@/services/websocket'
import { useInferenceStore } from '@/stores/inferenceStore'
import type { InferenceConfig, SessionInfo } from '@/types'

// ── Hook ──────────────────────────────────────────────────────────────────

interface UseInferenceReturn {
  startInference: (config: InferenceConfig) => Promise<string>
  stopInference: () => void
  isRunning: boolean
  error: string | null
}

/**
 * Orchestrates inference: creates a session, opens a WebSocket, routes all
 * incoming events into the Zustand store, and returns control handles.
 */
export function useInference(): UseInferenceReturn {
  const addEvent = useInferenceStore(s => s.addEvent)
  const reset = useInferenceStore(s => s.reset)
  const setCurrentSession = useInferenceStore(s => s.setCurrentSession)
  const setWsStatus = useInferenceStore(s => s.setWsStatus)
  const setSessionStatus = useInferenceStore(s => s.setSessionStatus)

  const wsRef = useRef<InferenceWebSocket | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startInference = useCallback(
    async (config: InferenceConfig): Promise<string> => {
      // Clean up any existing session
      wsRef.current?.disconnect()
      reset()
      setError(null)
      setIsRunning(true)

      try {
        // 1. Create session on the backend
        const session = await api.sessions.create<SessionInfo>(config)
        const sessionId = session.session_id

        setCurrentSession(sessionId)
        setSessionStatus('running')

        // 2. Open WebSocket for event streaming
        const ws = new InferenceWebSocket(sessionId)
        wsRef.current = ws

        // Route status changes to the store
        ws.onStatus(status => {
          setWsStatus(status)
          if (status === 'disconnected' || status === 'error') {
            setIsRunning(false)
          }
        })

        // Route inference events into the store
        ws.onEvent(event => {
          addEvent(event)
          if (
            event.event_type === 'generation_completed' ||
            event.event_type === 'inference_error'
          ) {
            setIsRunning(false)
          }
        })

        ws.connect()
        return sessionId
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to start inference'
        setError(message)
        setIsRunning(false)
        setSessionStatus('error')
        throw err
      }
    },
    [addEvent, reset, setCurrentSession, setWsStatus, setSessionStatus]
  )

  const stopInference = useCallback(() => {
    wsRef.current?.disconnect()
    wsRef.current = null
    setIsRunning(false)
    setSessionStatus('completed')
  }, [setSessionStatus])

  return { startInference, stopInference, isRunning, error }
}
