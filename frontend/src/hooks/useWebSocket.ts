import { useEffect, useRef, useState } from 'react'
import { InferenceWebSocket } from '@/services/websocket'
import type { InferenceEvent } from '@/types'

// ── Types ─────────────────────────────────────────────────────────────────

type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketOptions {
  /** Called whenever a new InferenceEvent arrives */
  onEvent?: (event: InferenceEvent) => void
  /** Whether to auto-connect when sessionId is provided */
  autoConnect?: boolean
}

interface UseWebSocketReturn {
  status: WsStatus
  connect: () => void
  disconnect: () => void
}

// ── Hook ──────────────────────────────────────────────────────────────────

/**
 * Lower-level hook that wraps InferenceWebSocket with React lifecycle
 * management. Automatically cleans up on unmount or sessionId change.
 */
export function useWebSocket(
  sessionId: string | null,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const { onEvent, autoConnect = true } = options

  const [status, setStatus] = useState<WsStatus>('disconnected')
  const wsRef = useRef<InferenceWebSocket | null>(null)
  const onEventRef = useRef(onEvent)

  // Keep the callback ref up to date without triggering a reconnect
  useEffect(() => {
    onEventRef.current = onEvent
  })

  const disconnect = () => {
    wsRef.current?.disconnect()
    wsRef.current = null
    setStatus('disconnected')
  }

  const connect = () => {
    if (!sessionId) return
    disconnect()

    const ws = new InferenceWebSocket(sessionId)
    wsRef.current = ws

    ws.onStatus(s => setStatus(s))
    ws.onEvent(evt => onEventRef.current?.(evt))

    ws.connect()
  }

  // Auto connect/disconnect when sessionId changes
  useEffect(() => {
    if (!sessionId) {
      disconnect()
      return
    }
    if (autoConnect) connect()

    return () => {
      wsRef.current?.disconnect()
      wsRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, autoConnect])

  return { status, connect, disconnect }
}
