import type { InferenceEvent } from '@/types'

// ── Handler types ─────────────────────────────────────────────────────────

type EventHandler = (event: InferenceEvent) => void
type StatusHandler = (
  status: 'connecting' | 'connected' | 'disconnected' | 'error'
) => void

// ── InferenceWebSocket ────────────────────────────────────────────────────

/**
 * Manages a single WebSocket connection for one inference session.
 * Automatically parses JSON messages and dispatches to registered handlers.
 *
 * Usage:
 *   const ws = new InferenceWebSocket(sessionId)
 *   const unsub = ws.onEvent(e => store.addEvent(e))
 *   ws.connect()
 *   // later:
 *   ws.disconnect()
 *   unsub()
 */
export class InferenceWebSocket {
  private ws: WebSocket | null = null
  private readonly sessionId: string
  private eventHandlers: EventHandler[] = []
  private statusHandlers: StatusHandler[] = []
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private shouldReconnect = false
  private reconnectAttempts = 0
  private readonly maxReconnectAttempts = 5
  private readonly reconnectBaseDelayMs = 1000

  constructor(sessionId: string) {
    this.sessionId = sessionId
  }

  /** Opens the WebSocket connection. */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return

    this.shouldReconnect = true
    this.notifyStatus('connecting')

    let wsUrl: string
    const localBackend = typeof window !== 'undefined' ? localStorage.getItem('neuralscope_backend_url') : null
    if (localBackend) {
      const clean = localBackend.endsWith('/') ? localBackend.slice(0, -1) : localBackend
      const wsOrigin = clean.replace(/^http/, 'ws')
      wsUrl = `${wsOrigin}/ws/${this.sessionId}`
    } else if (import.meta.env.VITE_WS_URL) {
      const customWs = import.meta.env.VITE_WS_URL
      const cleanWs = customWs.endsWith('/') ? customWs.slice(0, -1) : customWs
      wsUrl = `${cleanWs}/${this.sessionId}`
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`
    }
    this.ws = new WebSocket(wsUrl)

    this.ws.onopen = () => {
      this.reconnectAttempts = 0
      this.notifyStatus('connected')
    }

    this.ws.onmessage = (e: MessageEvent<string>) => {
      try {
        const event = JSON.parse(e.data) as InferenceEvent
        if (event.event_type === 'generation_completed' || event.event_type === 'inference_error') {
          this.shouldReconnect = false
        }
        this.eventHandlers.forEach(h => h(event))
      } catch {
        console.warn('[InferenceWebSocket] Failed to parse message:', e.data)
      }
    }

    this.ws.onclose = () => {
      this.notifyStatus('disconnected')
      if (this.shouldReconnect) {
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = () => {
      this.notifyStatus('error')
    }
  }

  /** Closes the WebSocket connection and disables auto-reconnect. */
  disconnect(): void {
    this.shouldReconnect = false
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
  }

  /**
   * Register a handler for inference events.
   * @returns An unsubscribe function.
   */
  onEvent(handler: EventHandler): () => void {
    this.eventHandlers.push(handler)
    return () => {
      this.eventHandlers = this.eventHandlers.filter(h => h !== handler)
    }
  }

  /**
   * Register a handler for connection status changes.
   * @returns An unsubscribe function.
   */
  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.push(handler)
    return () => {
      this.statusHandlers = this.statusHandlers.filter(h => h !== handler)
    }
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  // ── Private ─────────────────────────────────────────────────────────────

  private notifyStatus(
    status: 'connecting' | 'connected' | 'disconnected' | 'error'
  ): void {
    this.statusHandlers.forEach(h => h(status))
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('[InferenceWebSocket] Max reconnect attempts reached.')
      return
    }

    const delay =
      this.reconnectBaseDelayMs * Math.pow(2, this.reconnectAttempts)
    this.reconnectAttempts += 1

    this.reconnectTimer = setTimeout(() => {
      if (this.shouldReconnect) {
        console.info(
          `[InferenceWebSocket] Reconnecting (attempt ${this.reconnectAttempts})…`
        )
        this.connect()
      }
    }, delay)
  }
}
