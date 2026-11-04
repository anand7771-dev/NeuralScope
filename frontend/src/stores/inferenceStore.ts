import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type {
  InferenceEvent,
  SessionInfo,
  TokenInfo,
  GeneratedToken,
  TopToken,
  LayerState,
} from '@/types'

// ── State Shape ───────────────────────────────────────────────────────────

interface InferenceStore {
  // Session
  currentSessionId: string | null;
  sessionStatus: SessionInfo['status'];

  // Raw event log
  events: InferenceEvent[];
  /** Current playback index (for replay mode) */
  eventIndex: number;

  // Tokenization
  tokens: TokenInfo[];

  // Live generation output
  generatedTokens: GeneratedToken[];
  fullGeneratedText: string;

  // Transformer layer tracking
  currentLayer: number;
  totalLayers: number;
  layers: LayerState[];

  // Top-k sampling output
  topTokens: TopToken[];
  selectedToken: TopToken | null;
  samplingStrategy: string;
  temperature: number;

  // Performance metrics
  timeToFirstToken: number | null;  // ms
  tokensPerSecond: number | null;
  totalDurationMs: number | null;

  // Connection
  wsStatus: 'connecting' | 'connected' | 'disconnected' | 'error';

  // ── Actions ──────────────────────────────────────────────────────────────

  addEvent: (event: InferenceEvent) => void;
  reset: () => void;
  setCurrentSession: (sessionId: string) => void;
  setSessionStatus: (status: SessionInfo['status']) => void;
  setWsStatus: (status: InferenceStore['wsStatus']) => void;
  setTotalLayers: (n: number) => void;
  setEventIndex: (index: number) => void;
}

// ── Initial State ─────────────────────────────────────────────────────────

const INITIAL_STATE = {
  currentSessionId: null,
  sessionStatus: 'idle' as SessionInfo['status'],
  events: [] as InferenceEvent[],
  eventIndex: 0,
  tokens: [] as TokenInfo[],
  generatedTokens: [] as GeneratedToken[],
  fullGeneratedText: '',
  currentLayer: -1,
  totalLayers: 0,
  layers: [] as LayerState[],
  topTokens: [] as TopToken[],
  selectedToken: null,
  samplingStrategy: 'greedy',
  temperature: 1.0,
  timeToFirstToken: null,
  tokensPerSecond: null,
  totalDurationMs: null,
  wsStatus: 'disconnected' as InferenceStore['wsStatus'],
}

// ── Store ─────────────────────────────────────────────────────────────────

export const useInferenceStore = create<InferenceStore>()(
  subscribeWithSelector((set, get) => ({
    ...INITIAL_STATE,

    addEvent: (event: InferenceEvent) => {
      set(state => ({ events: [...state.events, event] }))
      processEvent(event, set, get)
    },

    reset: () => set(INITIAL_STATE),

    setCurrentSession: (sessionId: string) =>
      set({ currentSessionId: sessionId }),

    setSessionStatus: (status: SessionInfo['status']) =>
      set({ sessionStatus: status }),

    setWsStatus: (status: InferenceStore['wsStatus']) =>
      set({ wsStatus: status }),

    setTotalLayers: (n: number) => {
      set({
        totalLayers: n,
        layers: Array.from({ length: n }, (_, i) => ({
          index: i,
          status: 'pending',
          attentionDone: false,
          mlpDone: false,
          normDone: false,
        })),
      })
    },

    setEventIndex: (index: number) => set({ eventIndex: index }),
  }))
)

// ── Event Processing Logic ─────────────────────────────────────────────────
// Kept here (not in components) so components stay presentational.

type Setter = Parameters<Parameters<typeof create>[0]>[0]
type Getter = Parameters<Parameters<typeof create>[0]>[1]

function processEvent(
  event: InferenceEvent,
  set: Setter,
  get: Getter
): void {
  const { event_type, data, layer_index, generation_step } = event
  const now = Date.now()

  switch (event_type) {
    case 'session_started': {
      set({ sessionStatus: 'running' })
      const numLayers = data['num_layers'] as number | undefined
      if (numLayers != null && numLayers > 0) {
        set({
          totalLayers: numLayers,
          layers: Array.from({ length: numLayers }, (_, i) => ({
            index: i,
            status: 'pending',
            attentionDone: false,
            mlpDone: false,
            normDone: false,
          })),
        })
      }
      const temp = data['temperature'] as number | undefined
      const strategy = data['sampling_strategy'] as string | undefined
      if (temp != null) set({ temperature: temp })
      if (strategy != null) set({ samplingStrategy: strategy })
      break
    }

    case 'tokenization_completed': {
      const rawTokens = data['tokens'] as any
      const inputIds = (data['input_ids'] as number[]) || []

      if (Array.isArray(rawTokens)) {
        if (typeof rawTokens[0] === 'string') {
          const tokens: TokenInfo[] = (rawTokens as string[]).map((str, idx) => ({
            token_id: inputIds[idx] ?? 0,
            token_string: str,
            position: idx,
            is_special: false,
          }))
          set({ tokens })
        } else {
          const tokens: TokenInfo[] = rawTokens.map((t: any, idx: number) => ({
            token_id: t.token_id ?? inputIds[idx] ?? 0,
            token_string: t.token_string ?? String(t),
            position: t.position ?? idx,
            is_special: t.is_special ?? false,
          }))
          set({ tokens })
        }
      }
      break
    }

    case 'transformer_layer_started': {
      if (layer_index == null) break
      set(state => ({
        currentLayer: layer_index,
        layers: state.layers.map(l =>
          l.index === layer_index
            ? { ...l, status: 'processing', startedAt: now }
            : l
        ),
      }))
      break
    }

    case 'attention_computed': {
      if (layer_index == null) break
      const refId = event.tensor_references[0]?.ref_id
      set(state => ({
        layers: state.layers.map(l =>
          l.index === layer_index
            ? { ...l, attentionDone: true, attentionRefId: refId }
            : l
        ),
      }))
      break
    }

    case 'mlp_computed': {
      if (layer_index == null) break
      set(state => ({
        layers: state.layers.map(l =>
          l.index === layer_index ? { ...l, mlpDone: true } : l
        ),
      }))
      break
    }

    case 'normalization_completed': {
      if (layer_index == null) break
      set(state => ({
        layers: state.layers.map(l =>
          l.index === layer_index ? { ...l, normDone: true } : l
        ),
      }))
      break
    }

    case 'transformer_layer_completed': {
      if (layer_index == null) break
      set(state => ({
        layers: state.layers.map(l =>
          l.index === layer_index
            ? { ...l, status: 'completed', completedAt: now }
            : l
        ),
      }))
      break
    }

    case 'probabilities_calculated': {
      const rawTopTokens = data['top_tokens'] as Array<{
        token_id: number
        token_string: string
        logit: number
        probability: number
        rank: number
      }> | undefined

      if (Array.isArray(rawTopTokens)) {
        const topTokens: TopToken[] = rawTopTokens.map(t => ({
          token_id: t.token_id,
          token_string: t.token_string,
          logit: t.logit ?? 0,
          probability: t.probability,
          rank: t.rank,
        }))
        set({ topTokens })
      }
      const strategy = data['sampling_strategy'] as string | undefined
      if (strategy) set({ samplingStrategy: strategy })
      break
    }

    case 'token_selected': {
      const selectedRaw = data['selected_token'] as {
        token_id: number
        token_string: string
        logit: number
        probability: number
        rank: number
      } | undefined

      if (selectedRaw) {
        set({
          selectedToken: {
            token_id: selectedRaw.token_id,
            token_string: selectedRaw.token_string,
            logit: selectedRaw.logit ?? 0,
            probability: selectedRaw.probability,
            rank: selectedRaw.rank ?? 1,
          },
        })
      }
      break
    }

    case 'token_generated': {
      const tokenStr = (data['token_string'] || data['token']) as string | undefined
      const tokenId = data['token_id'] as number | undefined
      const probability = data['probability'] as number | undefined
      const step = generation_step ?? get().generatedTokens.length

      if (tokenStr != null) {
        const newToken: GeneratedToken = {
          token: tokenStr,
          token_id: tokenId ?? 0,
          probability: probability ?? 0,
          step,
        }

        const isFirstToken = get().generatedTokens.length === 0
        if (isFirstToken) {
          const sessionStartTime = get().events[0]
            ? new Date(get().events[0].timestamp).getTime()
            : now
          set({ timeToFirstToken: now - sessionStartTime })
        }

        set(state => {
          const newTokens = [...state.generatedTokens, newToken]
          const newText = newTokens.map(t => t.token).join('')

          // Compute running tokens/sec
          const startMs = state.events[0]
            ? new Date(state.events[0].timestamp).getTime()
            : now
          const elapsedSec = (now - startMs) / 1000
          const tps = elapsedSec > 0 ? newTokens.length / elapsedSec : null

          return {
            generatedTokens: newTokens,
            fullGeneratedText: newText,
            tokensPerSecond: tps,
          }
        })
      }
      break
    }

    case 'generation_completed': {
      const durationMs = data['total_duration_ms'] as number | undefined
      const tps = data['tokens_per_second'] as number | undefined
      const fullText = data['generated_text'] as string | undefined

      set({
        sessionStatus: 'completed',
        ...(durationMs != null && { totalDurationMs: durationMs }),
        ...(tps != null && { tokensPerSecond: tps }),
        ...(fullText != null && { fullGeneratedText: fullText }),
      })
      break
    }

    case 'inference_error': {
      set({ sessionStatus: 'error' })
      break
    }

    default:
      break
  }
}
