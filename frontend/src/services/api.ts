import type { InferenceConfig } from '@/types'

const BASE_URL = '/api'

// ── Generic helpers ───────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`HTTP ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

function get<T>(path: string): Promise<T> {
  return fetch(`${BASE_URL}${path}`).then(r => handleResponse<T>(r))
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }).then(r => handleResponse<T>(r))
}

// ── API client ────────────────────────────────────────────────────────────

export const api = {
  models: {
    list: <T = unknown>() => get<T>('/models'),
    get:  <T = unknown>(id: string) => get<T>(`/models/${id}`),
    load: <T = unknown>(id: string) => post<T>(`/models/${id}/load`),
  },

  sessions: {
    create: <T = unknown>(config: InferenceConfig) =>
      post<T>('/sessions', config),
    get:    <T = unknown>(id: string) => get<T>(`/sessions/${id}`),
    list:   <T = unknown>() => get<T>('/sessions'),
    getEvents: <T = unknown>(id: string) => get<T>(`/sessions/${id}/events`),
  },

  tensors: {
    get: <T = unknown>(sessionId: string, refId: string, sliceSpec?: string): Promise<T> => {
      const url = new URL(
        `${window.location.origin}${BASE_URL}/tensors/${sessionId}/${refId}`
      )
      if (sliceSpec) url.searchParams.set('slice_spec', sliceSpec)
      return fetch(url.toString()).then(r => handleResponse<T>(r))
    },
    getSummary: <T = unknown>(sessionId: string, refId: string) =>
      get<T>(`/tensors/${sessionId}/${refId}/summary`),
    getHistogram: <T = unknown>(sessionId: string, refId: string, bins = 20) =>
      get<T>(`/tensors/${sessionId}/${refId}/histogram?bins=${bins}`),
    getMatrix2D: <T = unknown>(
      sessionId: string,
      refId: string,
      params: { row_offset?: number; col_offset?: number; max_rows?: number; max_cols?: number; dim_indices?: string }
    ) => {
      const q = new URLSearchParams()
      if (params.row_offset !== undefined) q.set('row_offset', String(params.row_offset))
      if (params.col_offset !== undefined) q.set('col_offset', String(params.col_offset))
      if (params.max_rows !== undefined) q.set('max_rows', String(params.max_rows))
      if (params.max_cols !== undefined) q.set('max_cols', String(params.max_cols))
      if (params.dim_indices) q.set('dim_indices', params.dim_indices)
      return get<T>(`/tensors/${sessionId}/${refId}/matrix2d?${q.toString()}`)
    },
  },

  embeddings: {
    pca: <T = unknown>(sessionId: string, nComponents = 2) =>
      get<T>(`/embeddings/sessions/${sessionId}/pca?n_components=${nComponents}`),
    similarity: <T = unknown>(sessionId: string) =>
      get<T>(`/embeddings/sessions/${sessionId}/similarity`),
  },

  health: {
    check: <T = unknown>() => fetch('/health').then(r => handleResponse<T>(r)),
  },
}
