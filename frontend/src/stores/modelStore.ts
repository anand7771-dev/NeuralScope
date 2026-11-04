import { create } from 'zustand'
import { api } from '@/services/api'
import type { ModelInfo } from '@/types'

// ── State Shape ───────────────────────────────────────────────────────────

interface ModelStore {
  models: ModelInfo[];
  selectedModelId: string | null;
  isLoading: boolean;
  isLoadingModel: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  selectModel: (id: string) => void;
  loadModel: (id: string) => Promise<void>;
  getSelectedModel: () => ModelInfo | undefined;
}

// ── Store ─────────────────────────────────────────────────────────────────

export const useModelStore = create<ModelStore>((set, get) => ({
  models: [],
  selectedModelId: null,
  isLoading: false,
  isLoadingModel: false,
  error: null,

  fetchModels: async () => {
    set({ isLoading: true, error: null })
    try {
      const data = await api.models.list()
      // Backend returns { models: ModelInfo[] } or ModelInfo[]
      const models: ModelInfo[] = Array.isArray(data)
        ? data
        : (data as { models: ModelInfo[] }).models ?? []

      set({ models, isLoading: false })

      // Auto-select if there is exactly one already-loaded model
      const loaded = models.find(m => m.is_loaded)
      if (loaded && get().selectedModelId == null) {
        set({ selectedModelId: loaded.model_id })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch models'
      set({ error: message, isLoading: false })
    }
  },

  selectModel: (id: string) => set({ selectedModelId: id }),

  loadModel: async (id: string) => {
    set({ isLoadingModel: true, error: null })
    try {
      await api.models.load(id)
      // Re-fetch to get updated is_loaded state
      await get().fetchModels()
      set({ selectedModelId: id, isLoadingModel: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load model'
      set({ error: message, isLoadingModel: false })
    }
  },

  getSelectedModel: () => {
    const { models, selectedModelId } = get()
    return models.find(m => m.model_id === selectedModelId)
  },
}))
