import type { InferenceEvent } from '@/types'
import type { useInferenceStore } from '@/stores/inferenceStore'

// The store type, used as a parameter type for handlers below.
type StoreActions = Pick<
  ReturnType<typeof useInferenceStore.getState>,
  | 'addEvent'
  | 'setTotalLayers'
  | 'setSessionStatus'
>

/**
 * Process a tokenization_completed event.
 * Extracts the token list from event.data and updates the store.
 */
export function handleTokenizationCompleted(
  event: InferenceEvent,
  _actions: StoreActions
): void {
  // The store's addEvent already handles this via its switch block.
  // This module exists as an escape hatch for components that need
  // event-specific business logic outside the store.
  void event
}

/**
 * Process a token_generated event.
 * Returns the token string extracted from event.data, or null.
 */
export function extractGeneratedToken(event: InferenceEvent): {
  tokenString: string
  tokenId: number
  probability: number
  step: number
} | null {
  if (event.event_type !== 'token_generated') return null
  const tokenString = event.data['token_string'] as string | undefined
  if (!tokenString) return null
  return {
    tokenString,
    tokenId: (event.data['token_id'] as number | undefined) ?? 0,
    probability: (event.data['probability'] as number | undefined) ?? 0,
    step: event.generation_step ?? 0,
  }
}

/**
 * Returns the layer index from any layer-related event.
 */
export function extractLayerIndex(event: InferenceEvent): number | null {
  return event.layer_index ?? null
}

/**
 * Maps an EventType string to a human-readable label.
 */
export function eventTypeLabel(eventType: string): string {
  const labels: Record<string, string> = {
    session_started:          'Session Started',
    prompt_received:          'Prompt Received',
    prompt_formatted:         'Prompt Formatted',
    tokenization_started:     'Tokenization Started',
    tokenization_completed:   'Tokenization Complete',
    embeddings_generated:     'Embeddings Generated',
    input_tensor_prepared:    'Input Tensor Prepared',
    transformer_layer_started:'Layer Started',
    transformer_layer_completed:'Layer Completed',
    attention_computed:       'Attention Computed',
    mlp_computed:             'MLP Computed',
    normalization_completed:  'Normalization Done',
    residual_computed:        'Residual Computed',
    kv_cache_updated:         'KV Cache Updated',
    logits_generated:         'Logits Generated',
    probabilities_calculated: 'Probabilities Calculated',
    token_selected:           'Token Selected',
    token_generated:          'Token Generated',
    generation_completed:     'Generation Complete',
    inference_error:          'Inference Error',
    progress_update:          'Progress Update',
  }
  return labels[eventType] ?? eventType
}

/**
 * Maps an EventType to a color class used in the timeline / dots.
 */
export function eventTypeColor(eventType: string): string {
  if (eventType.startsWith('tokenization')) return '#3b82f6' // blue
  if (eventType.startsWith('transformer_layer') || eventType.startsWith('attention') || eventType === 'mlp_computed') {
    return '#8b5cf6' // purple
  }
  if (eventType === 'token_generated' || eventType === 'generation_completed') {
    return '#10b981' // green
  }
  if (eventType === 'inference_error') return '#ef4444' // red
  if (eventType === 'probabilities_calculated' || eventType === 'logits_generated') {
    return '#f59e0b' // amber
  }
  return '#64748b' // muted
}
