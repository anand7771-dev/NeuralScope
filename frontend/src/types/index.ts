// ── NeuralScope Type Definitions ──────────────────────────────────────────
// These types mirror the backend Pydantic schemas exactly.

/** Indicates the provenance/reliability of a piece of data shown in the UI */
export type DataSource = 'actual' | 'derived' | 'conceptual' | 'simulated';

/** All event types emitted during an inference session */
export type EventType =
  | 'session_started'
  | 'prompt_received'
  | 'prompt_formatted'
  | 'tokenization_started'
  | 'tokenization_completed'
  | 'embeddings_generated'
  | 'input_tensor_prepared'
  | 'transformer_layer_started'
  | 'transformer_layer_completed'
  | 'attention_computed'
  | 'mlp_computed'
  | 'normalization_completed'
  | 'residual_computed'
  | 'kv_cache_updated'
  | 'logits_generated'
  | 'probabilities_calculated'
  | 'token_selected'
  | 'token_generated'
  | 'generation_completed'
  | 'inference_error'
  | 'progress_update';

/** A reference to a tensor stored server-side; use the ref_id to fetch actual data */
export interface TensorReference {
  ref_id: string;
  name: string;
  shape: number[];
  dtype: string;
  device: string;
  data_source: DataSource;
  description: string;
  num_elements: number;
  memory_bytes: number;
}

/** A single event in the inference event stream */
export interface InferenceEvent {
  session_id: string;
  event_id: string;
  event_type: EventType;
  sequence_number: number;
  timestamp: string;        // ISO 8601
  stage: string;
  layer_index?: number;
  generation_step?: number;
  metadata: Record<string, unknown>;
  tensor_references: TensorReference[];
  data: Record<string, unknown>;
  data_source: DataSource;
  schema_version: string;
}

/** A single token from the tokenizer */
export interface TokenInfo {
  token_id: number;
  token_string: string;
  position: number;
  is_special: boolean;
}

/** Full metadata about a loaded (or loadable) model */
export interface ModelInfo {
  model_id: string;
  display_name: string;
  architecture: string;
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  num_parameters: number;
  supports_kv_cache: boolean;
  supports_attention_output: boolean;
  positional_encoding_type: string;
  normalization_type: string;
  activation_function: string;
  dtype: string;
  device: string;
  is_loaded: boolean;
}

/** Parameters submitted to start an inference session */
export interface InferenceConfig {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  do_sample: boolean;
  repetition_penalty: number;
}

/** Session lifecycle info */
export interface SessionInfo {
  session_id: string;
  model_id: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  created_at: string;
  completed_at?: string;
  prompt: string;
  generated_text: string;
  token_count: number;
  total_duration_ms: number;
}

/** A candidate token with its score from the sampling step */
export interface TopToken {
  token_id: number;
  token_string: string;
  logit: number;
  probability: number;
  rank: number;
}

/** Backend health response */
export interface HealthInfo {
  status: string;
  version: string;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  device: string;
  memory_gb?: number;
  cpu_cores?: number;
}

/** A generated token during streaming */
export interface GeneratedToken {
  token: string;
  token_id: number;
  probability: number;
  step: number;
}

/** Layer processing state for the layer explorer */
export type LayerStatus = 'pending' | 'processing' | 'completed';

export interface LayerState {
  index: number;
  status: LayerStatus;
  attentionDone: boolean;
  mlpDone: boolean;
  normDone: boolean;
  startedAt?: number;   // epoch ms
  completedAt?: number; // epoch ms
  attentionRefId?: string;
}
