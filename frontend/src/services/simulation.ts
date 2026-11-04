import type { InferenceConfig, InferenceEvent } from '@/types'

// Realistic candidate pool for GPT-2 autoregressive simulation
const VOCAB_SAMPLES = [
  { token: ' have', id: 423, logit: 14.2 },
  { token: ' will', id: 481, logit: 13.8 },
  { token: ' can', id: 460, logit: 12.9 },
  { token: ' are', id: 389, logit: 12.4 },
  { token: ' become', id: 1813, logit: 11.9 },
  { token: ' create', id: 2324, logit: 11.5 },
  { token: ' evolved', id: 7543, logit: 11.1 },
  { token: ' together', id: 2362, logit: 10.8 },
  { token: ' into', id: 656, logit: 10.5 },
  { token: ' a', id: 257, logit: 10.2 },
  { token: ' new', id: 649, logit: 9.9 },
  { token: ' civilization', id: 11479, logit: 9.6 },
  { token: ' across', id: 2478, logit: 9.2 },
  { token: ' the', id: 262, logit: 9.0 },
  { token: ' galaxy', id: 12842, logit: 8.8 },
  { token: '.', id: 13, logit: 8.5 },
]

export class SimulationEngine {
  private aborted = false
  private timer: number | null = null

  abort(): void {
    this.aborted = true
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }
  }

  async run(
    config: InferenceConfig,
    emit: (event: InferenceEvent) => void
  ): Promise<void> {
    this.aborted = false
    const sessionId = `demo-${Date.now().toString(36)}`
    const prompt = config.prompt || 'Hello world'
    const maxTokens = Math.min(config.max_tokens || 20, 25)

    const delay = (ms: number) =>
      new Promise<void>((resolve) => {
        this.timer = window.setTimeout(resolve, ms)
      })

    // 1. Session Started
    emit({
      event_id: `evt-${Date.now()}-0`,
      session_id: sessionId,
      event_type: 'session_started',
      timestamp: new Date().toISOString(),
      generation_step: 0,
      data: {
        num_layers: 12,
        num_heads: 12,
        embedding_dim: 768,
        vocab_size: 50257,
        temperature: config.temperature ?? 0.8,
        sampling_strategy: config.temperature === 0 ? 'greedy' : 'multinomial',
        is_simulation: true,
      },
      tensor_references: [],
    })
    await delay(120)
    if (this.aborted) return

    // 2. Tokenization Completed
    // Break prompt into simple tokens
    const words = prompt.match(/\S+|\s+/g) || [prompt]
    const promptTokens = words.map((w, idx) => ({
      token_id: 1000 + idx * 37,
      token_string: w,
      position: idx,
      is_special: false,
    }))

    emit({
      event_id: `evt-${Date.now()}-1`,
      session_id: sessionId,
      event_type: 'tokenization_completed',
      timestamp: new Date().toISOString(),
      generation_step: 0,
      data: {
        tokens: promptTokens,
        input_ids: promptTokens.map((t) => t.token_id),
      },
      tensor_references: [],
    })
    await delay(100)
    if (this.aborted) return

    // 3. Autoregressive Generation Loop
    const generatedWords = [
      ' have',
      ' begun',
      ' to',
      ' merge',
      ' into',
      ' an',
      ' unprecedented',
      ' symbiotic',
      ' intelligence',
      ',',
      ' reshaping',
      ' the',
      ' cosmos',
      ' forever',
      '.',
    ]

    const numTokensToGen = Math.min(maxTokens, generatedWords.length)

    for (let step = 0; step < numTokensToGen; step++) {
      if (this.aborted) return

      // Step through all 12 transformer layers
      for (let l = 0; l < 12; l++) {
        if (this.aborted) return

        emit({
          event_id: `evt-${Date.now()}-l${l}s`,
          session_id: sessionId,
          event_type: 'transformer_layer_started',
          timestamp: new Date().toISOString(),
          generation_step: step,
          layer_index: l,
          data: {},
          tensor_references: [],
        })

        // Quick compute time per layer (10-20ms)
        await delay(15)

        emit({
          event_id: `evt-${Date.now()}-l${l}c`,
          session_id: sessionId,
          event_type: 'transformer_layer_completed',
          timestamp: new Date().toISOString(),
          generation_step: step,
          layer_index: l,
          data: {},
          tensor_references: [],
        })
      }

      // Next token selection
      const nextWord = generatedWords[step % generatedWords.length]
      const chosenId = 2000 + step * 43

      // Probabilities candidate list
      const topTokens = VOCAB_SAMPLES.map((s, idx) => ({
        token_id: idx === 0 ? chosenId : s.id + step,
        token_string: idx === 0 ? nextWord : s.token,
        logit: s.logit - idx * 0.7,
        probability: Math.max(0.01, 0.45 / (idx + 1)),
        rank: idx + 1,
      }))

      emit({
        event_id: `evt-${Date.now()}-prob`,
        session_id: sessionId,
        event_type: 'probabilities_calculated',
        timestamp: new Date().toISOString(),
        generation_step: step,
        data: {
          top_tokens: topTokens,
          sampling_strategy: 'multinomial',
        },
        tensor_references: [],
      })

      emit({
        event_id: `evt-${Date.now()}-sel`,
        session_id: sessionId,
        event_type: 'token_selected',
        timestamp: new Date().toISOString(),
        generation_step: step,
        data: {
          selected_token: topTokens[0],
        },
        tensor_references: [],
      })

      emit({
        event_id: `evt-${Date.now()}-gen`,
        session_id: sessionId,
        event_type: 'token_generated',
        timestamp: new Date().toISOString(),
        generation_step: step,
        data: {
          token_string: nextWord,
          token_id: chosenId,
          probability: topTokens[0].probability,
        },
        tensor_references: [],
      })

      // Realistic generation delay between tokens (60-90ms)
      await delay(75)
    }

    // 4. Generation Completed
    emit({
      event_id: `evt-${Date.now()}-done`,
      session_id: sessionId,
      event_type: 'generation_completed',
      timestamp: new Date().toISOString(),
      generation_step: numTokensToGen,
      data: {
        total_tokens: numTokensToGen,
        duration_ms: numTokensToGen * 180,
      },
      tensor_references: [],
    })
  }
}
