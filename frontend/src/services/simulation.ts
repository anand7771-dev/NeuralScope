import type { InferenceConfig, InferenceEvent } from '@/types'

// Gemini API key: read at build time from env, fallback to localStorage at runtime
const BUILT_IN_GEMINI_KEY: string = import.meta.env.VITE_GEMINI_API_KEY || ''

// ── Seeded PRNG (deterministic per prompt, varied across prompts) ──────

function hashString(str: string): number {
  let hash = 5381
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0
  }
  return Math.abs(hash)
}

function createRng(seed: number) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) | 0
    return ((s >>> 0) / 4294967296)
  }
}

// ── Response corpus organized by topic keywords ────────────────────────

const TOPIC_RESPONSES: Record<string, string[][]> = {
  // Science & Nature
  science: [
    [' is', ' a', ' fascinating', ' topic', ' that', ' scientists', ' have', ' studied', ' for', ' centuries', ',', ' revealing', ' deep', ' patterns', ' in', ' the', ' natural', ' world', '.'],
    [' remains', ' one', ' of', ' the', ' most', ' intriguing', ' questions', ' in', ' modern', ' science', ',', ' with', ' researchers', ' uncovering', ' new', ' insights', ' every', ' year', '.'],
  ],
  sun: [
    [' The', ' sun', ' rises', ' from', ' the', ' east', ' and', ' sets', ' in', ' the', ' west', ',', ' a', ' phenomenon', ' caused', ' by', ' Earth', "'s", ' rotation', ' on', ' its', ' axis', '.'],
    [' rises', ' in', ' the', ' eastern', ' horizon', ' due', ' to', ' the', ' Earth', "'s", ' counter', '-', 'clock', 'wise', ' rotation', ' when', ' viewed', ' from', ' above', ' the', ' North', ' Pole', '.'],
  ],
  sunrise: [
    [' occur', ' in', ' the', ' east', '.', ' This', ' is', ' because', ' the', ' Earth', ' rotates', ' from', ' west', ' to', ' east', ' on', ' its', ' axis', ',', ' making', ' the', ' sun', ' appear', ' to', ' rise', ' in', ' the', ' eastern', ' sky', '.'],
    [' are', ' visible', ' from', ' the', ' eastern', ' horizon', '.', ' The', ' direction', ' is', ' determined', ' by', ' Earth', "'s", ' rotation', ' about', ' its', ' polar', ' axis', '.'],
  ],
  direction: [
    [' is', ' east', '.', ' The', ' sun', ' always', ' rises', ' in', ' the', ' east', ' and', ' sets', ' in', ' the', ' west', ' because', ' of', ' Earth', "'s", ' rotation', '.'],
    [' depends', ' on', ' multiple', ' factors', ' including', ' latitude', ',', ' season', ',', ' and', ' the', ' tilt', ' of', ' Earth', "'s", ' axis', '.'],
  ],
  east: [
    [' is', ' where', ' the', ' sun', ' rises', ' every', ' morning', '.', ' This', ' consistent', ' pattern', ' results', ' from', ' the', ' rotation', ' of', ' our', ' planet', '.'],
  ],
  // Technology & AI
  ai: [
    [' has', ' transformed', ' the', ' way', ' we', ' interact', ' with', ' technology', ',', ' enabling', ' machines', ' to', ' learn', ' patterns', ' from', ' vast', ' amounts', ' of', ' data', '.'],
    [' continues', ' to', ' evolve', ' rapidly', ',', ' with', ' breakthroughs', ' in', ' natural', ' language', ' processing', ' and', ' computer', ' vision', ' reshaping', ' industries', '.'],
  ],
  model: [
    [' processes', ' input', ' tokens', ' through', ' multiple', ' transformer', ' layers', ',', ' each', ' containing', ' self', '-', 'attention', ' and', ' feed', '-', 'forward', ' networks', '.'],
    [' uses', ' learned', ' weight', ' matrices', ' to', ' transform', ' input', ' representations', ' into', ' meaningful', ' output', ' predictions', '.'],
  ],
  transformer: [
    [' architecture', ' revolutionized', ' natural', ' language', ' processing', ' by', ' introducing', ' self', '-', 'attention', ' mechanisms', ' that', ' capture', ' long', '-', 'range', ' dependencies', '.'],
    [' models', ' process', ' sequences', ' in', ' parallel', ' using', ' multi', '-', 'head', ' attention', ',', ' making', ' them', ' far', ' more', ' efficient', ' than', ' recurrent', ' networks', '.'],
  ],
  neural: [
    [' networks', ' learn', ' hierarchical', ' representations', ' of', ' data', ',', ' with', ' each', ' layer', ' extracting', ' increasingly', ' abstract', ' features', '.'],
    [' computation', ' involves', ' matrix', ' multiplications', ',', ' non', '-', 'linear', ' activations', ',', ' and', ' gradient', '-', 'based', ' optimization', '.'],
  ],
  // General knowledge
  water: [
    [' is', ' essential', ' for', ' all', ' known', ' forms', ' of', ' life', '.', ' It', ' covers', ' about', ' 71', '%', ' of', ' the', ' Earth', "'s", ' surface', '.'],
    [' exists', ' in', ' three', ' states', ':', ' solid', ',', ' liquid', ',', ' and', ' gas', ',', ' cycling', ' continuously', ' through', ' the', ' environment', '.'],
  ],
  earth: [
    [' is', ' the', ' third', ' planet', ' from', ' the', ' Sun', ' and', ' the', ' only', ' known', ' celestial', ' body', ' to', ' harbor', ' life', '.'],
    [' rotates', ' on', ' its', ' axis', ' once', ' every', ' 24', ' hours', ',', ' creating', ' the', ' cycle', ' of', ' day', ' and', ' night', '.'],
  ],
  moon: [
    [' orbits', ' Earth', ' approximately', ' every', ' 27', '.', '3', ' days', ' and', ' influences', ' ocean', ' tides', ' through', ' gravitational', ' pull', '.'],
  ],
  history: [
    [' teaches', ' us', ' valuable', ' lessons', ' about', ' human', ' civilization', ',', ' culture', ',', ' and', ' the', ' forces', ' that', ' shape', ' societies', '.'],
    [' is', ' filled', ' with', ' pivotal', ' moments', ' that', ' altered', ' the', ' course', ' of', ' human', ' development', '.'],
  ],
  future: [
    [' holds', ' both', ' challenges', ' and', ' opportunities', ' as', ' technology', ' continues', ' to', ' advance', ' at', ' an', ' unprecedented', ' pace', '.'],
    [' will', ' likely', ' be', ' shaped', ' by', ' artificial', ' intelligence', ',', ' renewable', ' energy', ',', ' and', ' space', ' exploration', '.'],
  ],
  code: [
    [' is', ' the', ' foundation', ' of', ' modern', ' software', ' development', ',', ' enabling', ' humans', ' to', ' communicate', ' instructions', ' to', ' machines', '.'],
  ],
  python: [
    [' is', ' a', ' versatile', ' programming', ' language', ' known', ' for', ' its', ' clean', ' syntax', ' and', ' extensive', ' ecosystem', ' of', ' libraries', '.'],
  ],
  math: [
    [' provides', ' the', ' universal', ' language', ' for', ' describing', ' patterns', ',', ' structures', ',', ' and', ' relationships', ' in', ' the', ' world', '.'],
  ],
  life: [
    [' is', ' a', ' complex', ' phenomenon', ' that', ' emerges', ' from', ' the', ' interaction', ' of', ' matter', ' and', ' energy', ' over', ' billions', ' of', ' years', '.'],
  ],
  learn: [
    [' by', ' forming', ' new', ' neural', ' connections', ' and', ' strengthening', ' existing', ' pathways', ' through', ' repeated', ' practice', ' and', ' exposure', '.'],
  ],
  hello: [
    ['!', ' How', ' are', ' you', ' doing', ' today', '?', ' I', ' am', ' a', ' language', ' model', ' ready', ' to', ' help', ' with', ' your', ' questions', '.'],
    ['!', ' It', "'s", ' great', ' to', ' connect', '.', ' What', ' would', ' you', ' like', ' to', ' explore', ' today', '?'],
  ],
  // People & language
  love: [
    [' is', ' a', ' complex', ' emotion', ' involving', ' deep', ' affection', ',', ' attachment', ',', ' and', ' care', ' for', ' another', ' being', '.'],
  ],
  music: [
    [' has', ' been', ' a', ' fundamental', ' part', ' of', ' human', ' culture', ' for', ' thousands', ' of', ' years', ',', ' expressing', ' emotions', ' beyond', ' words', '.'],
  ],
  space: [
    [' is', ' vast', ' and', ' largely', ' unexplored', '.', ' Our', ' universe', ' contains', ' billions', ' of', ' galaxies', ',', ' each', ' with', ' billions', ' of', ' stars', '.'],
  ],
}

// Fallback responses when no keyword matches
const FALLBACK_RESPONSES: string[][] = [
  [' is', ' an', ' interesting', ' topic', ' that', ' involves', ' multiple', ' interconnected', ' concepts', ' and', ' perspectives', ' worth', ' exploring', ' in', ' greater', ' detail', '.'],
  [' raises', ' important', ' questions', ' about', ' how', ' we', ' understand', ' and', ' interpret', ' the', ' world', ' around', ' us', '.'],
  [' can', ' be', ' understood', ' through', ' careful', ' analysis', ' of', ' the', ' underlying', ' principles', ' and', ' mechanisms', ' at', ' work', '.'],
  [' represents', ' a', ' fascinating', ' area', ' of', ' inquiry', ' that', ' has', ' captured', ' the', ' attention', ' of', ' researchers', ' worldwide', '.'],
  [' continues', ' to', ' be', ' debated', ' among', ' experts', ',', ' with', ' new', ' evidence', ' emerging', ' from', ' recent', ' studies', ' and', ' experiments', '.'],
  [' offers', ' a', ' unique', ' window', ' into', ' the', ' complexities', ' of', ' nature', ' and', ' human', ' understanding', '.'],
  [',', ' when', ' examined', ' closely', ',', ' reveals', ' surprising', ' depth', ' and', ' nuance', ' that', ' challenge', ' conventional', ' assumptions', '.'],
  [' has', ' significant', ' implications', ' for', ' our', ' understanding', ' of', ' both', ' theoretical', ' and', ' practical', ' domains', '.'],
]

// ── Token-level vocabulary for realistic probability distributions ──────

function buildCandidatePool(chosenToken: string, rng: () => number): Array<{token: string; id: number; logit: number}> {
  const candidates = [
    ' the', ' a', ' is', ' of', ' to', ' and', ' in', ' that', ' for', ' it',
    ' with', ' as', ' on', ' are', ' was', ' be', ' at', ' by', ' from', ' or',
    ' an', ' have', ' has', ' this', ' which', ' their', ' its', ' not', ' but', ' they',
    ' can', ' will', ' more', ' when', ' about', ' up', ' been', ' would', ' than', ' each',
    ',', '.', ':', ';', '!', '?', ' -', "'s",
  ]
  // Shuffle deterministically and pick top candidates
  const shuffled = candidates
    .filter(c => c !== chosenToken)
    .sort(() => rng() - 0.5)
    .slice(0, 15)

  return shuffled.map((token, idx) => ({
    token,
    id: hashString(token) % 50257,
    logit: 12.0 - idx * 0.8 + (rng() - 0.5) * 2,
  }))
}

// ── Select response based on prompt content ────────────────────────────

function selectResponse(prompt: string, rng: () => number): string[] {
  const lower = prompt.toLowerCase()

  // Check each topic keyword
  const matched: string[][] = []
  for (const [keyword, responses] of Object.entries(TOPIC_RESPONSES)) {
    if (lower.includes(keyword)) {
      matched.push(...responses)
    }
  }

  if (matched.length > 0) {
    const idx = Math.floor(rng() * matched.length)
    return matched[idx]
  }

  // No keyword match — use fallback, but vary based on prompt hash
  const idx = Math.floor(rng() * FALLBACK_RESPONSES.length)
  return FALLBACK_RESPONSES[idx]
}

// ── Simulation Engine ──────────────────────────────────────────────────

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
    const maxTokens = config.max_tokens || 20

    const seed = hashString(prompt + Date.now().toString())
    const rng = createRng(seed)

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
    await delay(100 + rng() * 80)
    if (this.aborted) return

    // 2. Tokenization
    const words = prompt.match(/\S+|\s+/g) || [prompt]
    const promptTokens = words.map((w, idx) => ({
      token_id: hashString(w) % 50257,
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
    await delay(80 + rng() * 60)
    if (this.aborted) return

    // 3. Check for direct Gemini API call from browser
    let generatedWords: string[] | null = null
    const geminiKey = typeof window !== 'undefined'
      ? (localStorage.getItem('gemini_api_key') || BUILT_IN_GEMINI_KEY || null)
      : null

    if (geminiKey) {
      // Try multiple model names (older ones get deprecated)
      const modelCandidates = [
        'gemini-3.8-flash',
        'gemini-flash-latest',
        'gemini-3.7-flash',
        'gemini-3.5-flash',
      ]
      for (const modelName of modelCandidates) {
        try {
          const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${geminiKey}`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: {
                  maxOutputTokens: maxTokens * 4,
                  temperature: config.temperature ?? 0.7,
                },
              }),
            }
          )
          const resJson = await response.json()
          const text = resJson?.candidates?.[0]?.content?.parts?.[0]?.text
          if (text) {
            const rawWords = text.match(/\S+|\s+/g) || []
            generatedWords = rawWords.map((w: string) => w.startsWith(' ') ? w : ' ' + w)
            break  // Success — stop trying other models
          }
          // If 404 or error, try next model
          if (resJson?.error?.code === 404) continue
          if (resJson?.error?.code === 503) continue
        } catch (err) {
          console.warn(`Gemini model ${modelName} failed:`, err)
          continue
        }
      }
    }

    if (!generatedWords || generatedWords.length === 0) {
      generatedWords = selectResponse(prompt, rng)
    }

    const numTokensToGen = Math.min(maxTokens, generatedWords.length)

    // 4. Autoregressive Generation Loop
    for (let step = 0; step < numTokensToGen; step++) {
      if (this.aborted) return

      // Process all 12 transformer layers
      for (let l = 0; l < 12; l++) {
        if (this.aborted) return

        emit({
          event_id: `evt-${Date.now()}-s${step}l${l}s`,
          session_id: sessionId,
          event_type: 'transformer_layer_started',
          timestamp: new Date().toISOString(),
          generation_step: step,
          layer_index: l,
          data: {},
          tensor_references: [],
        })

        await delay(10 + rng() * 15)

        emit({
          event_id: `evt-${Date.now()}-s${step}l${l}c`,
          session_id: sessionId,
          event_type: 'transformer_layer_completed',
          timestamp: new Date().toISOString(),
          generation_step: step,
          layer_index: l,
          data: {},
          tensor_references: [],
        })
      }

      // Build token candidates
      const nextWord = generatedWords[step]
      const chosenId = hashString(nextWord + step.toString()) % 50257

      const alternates = buildCandidatePool(nextWord, rng)
      const chosenLogit = 14.5 + rng() * 2
      const allCandidates = [
        { token: nextWord, id: chosenId, logit: chosenLogit },
        ...alternates,
      ]

      // Compute softmax probabilities
      const maxLogit = Math.max(...allCandidates.map(c => c.logit))
      const exps = allCandidates.map(c => Math.exp(c.logit - maxLogit))
      const sumExp = exps.reduce((a, b) => a + b, 0)
      const probs = exps.map(e => e / sumExp)

      const topTokens = allCandidates.map((c, idx) => ({
        token_id: c.id,
        token_string: c.token,
        logit: Math.round(c.logit * 1000) / 1000,
        probability: Math.round(probs[idx] * 10000) / 10000,
        rank: idx + 1,
      }))

      emit({
        event_id: `evt-${Date.now()}-s${step}prob`,
        session_id: sessionId,
        event_type: 'probabilities_calculated',
        timestamp: new Date().toISOString(),
        generation_step: step,
        data: {
          top_tokens: topTokens,
          sampling_strategy: config.temperature === 0 ? 'greedy' : 'multinomial',
        },
        tensor_references: [],
      })

      emit({
        event_id: `evt-${Date.now()}-s${step}sel`,
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
        event_id: `evt-${Date.now()}-s${step}gen`,
        session_id: sessionId,
        event_type: 'token_generated',
        timestamp: new Date().toISOString(),
        generation_step: step,
        data: {
          token_string: nextWord,
          token_id: chosenId,
          probability: probs[0],
        },
        tensor_references: [],
      })

      // Realistic variable delay between tokens
      await delay(50 + rng() * 80)
    }

    // 5. Generation Completed
    emit({
      event_id: `evt-${Date.now()}-done`,
      session_id: sessionId,
      event_type: 'generation_completed',
      timestamp: new Date().toISOString(),
      generation_step: numTokensToGen,
      data: {
        total_tokens: numTokensToGen,
        duration_ms: numTokensToGen * (120 + Math.floor(rng() * 60)),
      },
      tensor_references: [],
    })
  }
}
