import { runChatCompletions } from './chat-completions.js';
import {
  buildLlmUsage,
  type LlmStepOptions,
  type LlmStepResult,
  type LlmUsage,
} from './llm-client.js';
import { DEFAULT_OPENROUTER_MODEL, type ThinkingEffort } from './storage.js';
import {
  getOpenRouterApiKey,
  getOpenRouterBaseUrl,
  getOpenRouterProviderSort,
  getOpenRouterReasoning,
} from './openrouter-models.js';

export { DEFAULT_OPENROUTER_MODEL } from './storage.js';

/** OpenRouter's streaming usage object (normalized OpenAI shape). */
interface OpenRouterUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  /** Provider passthrough — DeepSeek routes report cache hits this way. */
  prompt_cache_hit_tokens?: number;
  /** Anthropic-style passthrough of cached input tokens. */
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

function toNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * Parse OpenRouter's usage chunk. Cache-hit and reasoning-token fields are
 * only present when the upstream provider passes them through; everything
 * else falls back to 0 (stats then show cache hit 0% / no reasoning tokens,
 * which is accurate for providers that do not report them).
 */
export function parseOpenRouterUsage(
  usage: OpenRouterUsage | undefined,
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (!usage) {
    return null;
  }
  const inputTokens = toNumber(usage.prompt_tokens);
  const outputTokens = toNumber(usage.completion_tokens);
  const cacheReadTokens =
    usage.prompt_cache_hit_tokens !== undefined
      ? toNumber(usage.prompt_cache_hit_tokens)
      : toNumber(usage.prompt_tokens_details?.cached_tokens);
  return buildLlmUsage(
    {
      inputTokens,
      outputTokens,
      totalTokens: toNumber(usage.total_tokens),
      cacheReadTokens,
      cacheMissTokens: Math.max(0, inputTokens - cacheReadTokens),
      reasoningTokens: toNumber(usage.completion_tokens_details?.reasoning_tokens),
    },
    timing,
  );
}

/** OpenRouter key info from `GET /auth/key`, in USD credits. */
export interface OpenRouterBalance {
  isAvailable: boolean;
  currency: string;
  /** Remaining credits: the key's limit minus its usage. */
  remainingUsd: number;
  usageUsd: number;
  limitUsd: number;
  isFreeTier: boolean;
}

/**
 * Fetch the current key usage/limit from OpenRouter's `GET /auth/key`.
 * Used to cross-check the locally estimated token cost against actual
 * credit consumption (see ZenAgent.verifyTurnCost).
 */
export async function fetchOpenRouterBalance(): Promise<OpenRouterBalance> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const response = await fetch(`${baseUrl}/auth/key`, {
    headers: { authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    throw new Error(
      `OpenRouter auth/key API error ${response.status}: ${(await response.text().catch(() => '')).slice(0, 500)}`,
    );
  }
  const data = (await response.json()) as {
    data?: { label?: string; usage?: number; limit?: number; is_free_tier?: boolean };
  };
  const usageUsd = toNumber(data.data?.usage);
  const limitUsd = toNumber(data.data?.limit);
  return {
    isAvailable: true,
    currency: 'USD',
    remainingUsd: Math.max(0, limitUsd - usageUsd),
    usageUsd,
    limitUsd,
    isFreeTier: data.data?.is_free_tier ?? false,
  };
}

/**
 * OpenRouter's gateway-wide `reasoning_effort` ladder, highest first
 * (mirrors the catalog's `supported_efforts` ordering and the API schema).
 * `none` disables reasoning; it is the wire equivalent of the session's
 * `off` value.
 */
export const OPENROUTER_EFFORT_LADDER: readonly string[] = [
  'max',
  'xhigh',
  'high',
  'medium',
  'low',
  'minimal',
  'none',
];

/** Gateway effort values excluding the disable value (session selector list). */
export const OPENROUTER_EFFORT_VALUES: readonly string[] = OPENROUTER_EFFORT_LADDER.filter(
  (effort) => effort !== 'none',
);

/**
 * Map a session thinking effort to the `reasoning_effort` value sent to an
 * OpenRouter model, honoring the model's `supported_efforts` allowlist from
 * the catalog:
 *
 * - `off` sends `none` when the model supports it; on mandatory-reasoning
 *   models (`mandatory: true`, no `none` in the allowlist) it sends the
 *   model's LOWEST supported effort — the closest the model can get to
 *   disabled. Non-mandatory models without a `none` tier, and unknown
 *   allowlists (offline start / unknown slug), omit the field so the
 *   provider's default (usually off/low) applies.
 * - any other value is sent unchanged when the allowlist is unknown or
 *   contains it; otherwise it is remapped to the nearest supported effort by
 *   ladder distance (ties resolve toward the HIGHER effort, so `medium` on a
 *   `[max, high, low]` model becomes `high`, not `low`).
 *
 * Returns null when the field should be omitted entirely.
 */
export function mapOpenRouterEffort(
  effort: ThinkingEffort,
  supportedEfforts: readonly string[] | null,
  mandatory = false,
): string | null {
  if (supportedEfforts === null) {
    // Unknown model/catalog: every gateway value is accepted, `off` omits
    // the field (the provider picks its default).
    return effort === 'off' ? null : effort;
  }
  if (effort === 'off') {
    if (supportedEfforts.includes('none')) {
      return 'none';
    }
    // Mandatory-reasoning models cannot disable thinking: `none` is absent
    // from the allowlist and `mandatory` is true. Fall back to the model's
    // lowest supported effort (the gateway's own behavior for unsupported
    // `none` on `~latest` slugs). Non-mandatory models simply omit the field
    // so their default (usually off/low) applies.
    if (mandatory && supportedEfforts.length > 0) {
      return supportedEfforts[supportedEfforts.length - 1];
    }
    return null;
  }
  if (supportedEfforts.includes(effort)) {
    return effort;
  }
  const requested = OPENROUTER_EFFORT_LADDER.indexOf(effort);
  if (requested === -1) {
    return null;
  }
  // Nearest by ladder distance; ties resolve to the higher effort
  // (supportedEfforts is ordered highest-first, so the first candidate at a
  // given distance wins).
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of supportedEfforts) {
    const index = OPENROUTER_EFFORT_LADDER.indexOf(candidate);
    if (index === -1) {
      continue;
    }
    const distance = Math.abs(index - requested);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
}

/**
 * OpenRouter's chat completions step (OpenAI-compatible). Shares the SSE
 * client with DeepSeek; OpenRouter-specific bits:
 *
 * - reasoning streams as `delta.reasoning` (OpenRouter's normalized field;
 *   `delta.reasoning_content` is also accepted for DeepSeek passthrough
 *   routes), and stored reasoning is sent back as `reasoning` in history
 * - `stream_options.include_usage` requests the final usage chunk (OpenRouter
 *   does not send usage otherwise)
 * - `provider.sort` defaults to `price` so requests route to the cheapest
 *   provider for the model (OPENROUTER_PROVIDER_SORT overrides; empty disables)
 * - `reasoning_effort` honors the model's `supported_efforts` allowlist from
 *   the catalog (see {@link mapOpenRouterEffort}): the session's `off`/`high`/
 *   `max` map to the model's own vocabulary (`none`/`minimal`/.../`xhigh`),
 *   so e.g. `max` reaches models that support `xhigh` or `max` natively
 *   instead of being collapsed to `high`
 * - optional `HTTP-Referer` / `X-Title` headers identify the app
 *   (OPENROUTER_SITE_URL / OPENROUTER_APP_NAME)
 */
export async function runOpenRouterStep(options: LlmStepOptions): Promise<LlmStepResult> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const modelName = options.model ?? process.env.OPENROUTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;

  const extraHeaders: Record<string, string> = {};
  const siteUrl = process.env.OPENROUTER_SITE_URL;
  if (siteUrl) {
    extraHeaders['HTTP-Referer'] = siteUrl;
  }
  const appName = process.env.OPENROUTER_APP_NAME;
  if (appName) {
    extraHeaders['X-Title'] = appName;
  }
  const providerSort = getOpenRouterProviderSort();
  // Best-effort catalog lookup (cached in memory after the first fetch): a
  // failed/offline lookup degrades to "all effort values accepted".
  const reasoning = await getOpenRouterReasoning(modelName);
  const reasoningEffort = mapOpenRouterEffort(
    options.thinkingEffort ?? 'off',
    reasoning.supportedEfforts,
    reasoning.mandatory === true,
  );

  return runChatCompletions({
    baseUrl,
    apiKey,
    label: 'OpenRouter',
    model: modelName,
    messages: options.messages,
    tools: options.tools,
    system: options.system,
    signal: options.signal,
    onTextDelta: options.onTextDelta,
    onReasoningDelta: options.onReasoningDelta,
    logRuntime: options.logRuntime,
    thinkingEffort: options.thinkingEffort,
    reasoningMessageField: 'reasoning',
    reasoningDeltaFields: ['reasoning', 'reasoning_content'],
    effortBody: () =>
      reasoningEffort === null ? undefined : { reasoning_effort: reasoningEffort },
    extraBody: {
      stream_options: { include_usage: true },
      ...(providerSort ? { provider: { sort: providerSort } } : {}),
      // OpenRouter's provider sticky routing keeps consecutive requests of a
      // conversation on the same provider endpoint so the upstream context
      // cache stays warm. Without a session_id it only activates AFTER a
      // cache hit is observed, and the conversation identity falls back to a
      // hash of the first system + first non-system message. Both break when
      // read_media injects image/audio content mid-turn: the request becomes
      // multimodal and can route to a different provider (GLM models have
      // several hosts on OpenRouter), so the previously cached prefix misses
      // and the hit ratio drops to 0. Passing the zen-agent sessionId pins
      // routing from the FIRST request and gives Z.AI a session affinity key
      // (OpenRouter docs, "Prompt Caching" -> Z.AI).
      ...(options.sessionId ? { session_id: options.sessionId } : {}),
    },
    extraHeaders,
    parseUsage: (raw, timing) => parseOpenRouterUsage(raw as OpenRouterUsage, timing),
  });
}
