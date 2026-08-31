import { runChatCompletions } from './chat-completions.js';
import { type LlmStepOptions, type LlmStepResult } from './llm-client.js';
import { DEFAULT_OPENROUTER_MODEL, type ThinkingEffort } from './storage.js';
import {
  getOpenRouterApiKey,
  getOpenRouterBaseUrl,
  getOpenRouterProviderSort,
  getOpenRouterReasoning,
} from './openrouter-models.js';

export { DEFAULT_OPENROUTER_MODEL } from './storage.js';

function finiteNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
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
  const usageUsd = finiteNumber(data.data?.usage);
  const limitUsd = finiteNumber(data.data?.limit);
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
 * OpenRouter's OpenAI-compatible chat-completions step. Pi owns the request
 * and stream protocol, including normalized reasoning deltas and usage. Zen
 * supplies gateway routing, its dynamic effort allowlist, and app headers.
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

  const requestedEffort = options.thinkingEffort ?? 'off';
  const thinkingLevelMap =
    requestedEffort === 'off'
      ? { off: reasoningEffort }
      : reasoningEffort === null
        ? undefined
        : { [requestedEffort]: reasoningEffort };

  return runChatCompletions({
    baseUrl,
    apiKey,
    provider: 'openrouter',
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
    thinkingLevelMap,
    compat: {
      supportsDeveloperRole: false,
      thinkingFormat: 'openrouter',
    },
    extraBody: {
      ...(providerSort ? { provider: { sort: providerSort } } : {}),
      // OpenRouter's provider sticky routing keeps consecutive requests of a
      // conversation on the same provider endpoint so the upstream context
      // cache stays warm. Passing the zen-agent session id pins that routing
      // from the first request and gives Z.AI a session affinity key.
      ...(options.sessionId ? { session_id: options.sessionId } : {}),
    },
    extraHeaders,
    sessionId: options.sessionId,
  });
}
