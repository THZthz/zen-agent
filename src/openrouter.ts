import type { ThinkingEffort } from "./storage.js";
import {
  runChatCompletions,
  type LlmStepOptions,
  type LlmStepResult,
  type LlmUsage,
} from "./llm-client.js";

/** Default OpenRouter model used when OPENROUTER_MODEL is unset. */
export const DEFAULT_OPENROUTER_MODEL = "openrouter/free";

function getOpenRouterApiKey(): string {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY environment variable is required");
  }
  return apiKey;
}

function getOpenRouterBaseUrl(): string {
  return (
    process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1"
  ).replace(/\/+$/, "");
}

/**
 * Per-model metadata (USD per 1M tokens, context window).
 *
 * The live `GET /models` endpoint is the source of truth (fetched once and
 * cached); the fallbacks below only apply when that fetch fails (offline
 * start, invalid key, ...) or for models that left the catalog. The values
 * are indicative snapshots, not guaranteed current.
 */
export interface OpenRouterModelInfo {
  /** USD per 1M input tokens. */
  inputPerM: number;
  /** USD per 1M output tokens. */
  outputPerM: number;
  contextLength: number;
}

const MODEL_FALLBACKS: Record<string, OpenRouterModelInfo> = {
  // `openrouter/free` routes to free models, billed at $0.
  "openrouter/free": { inputPerM: 0, outputPerM: 0, contextLength: 128_000 },
};

/** Conservative default for models unknown to both the live catalog and the fallback table. */
const UNKNOWN_MODEL_FALLBACK: OpenRouterModelInfo = {
  inputPerM: 1,
  outputPerM: 2,
  contextLength: 200_000,
};

function parsePrice(raw: string | undefined): number {
  const parsed = Number.parseFloat(raw ?? "");
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

let modelsCache: { key: string; promise: Promise<Map<string, OpenRouterModelInfo>> } | null =
  null;

/** Test hook: drop the cached /models response (env/port changes between tests). */
export function resetOpenRouterModelsCache(): void {
  modelsCache = null;
}

/**
 * Fetch OpenRouter's model catalog once per (base URL, key) and cache it.
 * Never throws: callers fall back to the curated table when the fetch fails.
 */
async function fetchOpenRouterModels(): Promise<Map<string, OpenRouterModelInfo>> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const key = `${baseUrl}|${apiKey}`;
  if (modelsCache?.key === key) {
    return modelsCache.promise;
  }
  const promise = (async () => {
    const response = await fetch(`${baseUrl}/models`, {
      headers: { authorization: `Bearer ${apiKey}` },
    });
    if (!response.ok) {
      throw new Error(
        `OpenRouter models API error ${response.status}: ${(await response.text().catch(() => "")).slice(0, 500)}`,
      );
    }
    const data = (await response.json()) as {
      data?: Array<{
        id?: string;
        context_length?: number;
        pricing?: { prompt?: string; completion?: string; request?: string };
      }>;
    };
    const models = new Map<string, OpenRouterModelInfo>();
    for (const model of data.data ?? []) {
      if (!model.id) {
        continue;
      }
      models.set(model.id, {
        inputPerM: parsePrice(model.pricing?.prompt),
        outputPerM: parsePrice(model.pricing?.completion),
        contextLength:
          typeof model.context_length === "number" && model.context_length > 0
            ? model.context_length
            : 200_000,
      });
    }
    return models;
  })();
  modelsCache = { key, promise };
  return promise;
}

/**
 * Effective pricing/context info for an OpenRouter model id. Prefers the
 * live catalog; falls back to the static table, then to generic defaults.
 */
export async function getOpenRouterModelInfo(model: string): Promise<OpenRouterModelInfo> {
  try {
    const info = (await fetchOpenRouterModels()).get(model);
    if (info) {
      return info;
    }
  } catch {
    // Offline start, bad key, ... — use the static fallbacks.
  }
  return MODEL_FALLBACKS[model] ?? UNKNOWN_MODEL_FALLBACK;
}

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
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
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
  const cacheMissTokens = Math.max(0, inputTokens - cacheReadTokens);
  const reasoningTokens = toNumber(usage.completion_tokens_details?.reasoning_tokens);

  if (inputTokens === 0 && outputTokens === 0 && cacheReadTokens === 0) {
    return null;
  }

  return {
    inputTokens,
    outputTokens,
    totalTokens: toNumber(usage.total_tokens) || inputTokens + outputTokens,
    cacheReadTokens,
    cacheMissTokens,
    reasoningTokens,
    llmMs: timing.llmMs,
    thinkingMs: timing.thinkingMs,
    answeringMs: timing.answeringMs,
  };
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
      `OpenRouter auth/key API error ${response.status}: ${(await response.text().catch(() => "")).slice(0, 500)}`,
    );
  }
  const data = (await response.json()) as {
    data?: { label?: string; usage?: number; limit?: number; is_free_tier?: boolean };
  };
  const usageUsd = toNumber(data.data?.usage);
  const limitUsd = toNumber(data.data?.limit);
  return {
    isAvailable: true,
    currency: "USD",
    remainingUsd: Math.max(0, limitUsd - usageUsd),
    usageUsd,
    limitUsd,
    isFreeTier: data.data?.is_free_tier ?? false,
  };
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
 * - `reasoning_effort` uses the OpenAI vocabulary (low/medium/high), so
 *   DeepSeek's `max` maps to `high`
 * - optional `HTTP-Referer` / `X-Title` headers identify the app
 *   (OPENROUTER_SITE_URL / OPENROUTER_APP_NAME)
 */
export async function runOpenRouterStep(options: LlmStepOptions): Promise<LlmStepResult> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const modelName =
    options.model ?? process.env.OPENROUTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;

  const extraHeaders: Record<string, string> = {};
  const siteUrl = process.env.OPENROUTER_SITE_URL;
  if (siteUrl) {
    extraHeaders["HTTP-Referer"] = siteUrl;
  }
  const appName = process.env.OPENROUTER_APP_NAME;
  if (appName) {
    extraHeaders["X-Title"] = appName;
  }

  return runChatCompletions({
    baseUrl,
    apiKey,
    label: "OpenRouter",
    model: modelName,
    messages: options.messages,
    system: options.system,
    signal: options.signal,
    onTextDelta: options.onTextDelta,
    onReasoningDelta: options.onReasoningDelta,
    thinkingEffort: options.thinkingEffort,
    reasoningMessageField: "reasoning",
    reasoningDeltaFields: ["reasoning", "reasoning_content"],
    effortBody: (effort: ThinkingEffort) =>
      effort === "off"
        ? undefined
        : { reasoning_effort: effort === "max" ? "high" : effort },
    extraBody: { stream_options: { include_usage: true } },
    extraHeaders,
    parseUsage: (raw, timing) => parseOpenRouterUsage(raw as OpenRouterUsage, timing),
  });
}
