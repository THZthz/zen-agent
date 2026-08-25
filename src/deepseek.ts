import type { ModelId } from './storage.js';
import { envNonNegativeFloat, envPositiveInt } from './env.js';
import {
  buildLlmUsage,
  costFromUsage,
  runChatCompletions,
  type LlmStepOptions,
  type LlmStepResult,
  type LlmUsage,
} from './llm-client.js';
export { SYSTEM_PROMPT } from './system-prompt.js';
export { BASH_TOOL_SCHEMA } from './llm-client.js';

export type { LlmStepResult, LlmStepOptions, LlmToolCall, LlmUsage } from './llm-client.js';

export interface ModelPricing {
  /** CNY per 1M input tokens served from cache. */
  cacheHitCnyPerM: number;
  /** CNY per 1M input tokens not served from cache. */
  cacheMissCnyPerM: number;
  /** CNY per 1M output tokens. */
  outputCnyPerM: number;
}

interface ModelRateTable {
  /** CNY per 1M input tokens served from cache. */
  cacheHit: { peak: number; offPeak: number };
  /** CNY per 1M input tokens not served from cache. */
  cacheMiss: { peak: number; offPeak: number };
  /** CNY per 1M output tokens. */
  output: { peak: number; offPeak: number };
}

/**
 * Official DeepSeek pricing (CNY per 1M tokens) for the V4 models.
 * Off-peak price is half the peak price. Peak hours are Beijing time
 * 09:00-12:00 and 14:00-18:00; all other hours are off-peak.
 * Values can be overridden with DEEPSEEK_PRICE_* environment variables.
 */
const MODEL_RATE_TABLE: Record<string, ModelRateTable> = {
  'deepseek-v4-flash': {
    cacheHit: { peak: 0.1, offPeak: 0.05 },
    cacheMiss: { peak: 3.0, offPeak: 1.5 },
    output: { peak: 9.0, offPeak: 4.5 },
  },
  'deepseek-v4-pro': {
    cacheHit: { peak: 0.3, offPeak: 0.15 },
    cacheMiss: { peak: 9.0, offPeak: 4.5 },
    output: { peak: 27.0, offPeak: 13.5 },
  },
};

/**
 * Whether `now` (UTC) falls in DeepSeek's peak pricing window.
 * Peak hours are Beijing time (UTC+8) 09:00-12:00 and 14:00-18:00;
 * 12:00 and 18:00 themselves are off-peak.
 */
export function isPeakTime(now: Date = new Date()): boolean {
  // Beijing time is always UTC+8 (no DST); read the shifted UTC clock directly.
  const beijing = new Date(now.getTime() + 8 * 3_600_000);
  const minutes = beijing.getUTCHours() * 60 + beijing.getUTCMinutes();
  return (minutes >= 9 * 60 && minutes < 12 * 60) || (minutes >= 14 * 60 && minutes < 18 * 60);
}

/** Effective pricing for a model at `now`, including peak/off-peak selection. */
export function getModelPricing(model: ModelId, now: Date = new Date()): ModelPricing {
  const base = MODEL_RATE_TABLE[model] ?? MODEL_RATE_TABLE['deepseek-v4-flash'];
  const peak = isPeakTime(now);
  return {
    cacheHitCnyPerM: envNonNegativeFloat(
      'DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK',
      peak ? base.cacheHit.peak : base.cacheHit.offPeak,
    ),
    cacheMissCnyPerM: envNonNegativeFloat(
      'DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK',
      peak ? base.cacheMiss.peak : base.cacheMiss.offPeak,
    ),
    outputCnyPerM: envNonNegativeFloat(
      'DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK',
      peak ? base.output.peak : base.output.offPeak,
    ),
  };
}

/** Session context window size in tokens, used for usage_update.size. */
export function getContextWindowTokens(): number {
  return envPositiveInt('DEEPSEEK_CONTEXT_WINDOW', 1_000_000);
}

/** Cost in CNY for a single LLM step's token usage. */
export function costYuan(usage: LlmUsage, pricing: ModelPricing): number {
  return costFromUsage(usage, {
    currency: 'CNY',
    cacheHitPerM: pricing.cacheHitCnyPerM,
    cacheMissPerM: pricing.cacheMissCnyPerM,
    outputPerM: pricing.outputCnyPerM,
  });
}

/**
 * Account balance snapshot from DeepSeek's `GET /user/balance` endpoint.
 *
 * Used to cross-check the locally estimated token cost (`cost`, in CNY) against
 * what DeepSeek actually bills: the balance delta across a turn should match
 * the turn's estimated cost. Balance values come back as strings with two
 * decimals, so a single turn's delta is only accurate to ~¥0.01.
 */
export interface DeepSeekBalance {
  isAvailable: boolean;
  currency: string;
  totalBalanceCny: number;
  grantedBalanceCny: number;
  toppedUpBalanceCny: number;
}

/**
 * Fetch the current account balance from DeepSeek. Honors DEEPSEEK_BASE_URL
 * (used by the integration tests) and fails loudly on HTTP errors so callers
 * can log a verification failure instead of silently comparing stale data.
 */
export async function fetchDeepSeekBalance(
  opts: { signal?: AbortSignal } = {},
): Promise<DeepSeekBalance> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error('DEEPSEEK_API_KEY environment variable is required');
  }
  const baseURL = (process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com').replace(/\/+$/, '');
  const response = await fetch(`${baseURL}/user/balance`, {
    headers: { authorization: `Bearer ${apiKey}` },
    signal: opts.signal,
  });
  if (!response.ok) {
    throw new Error(
      `DeepSeek balance API error ${response.status}: ${(await response.text().catch(() => '')).slice(0, 500)}`,
    );
  }
  const data = (await response.json()) as {
    is_available?: boolean;
    balance_infos?: Array<{
      currency?: string;
      total_balance?: string;
      granted_balance?: string;
      topped_up_balance?: string;
    }>;
  };
  const info =
    data.balance_infos?.find((entry) => entry.currency === 'CNY') ?? data.balance_infos?.[0];
  if (!info || info.total_balance === undefined) {
    throw new Error(`Unexpected balance API response: ${JSON.stringify(data)}`);
  }
  return {
    isAvailable: data.is_available ?? false,
    currency: info.currency ?? 'CNY',
    totalBalanceCny: Number.parseFloat(info.total_balance),
    grantedBalanceCny: Number.parseFloat(info.granted_balance ?? '0'),
    toppedUpBalanceCny: Number.parseFloat(info.topped_up_balance ?? '0'),
  };
}

/**
 * DeepSeek's streaming usage object. We read the cache tokens from the raw
 * chunk usage: DeepSeek reports `prompt_cache_hit_tokens` /
 * `prompt_cache_miss_tokens` which OpenAI's schema (and the AI SDK's zod
 * parser) strips, so they must come straight from the wire.
 */
interface DeepSeekUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  completion_tokens_details?: { reasoning_tokens?: number };
}

function toNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function parseDeepSeekUsage(
  usage: DeepSeekUsage | undefined,
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (!usage) {
    return null;
  }
  const inputTokens = toNumber(usage.prompt_tokens);
  const outputTokens = toNumber(usage.completion_tokens);
  const cacheReadTokens = toNumber(usage.prompt_cache_hit_tokens);
  // DeepSeek reports prompt_cache_miss_tokens explicitly; when absent (some
  // proxies/models), fall back to input - cache-hit.
  const cacheMissTokens =
    usage.prompt_cache_miss_tokens !== undefined
      ? toNumber(usage.prompt_cache_miss_tokens)
      : Math.max(0, inputTokens - cacheReadTokens);
  const reasoningTokens = toNumber(usage.completion_tokens_details?.reasoning_tokens);

  return buildLlmUsage(
    {
      inputTokens,
      outputTokens,
      totalTokens: toNumber(usage.total_tokens),
      cacheReadTokens,
      cacheMissTokens,
      reasoningTokens,
    },
    timing,
  );
}

/**
 * DeepSeek's OpenAI-compatible chat completions step. The SSE client itself
 * is shared with the OpenRouter provider (see runChatCompletions); this
 * wrapper supplies DeepSeek's endpoint, reasoning field (`reasoning_content`)
 * and usage parsing.
 */
export async function runLlmStep(options: LlmStepOptions): Promise<LlmStepResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error('DEEPSEEK_API_KEY environment variable is required');
  }
  const baseURL = (process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com').replace(/\/+$/, '');
  const modelName = options.model ?? process.env.DEEPSEEK_MODEL ?? 'deepseek-v4-flash';
  return runChatCompletions({
    baseUrl: baseURL,
    apiKey,
    label: 'DeepSeek',
    model: modelName,
    messages: options.messages,
    tools: options.tools,
    system: options.system,
    signal: options.signal,
    onTextDelta: options.onTextDelta,
    onReasoningDelta: options.onReasoningDelta,
    logRuntime: options.logRuntime,
    thinkingEffort: options.thinkingEffort,
    reasoningMessageField: 'reasoning_content',
    reasoningDeltaFields: ['reasoning_content'],
    effortBody: (effort) => (effort === 'off' ? undefined : { reasoning_effort: effort }),
    parseUsage: (raw, timing) => parseDeepSeekUsage(raw as DeepSeekUsage, timing),
  });
}
