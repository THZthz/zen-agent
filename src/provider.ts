import { type GenericPricing, type LlmStepOptions, type LlmStepResult } from './llm-client.js';
import { envNonNegativeFloat } from './env.js';
import { getPiModel, getSupportedThinkingEfforts, ensureProviderRefreshed } from './provider-pi.js';
import {
  requireProviderDefinition,
  resolveApiKey,
  type BalanceSnapshot,
  type ProviderDefinition,
} from './provider-registry.js';
import { fetchProviderBalance } from './provider-balances.js';
import { getCatalog } from './provider-catalog.js';
import { runChatCompletions } from './chat-completions.js';
import type { ModelId, ProviderId, ThinkingEffort } from './storage.js';

export type { LlmStepResult, LlmStepOptions, LlmToolCall, LlmUsage } from './llm-client.js';
export { costFromUsage } from './llm-client.js';
export type { BalanceSnapshot } from './provider-registry.js';
export { getModelOptions, type ModelOption } from './provider-pi.js';

/** Default model for a provider: its configured fallback (env-driven). */
export function getDefaultModel(provider: ProviderId): ModelId {
  return requireProviderDefinition(provider).defaultModel;
}

/** Billing currency of the session's provider (CNY for DeepSeek, USD for OpenRouter). */
export function getProviderCurrency(provider: ProviderId): string {
  return requireProviderDefinition(provider).currency;
}

/** Human display name for a provider (used in diagnostics). */
export function getProviderName(provider: ProviderId): string {
  return requireProviderDefinition(provider).name;
}

/** Run one LLM step with the session's provider (per-session, not process-wide). */
export async function runLlmStep(
  provider: ProviderId,
  options: LlmStepOptions,
): Promise<LlmStepResult> {
  const def = requireProviderDefinition(provider);
  // Discovery providers load their catalog once per process (best-effort,
  // 5s timeout); static providers skip this entirely.
  await ensureProviderRefreshed(provider);
  const model = getPiModel(provider, options.model ?? def.defaultModel);
  return runChatCompletions({
    model,
    apiKey: resolveApiKey(def),
    label: def.label,
    messages: options.messages,
    tools: options.tools,
    system: options.system,
    signal: options.signal,
    onTextDelta: options.onTextDelta,
    onReasoningDelta: options.onReasoningDelta,
    logRuntime: options.logRuntime,
    thinkingEffort: options.thinkingEffort,
    extraBody: buildExtraBody(def, options.sessionId),
    sessionId: def.sendSessionId ? options.sessionId : undefined,
  });
}

/** Merge provider-declared body fields with the session affinity field. */
function buildExtraBody(
  def: ProviderDefinition,
  sessionId: string | undefined,
): Record<string, unknown> | undefined {
  const body: Record<string, unknown> = { ...def.extraBody };
  if (def.sendSessionId && sessionId) {
    body.session_id = sessionId;
  }
  return Object.keys(body).length > 0 ? body : undefined;
}

/**
 * Effective pricing for a model on a provider, in the provider's billing
 * currency. Static tables (DeepSeek) honor peak/off-peak windows; catalog
 * providers use the discovered per-model prices.
 */
export async function getModelPricing(
  provider: ProviderId,
  model: ModelId,
  now: Date = new Date(),
): Promise<GenericPricing> {
  const def = requireProviderDefinition(provider);
  if (def.pricing.kind === 'table') {
    const rates = def.pricing.rates[model] ?? def.pricing.rates[def.pricing.defaultModel]!;
    const peak = isPeakTime(now);
    return {
      currency: def.currency,
      cacheHitPerM: envNonNegativeFloat(
        'DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK',
        peak ? rates.cacheHit.peak : rates.cacheHit.offPeak,
      ),
      cacheMissPerM: envNonNegativeFloat(
        'DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK',
        peak ? rates.cacheMiss.peak : rates.cacheMiss.offPeak,
      ),
      outputPerM: envNonNegativeFloat(
        'DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK',
        peak ? rates.output.peak : rates.output.offPeak,
      ),
    };
  }
  await ensureProviderRefreshed(provider);
  const piModel = getPiModel(provider, model);
  return {
    currency: def.currency,
    cacheHitPerM: piModel.cost.cacheRead,
    cacheMissPerM: piModel.cost.input,
    outputPerM: piModel.cost.output,
  };
}

/**
 * Whether `now` (UTC) falls in DeepSeek's peak pricing window.
 * Peak hours are Beijing time (UTC+8) 09:00-12:00 and 14:00-18:00;
 * 12:00 and 18:00 themselves are off-peak.
 */
export function isPeakTime(now: Date = new Date()): boolean {
  const beijing = new Date(now.getTime() + 8 * 3_600_000);
  const minutes = beijing.getUTCHours() * 60 + beijing.getUTCMinutes();
  return (minutes >= 9 * 60 && minutes < 12 * 60) || (minutes >= 14 * 60 && minutes < 18 * 60);
}

/** Session context window size in tokens, used for usage_update.size. */
export async function getContextWindowTokens(
  provider: ProviderId,
  model: ModelId,
): Promise<number> {
  // Discovery providers load their catalog first so catalog-known models get
  // their real context window; static providers skip this entirely.
  await ensureProviderRefreshed(provider);
  return getPiModel(provider, model).contextWindow;
}

/**
 * Input modalities accepted by the session's model.
 *
 * Returns `null` when the answer is TEMPORARILY unknown — a discovery
 * provider whose catalog has not been fetched or whose slug is not in any
 * table — so callers can retry later instead of caching a wrong "text-only"
 * answer for the whole session (a memoized negative would permanently hide
 * read_media from the model). Static providers (DeepSeek) are text-only by
 * definition, so that branch is always definitive.
 */
export async function getModelModalities(
  provider: ProviderId,
  model: ModelId,
): Promise<{ image: boolean; audio: boolean } | null> {
  const def = requireProviderDefinition(provider);
  if (!def.discovery.enabled) {
    return { image: false, audio: false };
  }
  // Static fallback models (e.g. openrouter/free offline) are known text-only.
  if (def.staticModels.some((opt) => opt.value === model)) {
    return { image: false, audio: false };
  }
  await ensureProviderRefreshed(provider);
  const entry = getCatalog(provider)?.get(model) ?? null;
  if (!entry) {
    return null;
  }
  return {
    image: entry.inputModalities?.includes('image') ?? false,
    audio: entry.inputModalities?.includes('audio') ?? false,
  };
}

/**
 * Conservative boolean view of {@link getModelModalities}: an unknown lookup
 * degrades to "no media support" instead of null. Use this where a decision
 * cannot wait for a retry (e.g. degrading attached media in a prompt); use
 * getModelModalities directly where the result may be cached per session.
 */
export async function resolveModelModalities(
  provider: ProviderId,
  model: ModelId,
): Promise<{ image: boolean; audio: boolean }> {
  return (await getModelModalities(provider, model)) ?? { image: false, audio: false };
}

/** Thinking-effort values the session selector offers for a provider/model. */
export async function getThinkingEfforts(
  provider: ProviderId,
  model: ModelId,
): Promise<readonly ThinkingEffort[]> {
  return getSupportedThinkingEfforts(provider, model);
}

/** Balance/credit snapshot for the active provider (best-effort). */
export async function fetchBalanceSnapshot(provider: ProviderId): Promise<BalanceSnapshot> {
  return fetchProviderBalance(requireProviderDefinition(provider));
}

/** Registry helpers re-exported for config surfaces that need them. */
export { getProviderDefinition, isKnownProvider } from './provider-registry.js';
