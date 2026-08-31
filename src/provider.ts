import { type GenericPricing, type LlmStepOptions, type LlmStepResult } from './llm-client.js';
import { getPiModel, getSupportedThinkingEfforts, ensureProviderRefreshed } from './provider-pi.js';
import {
  mapModelThinkingEffort,
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

/** Default model for a provider: its declared fallback. */
export function getDefaultModel(provider: ProviderId): ModelId {
  return requireProviderDefinition(provider).defaultModel;
}

/** Billing currency of the session's provider (declared in the provider config). */
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
  const modelId = options.model ?? def.defaultModel;
  const model = getPiModel(provider, modelId);
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
    // A declared per-model thinkingEfforts allowlist remaps unsupported
    // session values to the nearest accepted one (off → lowest on
    // mandatory-reasoning models); passthrough otherwise.
    thinkingEffort: mapModelThinkingEffort(
      options.thinkingEffort ?? 'off',
      def.staticModels.find((opt) => opt.value === modelId)?.thinkingEfforts,
    ),
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
 * currency: the model's declared cost, the discovered `/models` prices (cache
 * reads at `input_cache_read` when the gateway reports it), or the provider's
 * fallback.
 */
export async function getModelPricing(
  provider: ProviderId,
  model: ModelId,
): Promise<GenericPricing> {
  const def = requireProviderDefinition(provider);
  await ensureProviderRefreshed(provider);
  const piModel = getPiModel(provider, model);
  return {
    currency: def.currency,
    cacheHitPerM: piModel.cost.cacheRead,
    cacheMissPerM: piModel.cost.input,
    outputPerM: piModel.cost.output,
  };
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
 * Declared modalities are definitive; a discovery provider whose catalog has
 * not been fetched, or whose slug is not in any table, returns `null` so
 * callers retry later instead of caching a wrong "text-only" answer for the
 * whole session (a memoized negative would permanently hide read_media from
 * the model).
 */
export async function getModelModalities(
  provider: ProviderId,
  model: ModelId,
): Promise<{ image: boolean; audio: boolean } | null> {
  const def = requireProviderDefinition(provider);
  // Declared static models carry their modalities (text-only when omitted).
  const declared = def.staticModels.find((opt) => opt.value === model);
  if (declared) {
    return {
      image: declared.modalities?.includes('image') ?? false,
      audio: declared.modalities?.includes('audio') ?? false,
    };
  }
  if (!def.discovery.enabled) {
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
