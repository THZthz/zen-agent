import {
  costFromUsage,
  type GenericPricing,
  type LlmStepOptions,
  type LlmStepResult,
  type LlmToolCall,
  type LlmUsage,
} from "./llm-client.js";
import {
  fetchDeepSeekBalance,
  getContextWindowTokens as getDeepSeekContextWindow,
  getModelPricing as getDeepSeekPricing,
  runLlmStep as runDeepSeekStep,
} from "./deepseek.js";
import {
  DEFAULT_OPENROUTER_MODEL,
  fetchOpenRouterBalance,
  getOpenRouterModelInfo,
  getOpenRouterModelModalities,
  runOpenRouterStep,
} from "./openrouter.js";
import {
  DEFAULT_DEEPSEEK_MODEL,
  type ModelId,
  type ProviderId,
} from "./storage.js";

export type { LlmStepResult, LlmStepOptions, LlmToolCall, LlmUsage } from "./llm-client.js";
export { costFromUsage } from "./llm-client.js";

/** Balance/credit snapshot for the active provider, in its billing currency. */
export interface BalanceSnapshot {
  isAvailable: boolean;
  currency: string;
  /** Remaining balance in `currency` units. */
  total: number;
  /** Provider-specific extras for the debug log. */
  details: Record<string, unknown>;
}

/** Default model for a provider: DeepSeek's fallback or OPENROUTER_MODEL. */
export function getDefaultModel(provider: ProviderId): ModelId {
  if (provider === "openrouter") {
    return process.env.OPENROUTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;
  }
  return DEFAULT_DEEPSEEK_MODEL;
}

/** Run one LLM step with the session's provider (per-session, not process-wide). */
export async function runLlmStep(
  provider: ProviderId,
  options: LlmStepOptions,
): Promise<LlmStepResult> {
  if (provider === "openrouter") {
    return runOpenRouterStep(options);
  }
  return runDeepSeekStep(options);
}

/**
 * Effective pricing for a model on a provider, in the provider's billing
 * currency (CNY for DeepSeek, USD for OpenRouter).
 */
export async function getModelPricing(
  provider: ProviderId,
  model: ModelId,
  now: Date = new Date(),
): Promise<GenericPricing> {
  if (provider === "openrouter") {
    // OpenRouter bills cached input at the same rate as regular input (no
    // separate cache price), so both input rates are the model's prompt price.
    const info = await getOpenRouterModelInfo(model);
    return {
      currency: "USD",
      cacheHitPerM: info.inputPerM,
      cacheMissPerM: info.inputPerM,
      outputPerM: info.outputPerM,
    };
  }
  const pricing = getDeepSeekPricing(model, now);
  return {
    currency: "CNY",
    cacheHitPerM: pricing.cacheHitCnyPerM,
    cacheMissPerM: pricing.cacheMissCnyPerM,
    outputPerM: pricing.outputCnyPerM,
  };
}

/** Session context window size in tokens, used for usage_update.size. */
export async function getContextWindowTokens(
  provider: ProviderId,
  model: ModelId,
): Promise<number> {
  if (provider === "openrouter") {
    return (await getOpenRouterModelInfo(model)).contextLength;
  }
  return getDeepSeekContextWindow();
}

/**
 * Input modalities accepted by the session's model. Conservative: unknown
 * catalog entries (offline, unknown slug) and DeepSeek (text-only models)
 * report neither image nor audio support.
 */
export async function getModelModalities(
  provider: ProviderId,
  model: ModelId,
): Promise<{ image: boolean; audio: boolean }> {
  if (provider === "openrouter") {
    const modalities = await getOpenRouterModelModalities(model);
    if (modalities === null) {
      return { image: false, audio: false };
    }
    return { image: modalities.includes("image"), audio: modalities.includes("audio") };
  }
  return { image: false, audio: false };
}

/** Balance/credit snapshot for the active provider. */
export async function fetchBalanceSnapshot(provider: ProviderId): Promise<BalanceSnapshot> {
  if (provider === "openrouter") {
    const balance = await fetchOpenRouterBalance();
    return {
      isAvailable: balance.isAvailable,
      currency: balance.currency,
      total: balance.remainingUsd,
      details: {
        usageUsd: balance.usageUsd,
        limitUsd: balance.limitUsd,
        isFreeTier: balance.isFreeTier,
      },
    };
  }
  const balance = await fetchDeepSeekBalance();
  return {
    isAvailable: balance.isAvailable,
    currency: balance.currency,
    total: balance.totalBalanceCny,
    details: {
      grantedBalanceCny: balance.grantedBalanceCny,
      toppedUpBalanceCny: balance.toppedUpBalanceCny,
    },
  };
}
