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
  runOpenRouterStep,
} from "./openrouter.js";
import {
  DEFAULT_MODEL,
  DEFAULT_PROVIDER,
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

/**
 * Active LLM provider, from ZEN_AGENT_LLM_PROVIDER (default "deepseek").
 * One provider applies per agent process (each Zed agent server is its own
 * process); sessions persist the provider they were created with so cost and
 * currency stay consistent across restarts.
 */
export function getProvider(): ProviderId {
  const raw = process.env.ZEN_AGENT_LLM_PROVIDER;
  if (raw === undefined || raw === "" || raw === DEFAULT_PROVIDER) {
    return DEFAULT_PROVIDER;
  }
  if (raw === "openrouter") {
    return "openrouter";
  }
  throw new Error(
    `Unknown ZEN_AGENT_LLM_PROVIDER: ${raw} (expected "deepseek" or "openrouter")`,
  );
}

/** Default model for a provider: DeepSeek's fallback or ZEN_AGENT_OPENROUTER_MODEL. */
export function getDefaultModel(provider: ProviderId): ModelId {
  if (provider === "openrouter") {
    return process.env.ZEN_AGENT_OPENROUTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;
  }
  return DEFAULT_MODEL;
}

/** Run one LLM step with the active provider. */
export async function runLlmStep(options: LlmStepOptions): Promise<LlmStepResult> {
  if (getProvider() === "openrouter") {
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
