import type { BalanceSnapshot, ProviderDefinition } from './provider-registry.js';
import { resolveApiKey } from './provider-registry.js';

/**
 * Provider balance fetches: one generic HTTP path driven by the provider
 * definition (`balance.path` + `balance.parse`) plus the per-provider JSON
 * parsers (DeepSeek's `/user/balance` and OpenRouter's `/auth/key` have
 * completely different shapes).
 */

function finiteNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** DeepSeek `GET /user/balance` response parser (CNY balances are strings). */
export function parseDeepSeekBalance(json: unknown, label: string): BalanceSnapshot {
  const data = json as {
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
    throw new Error(`Unexpected ${label} balance API response: ${JSON.stringify(json)}`);
  }
  return {
    isAvailable: data.is_available ?? false,
    currency: info.currency ?? 'CNY',
    total: Number.parseFloat(info.total_balance),
    details: {
      grantedBalanceCny: Number.parseFloat(info.granted_balance ?? '0'),
      toppedUpBalanceCny: Number.parseFloat(info.topped_up_balance ?? '0'),
    },
  };
}

/** OpenRouter `GET /auth/key` response parser (USD credits). */
export function parseOpenRouterBalance(json: unknown, _label: string): BalanceSnapshot {
  const data = json as {
    data?: { label?: string; usage?: number; limit?: number; is_free_tier?: boolean };
  };
  const usageUsd = finiteNumber(data.data?.usage);
  const limitUsd = finiteNumber(data.data?.limit);
  return {
    isAvailable: true,
    currency: 'USD',
    total: Math.max(0, limitUsd - usageUsd),
    details: {
      usageUsd,
      limitUsd,
      isFreeTier: data.data?.is_free_tier ?? false,
    },
  };
}

/**
 * Fetch the current balance for a provider through its definition's balance
 * config. Providers without a balance endpoint return an unavailable snapshot
 * instead of failing (cost verification is best-effort data gathering).
 */
export async function fetchProviderBalance(
  def: ProviderDefinition,
  opts: { signal?: AbortSignal } = {},
): Promise<BalanceSnapshot> {
  if (!def.balance) {
    return { isAvailable: false, currency: def.currency, total: 0, details: {} };
  }
  const apiKey = resolveApiKey(def);
  const response = await fetch(`${def.baseUrl}${def.balance.path}`, {
    headers: { authorization: `Bearer ${apiKey}` },
    signal: opts.signal,
  });
  if (!response.ok) {
    throw new Error(
      `${def.label} balance API error ${response.status}: ${(await response.text().catch(() => '')).slice(0, 500)}`,
    );
  }
  const json: unknown = await response.json().catch(() => {
    throw new Error(`Unexpected ${def.label} balance API response: not JSON`);
  });
  return def.balance.parse(json, def.label);
}
