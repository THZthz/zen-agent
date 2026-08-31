import type { BalanceSnapshot, ProviderDefinition } from './registry.js';
import { resolveApiKey } from './registry.js';

/**
 * Generic provider balance fetch driven by the provider definition
 * (`balance.path` + `balance.parse`). User-defined providers currently have
 * no balance endpoint, so this returns an unavailable snapshot — cost
 * verification is best-effort data gathering and must never fail the turn.
 */

/**
 * Fetch the current balance for a provider through its definition's balance
 * config. Providers without a balance endpoint return an unavailable snapshot
 * instead of failing.
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
