import { createHash } from 'node:crypto';

/**
 * Per-turn cache diagnostics, ported from Reasonix's
 * telemetry/cache-diagnostics.ts. DeepSeek reports only hit/miss token counts;
 * the miss reason is inferred locally from hashes of the stable prefix
 * components (system prompt, tool schemas, frozen environment snapshot) that
 * must stay byte-identical for the context cache to keep hitting across steps.
 */

export const CACHE_DIAGNOSTICS_MAX_ENTRIES = 50;

export type CacheMissReason =
  | 'no-miss'
  | 'cold-start'
  | 'system-prompt-changed'
  | 'env-snapshot-changed'
  | 'tool-list-changed'
  | 'tool-schema-or-order-changed'
  | 'unknown';

export interface CacheDiagnosticEntry {
  ts: number;
  turn: number;
  model: string;
  inputTokens: number;
  cachedTokens: number;
  cacheMissTokens: number;
  cacheHitRate: number;
  /** Savings vs full cache-miss pricing, in the session's billing currency. */
  savedCost: number;
  prefixHash: string;
  systemHash: string;
  toolSpecsHash: string;
  envHash: string;
  toolCount: number;
  toolNames: string[];
  missReason: CacheMissReason;
  missReasonDetail: string;
}

export interface PrefixDiagnosticHashes {
  prefixHash: string;
  systemHash: string;
  toolSpecsHash: string;
  envHash: string;
  toolCount: number;
  toolNames: string[];
}

export interface CacheDiagnosticUsage {
  inputTokens: number;
  cachedTokens: number;
  missTokens: number;
}

export interface CacheDiagnosticInput {
  turn: number;
  model: string;
  usage: CacheDiagnosticUsage;
  /** Per-1M-token rates in the session's billing currency. */
  pricing: { cacheHitPerM: number; cacheMissPerM: number };
  prefix: PrefixDiagnosticHashes;
  previous?: CacheDiagnosticEntry | null;
  now?: number;
}

export function stableHash(value: unknown): string {
  const json = JSON.stringify(value);
  return createHash('sha256')
    .update(json ?? 'undefined')
    .digest('hex')
    .slice(0, 16);
}

export function prefixDiagnosticHashes(opts: {
  system: string;
  toolSpecs: readonly unknown[];
  env: unknown;
}): PrefixDiagnosticHashes {
  const toolNames = opts.toolSpecs
    .map((spec) => (spec as { function?: { name?: string } }).function?.name ?? '')
    .filter(Boolean);
  return {
    prefixHash: stableHash({ system: opts.system, tools: opts.toolSpecs, env: opts.env }),
    systemHash: stableHash(opts.system),
    toolSpecsHash: stableHash(opts.toolSpecs),
    envHash: stableHash(opts.env),
    toolCount: opts.toolSpecs.length,
    toolNames,
  };
}

export function buildCacheDiagnostic(input: CacheDiagnosticInput): CacheDiagnosticEntry {
  const { reason, detail } = inferCacheMissReason(
    input.previous ?? null,
    input.prefix,
    input.usage.missTokens,
  );
  return {
    ...input.prefix,
    ts: input.now ?? Date.now(),
    turn: input.turn,
    model: input.model,
    inputTokens: input.usage.inputTokens,
    cachedTokens: input.usage.cachedTokens,
    cacheMissTokens: input.usage.missTokens,
    cacheHitRate: cacheHitRate(input.usage),
    savedCost: cacheSavings(input.usage.cachedTokens, input.pricing),
    missReason: reason,
    missReasonDetail: detail,
  };
}

export function appendCacheDiagnostic(
  existing: readonly CacheDiagnosticEntry[] | undefined,
  entry: CacheDiagnosticEntry,
  limit = CACHE_DIAGNOSTICS_MAX_ENTRIES,
): CacheDiagnosticEntry[] {
  const safeExisting = Array.isArray(existing) ? existing.filter(isCacheDiagnosticEntry) : [];
  const next = [...safeExisting, entry];
  return next.slice(Math.max(0, next.length - limit));
}

export function latestCacheDiagnostic(
  entries: readonly CacheDiagnosticEntry[] | undefined,
): CacheDiagnosticEntry | null {
  if (!Array.isArray(entries)) return null;
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i];
    if (isCacheDiagnosticEntry(entry)) return entry;
  }
  return null;
}

export function inferCacheMissReason(
  previous: CacheDiagnosticEntry | null,
  current: PrefixDiagnosticHashes,
  missTokens: number,
): { reason: CacheMissReason; detail: string } {
  if (missTokens <= 0) {
    return { reason: 'no-miss', detail: 'No prompt-side cache miss tokens were reported.' };
  }
  if (!previous) {
    return { reason: 'cold-start', detail: 'No previous cache evidence exists for this session.' };
  }
  if (previous.systemHash !== current.systemHash) {
    return {
      reason: 'system-prompt-changed',
      detail: `systemHash ${short(previous.systemHash)} -> ${short(current.systemHash)}`,
    };
  }
  if (previous.envHash !== current.envHash) {
    return {
      reason: 'env-snapshot-changed',
      detail: `envHash ${short(previous.envHash)} -> ${short(current.envHash)}`,
    };
  }
  if (previous.toolSpecsHash !== current.toolSpecsHash) {
    const added = current.toolNames.filter((name) => !previous.toolNames.includes(name));
    const removed = previous.toolNames.filter((name) => !current.toolNames.includes(name));
    if (added.length > 0 || removed.length > 0 || previous.toolCount !== current.toolCount) {
      const parts: string[] = [];
      if (added.length > 0) parts.push(`added ${added.join(', ')}`);
      if (removed.length > 0) parts.push(`removed ${removed.join(', ')}`);
      if (parts.length === 0)
        parts.push(`tool count ${previous.toolCount} -> ${current.toolCount}`);
      return { reason: 'tool-list-changed', detail: parts.join('; ') };
    }
    return {
      reason: 'tool-schema-or-order-changed',
      detail: `toolSpecsHash ${short(previous.toolSpecsHash)} -> ${short(current.toolSpecsHash)}`,
    };
  }
  if (previous.prefixHash !== current.prefixHash) {
    return {
      reason: 'unknown',
      detail: `prefixHash changed (${short(previous.prefixHash)} -> ${short(current.prefixHash)}) but sub-hashes matched.`,
    };
  }
  return {
    reason: 'unknown',
    detail:
      'Prefix hashes matched. DeepSeek does not return cache-miss reasons, so this miss is likely provider-side (cache TTL/eviction) or outside the stable prefix.',
  };
}

export function isCacheDiagnosticEntry(value: unknown): value is CacheDiagnosticEntry {
  if (!value || typeof value !== 'object') return false;
  const entry = value as Partial<CacheDiagnosticEntry>;
  return (
    typeof entry.ts === 'number' &&
    typeof entry.turn === 'number' &&
    typeof entry.model === 'string' &&
    typeof entry.prefixHash === 'string' &&
    typeof entry.systemHash === 'string' &&
    typeof entry.toolSpecsHash === 'string' &&
    typeof entry.envHash === 'string' &&
    typeof entry.toolCount === 'number' &&
    Array.isArray(entry.toolNames) &&
    entry.toolNames.every((name) => typeof name === 'string') &&
    typeof entry.inputTokens === 'number' &&
    typeof entry.cachedTokens === 'number' &&
    typeof entry.cacheMissTokens === 'number' &&
    typeof entry.cacheHitRate === 'number' &&
    typeof entry.savedCost === 'number' &&
    typeof entry.missReason === 'string' &&
    typeof entry.missReasonDetail === 'string'
  );
}

function cacheHitRate(usage: CacheDiagnosticUsage): number {
  const total = usage.cachedTokens + usage.missTokens;
  return total > 0 ? usage.cachedTokens / total : 0;
}

function cacheSavings(
  cachedTokens: number,
  pricing: { cacheHitPerM: number; cacheMissPerM: number },
): number {
  return (cachedTokens / 1_000_000) * (pricing.cacheMissPerM - pricing.cacheHitPerM);
}

function short(hash: string): string {
  return hash.slice(0, 8);
}
