import { describe, expect, it } from 'vitest';
import {
  appendCacheDiagnostic,
  buildCacheDiagnostic,
  inferCacheMissReason,
  latestCacheDiagnostic,
  prefixDiagnosticHashes,
  stableHash,
  type CacheDiagnosticEntry,
} from '../cache-diagnostics.js';

const TOOL = { type: 'function', function: { name: 'bash', description: 'x', parameters: {} } };
const ENV = { role: 'user', name: 'Environment', content: 'env snapshot' };

function prefix(
  overrides: {
    system?: string;
    tools?: readonly unknown[];
    env?: unknown;
  } = {},
) {
  return prefixDiagnosticHashes({
    system: overrides.system ?? 'sys',
    toolSpecs: overrides.tools ?? [TOOL],
    env: overrides.env ?? ENV,
  });
}

function entry(
  overrides: Partial<Parameters<typeof buildCacheDiagnostic>[0]> = {},
): CacheDiagnosticEntry {
  return buildCacheDiagnostic({
    turn: 1,
    model: 'deepseek-v4-flash',
    usage: { inputTokens: 10_000, cachedTokens: 9_000, missTokens: 1_000 },
    pricing: { cacheHitPerM: 0.1, cacheMissPerM: 3.0 },
    prefix: prefix(),
    previous: null,
    now: 1_000,
    ...overrides,
  });
}

describe('stableHash', () => {
  it('is stable for the same input and distinct for different inputs', () => {
    expect(stableHash({ a: 1 })).toBe(stableHash({ a: 1 }));
    expect(stableHash({ a: 1 })).not.toBe(stableHash({ a: 2 }));
  });

  it('does not throw on undefined input', () => {
    expect(stableHash(undefined)).toBe(stableHash(undefined));
  });
});

describe('prefixDiagnosticHashes', () => {
  it('extracts tool names and hashes the prefix components', () => {
    const hashes = prefix();
    expect(hashes.toolNames).toEqual(['bash']);
    expect(hashes.toolCount).toBe(1);
    expect(hashes.systemHash).toBe(stableHash('sys'));
    expect(hashes.envHash).toBe(stableHash(ENV));
  });
});

describe('inferCacheMissReason', () => {
  it('reports no-miss when nothing missed', () => {
    const { reason } = inferCacheMissReason(entry(), prefix(), 0);
    expect(reason).toBe('no-miss');
  });

  it('reports cold-start without previous evidence', () => {
    const { reason } = inferCacheMissReason(null, prefix(), 1_000);
    expect(reason).toBe('cold-start');
  });

  it('detects a system prompt change', () => {
    const { reason, detail } = inferCacheMissReason(entry(), prefix({ system: 'other' }), 1_000);
    expect(reason).toBe('system-prompt-changed');
    expect(detail).toContain('systemHash');
  });

  it('detects an environment snapshot change', () => {
    const { reason } = inferCacheMissReason(
      entry(),
      prefix({ env: { role: 'user', name: 'Environment', content: 'different' } }),
      1_000,
    );
    expect(reason).toBe('env-snapshot-changed');
  });

  it('detects a tool list change', () => {
    const extra = { type: 'function', function: { name: 'grep', description: 'y' } };
    const { reason, detail } = inferCacheMissReason(
      entry(),
      prefix({ tools: [TOOL, extra] }),
      1_000,
    );
    expect(reason).toBe('tool-list-changed');
    expect(detail).toContain('added grep');
  });

  it('detects a tool schema change with the same name', () => {
    const changed = { type: 'function', function: { name: 'bash', description: 'y' } };
    const { reason } = inferCacheMissReason(entry(), prefix({ tools: [changed] }), 1_000);
    expect(reason).toBe('tool-schema-or-order-changed');
  });

  it('reports unknown when the prefix changed but sub-hashes matched', () => {
    const current = { ...prefix(), prefixHash: 'deadbeefdeadbeef' };
    const { reason, detail } = inferCacheMissReason(entry(), current, 1_000);
    expect(reason).toBe('unknown');
    expect(detail).toContain('prefixHash changed');
  });

  it('reports provider-side when everything matched', () => {
    const { reason, detail } = inferCacheMissReason(entry(), prefix(), 1_000);
    expect(reason).toBe('unknown');
    expect(detail).toContain('provider-side');
  });
});

describe('buildCacheDiagnostic', () => {
  it('computes hit rate and savings in the billing currency', () => {
    const built = entry();
    expect(built.cacheHitRate).toBe(0.9);
    // 9000 tokens at (3.0 - 0.1) CNY per 1M
    expect(built.savedCost).toBeCloseTo((9_000 / 1_000_000) * 2.9);
    expect(built.missReason).toBe('cold-start'); // no previous evidence
  });

  it('uses the previous entry when inferring the miss reason', () => {
    const previous = entry();
    const built = entry({
      prefix: prefix({ system: 'changed' }),
      previous,
    });
    expect(built.missReason).toBe('system-prompt-changed');
  });
});

describe('appendCacheDiagnostic / latestCacheDiagnostic', () => {
  it('ring-buffers at the limit', () => {
    const a = entry({ turn: 1 });
    const b = entry({ turn: 2 });
    const c = entry({ turn: 3 });
    const out = appendCacheDiagnostic([a, b], c, 2);
    expect(out).toEqual([b, c]);
  });

  it('filters invalid entries on append and skips them when finding the latest', () => {
    const a = entry({ turn: 1 });
    const b = entry({ turn: 2 });
    // Simulate corrupt persisted data.
    const corrupt = [{ garbage: true }, a, { turn: 'x' }] as unknown as CacheDiagnosticEntry[];
    const out = appendCacheDiagnostic(corrupt, b);
    expect(out).toEqual([a, b]);
    expect(
      latestCacheDiagnostic([{ garbage: true }, a] as unknown as CacheDiagnosticEntry[]),
    ).toEqual(a);
    expect(
      latestCacheDiagnostic([{ garbage: true }] as unknown as CacheDiagnosticEntry[]),
    ).toBeNull();
    expect(latestCacheDiagnostic(undefined)).toBeNull();
  });
});
