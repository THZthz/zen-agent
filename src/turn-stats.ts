export interface TurnStats {
  steps: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheMissTokens: number;
  reasoningTokens: number;
  /** Cost of the turn in the provider's billing currency (CNY or USD). */
  cost: number;
  llmMs: number;
  thinkingMs: number;
  answeringMs: number;
  toolMs: number;
}

export function emptyTurnStats(): TurnStats {
  return {
    steps: 0,
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheMissTokens: 0,
    reasoningTokens: 0,
    cost: 0,
    llmMs: 0,
    thinkingMs: 0,
    answeringMs: 0,
    toolMs: 0,
  };
}

export function formatMs(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatTokens(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}K`;
  }
  return String(count);
}

export function formatCost(amount: number): string {
  if (amount > 0 && amount < 0.01) {
    return amount.toFixed(4);
  }
  return amount.toFixed(3);
}

export function roundCost(amount: number): number {
  return Math.round(amount * 10_000) / 10_000;
}

export function cacheHitPercent(stats: {
  cacheReadTokens: number;
  cacheMissTokens: number;
}): string {
  const total = stats.cacheReadTokens + stats.cacheMissTokens;
  if (total === 0) {
    return 'n/a';
  }
  const percent = (stats.cacheReadTokens / total) * 100;
  return `${percent.toFixed(2)}%`;
}

export function showTurnStats(): boolean {
  const raw = process.env.ZEN_AGENT_SHOW_STATS;
  if (raw === undefined) {
    return true;
  }
  return !['0', 'false', 'no', 'off'].includes(raw.trim().toLowerCase());
}
