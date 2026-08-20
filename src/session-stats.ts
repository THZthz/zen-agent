import { readFile } from "node:fs/promises";
import type { ModelMessage } from "ai";
import {
  BASH_TOOL_SCHEMA,
  costYuan,
  getModelPricing,
  type LlmToolCall,
  type LlmUsage,
} from "./llm/deepseek.js";
import {
  emptySessionUsage,
  sessionLlmLogPath,
  type ModelId,
  type SessionUsage,
} from "./storage.js";
import { countTokensBounded, estimateRequestTokens } from "./tokenizer.js";
import {
  cacheHitPercent,
  emptyTurnStats,
  formatMs,
  formatTokens,
  formatYuan,
  type TurnStats,
} from "./turn-stats.js";

/**
 * Recovery of usage statistics for sessions whose files predate usage
 * tracking (no `usage` field) or whose `usage` was never merged (crash
 * mid-turn). Rebuilds per-turn stats from the persistent history:
 *
 *  - turns/steps: user-message boundaries in `llmMessages` / the llm JSONL
 *  - input/output tokens: DeepSeek V4 tokenizer over the exact request
 *    payloads (mean error ~3.6% vs API `prompt_tokens`, measured)
 *  - llm/tool wall time: request/response timestamps in the llm JSONL
 *    (matches recorded `llmMs` exactly)
 *  - cost: current rate card applied to the recovered/estimated tokens
 *
 * Fields that cannot be recovered (cache split, reasoning tokens,
 * thinking/answering split for legacy logs) stay 0, so the cache hit ratio
 * shows "n/a" instead of a fabricated number.
 */

export interface RebuiltSessionStats {
  turnStats: TurnStats[];
  usage: SessionUsage;
  /** Input tokens of the last LLM step — what `usage_update.used` should show. */
  lastContextUsed: number;
  /** True when any number is an estimate (legacy logs have no `usage`). */
  estimated: boolean;
}

/** Wire-format message as the tokenizer's chat template expects it. */
interface WireMessage {
  role: string;
  content?: string | null;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

interface LogStep {
  /** 1-based user-turn number (count of user messages in the request). */
  turn: number;
  requestTs: number;
  responseTs: number;
  llmMs: number;
  toolMs: number;
  text: string;
  toolCalls: LlmToolCall[];
  usage: LlmUsage | null;
  requestMessages: ModelMessage[];
  system: string;
}

const MAX_CACHEABLE_CHARS = 10 * 1024;

function boundedTokenCache(): Map<string, number> {
  return new Map<string, number>();
}

function cachedBoundedTokens(s: string, cache: Map<string, number>): number {
  if (s.length === 0) {
    return 0;
  }
  const cached = cache.get(s);
  if (cached !== undefined) {
    return cached;
  }
  const n = countTokensBounded(s);
  if (s.length <= MAX_CACHEABLE_CHARS) {
    cache.set(s, n);
  }
  return n;
}

/** Convert stored AI-SDK messages to the OpenAI wire format (same as runLlmStep). */
function toWireMessages(messages: ModelMessage[]): WireMessage[] {
  const out: WireMessage[] = [];
  for (const message of messages) {
    switch (message.role) {
      case "user": {
        const content =
          typeof message.content === "string"
            ? message.content
            : message.content
                .filter((part) => part.type === "text")
                .map((part) => (part as { text: string }).text)
                .join("");
        out.push({ role: "user", content });
        break;
      }
      case "assistant": {
        const parts = Array.isArray(message.content) ? message.content : [];
        const text = parts
          .filter((part) => part.type === "text")
          .map((part) => (part as { text: string }).text)
          .join("");
        const toolCalls = parts
          .filter((part) => part.type === "tool-call")
          .map((part) => ({
            id: (part as { toolCallId: string }).toolCallId,
            type: "function",
            function: {
              name: (part as { toolName: string }).toolName,
              arguments: JSON.stringify((part as { input: unknown }).input),
            },
          }));
        const assistantMessage: WireMessage = {
          role: "assistant",
          content: text || null,
        };
        if (toolCalls.length > 0) {
          assistantMessage.tool_calls = toolCalls;
        }
        out.push(assistantMessage);
        break;
      }
      case "tool": {
        const parts = Array.isArray(message.content) ? message.content : [];
        for (const part of parts) {
          if (part.type !== "tool-result") {
            continue;
          }
          const output = (part as { output: unknown }).output;
          const text =
            typeof output === "string"
              ? output
              : typeof output === "object" &&
                  output !== null &&
                  "value" in (output as Record<string, unknown>) &&
                  typeof (output as Record<string, unknown>).value === "string"
                ? ((output as Record<string, unknown>).value as string)
                : JSON.stringify(output);
          out.push({ role: "tool", tool_call_id: (part as { toolCallId: string }).toolCallId, content: text });
        }
        break;
      }
      case "system": {
        out.push({
          role: "system",
          content:
            typeof message.content === "string" ? message.content : JSON.stringify(message.content),
        });
        break;
      }
    }
  }
  return out;
}

/** Per-message template + content tokens, mirroring the tokenizer's chat template. */
function messageTokens(m: WireMessage, cache: Map<string, number>): number {
  let n = 6;
  if (typeof m.content === "string" && m.content.length > 0) {
    n += cachedBoundedTokens(m.content, cache);
  }
  if (m.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
    n += cachedBoundedTokens(JSON.stringify(m.tool_calls), cache);
  }
  return n;
}

function isUsageEmpty(usage: SessionUsage): boolean {
  return (
    usage.turns === 0 &&
    usage.steps === 0 &&
    usage.inputTokens === 0 &&
    usage.outputTokens === 0 &&
    usage.cacheReadTokens === 0 &&
    usage.cacheMissTokens === 0 &&
    usage.reasoningTokens === 0 &&
    usage.costYuan === 0 &&
    usage.llmMs === 0 &&
    usage.thinkingMs === 0 &&
    usage.answeringMs === 0 &&
    usage.toolMs === 0
  );
}

export function needsSessionStatsRecovery(session: {
  usage: SessionUsage;
  llmMessages: ModelMessage[];
}): boolean {
  return isUsageEmpty(session.usage) && session.llmMessages.length > 0;
}

function rollup(turnStats: TurnStats[]): SessionUsage {
  const usage = emptySessionUsage();
  for (const turn of turnStats) {
    usage.steps += turn.steps;
    usage.inputTokens += turn.inputTokens;
    usage.outputTokens += turn.outputTokens;
    usage.cacheReadTokens += turn.cacheReadTokens;
    usage.cacheMissTokens += turn.cacheMissTokens;
    usage.reasoningTokens += turn.reasoningTokens;
    usage.costYuan += turn.costYuan;
    usage.llmMs += turn.llmMs;
    usage.thinkingMs += turn.thinkingMs;
    usage.answeringMs += turn.answeringMs;
    usage.toolMs += turn.toolMs;
  }
  usage.turns = turnStats.length;
  return usage;
}

/** Parse the per-session llm JSONL into sequential LLM steps (request+response pairs). */
async function parseLlmLog(
  cwd: string,
  sessionId: string,
): Promise<LogStep[] | null> {
  let raw: string;
  try {
    raw = await readFile(sessionLlmLogPath(cwd, sessionId), "utf8");
  } catch {
    return null;
  }

  const steps: LogStep[] = [];
  let pending: {
    ts: number;
    messages: ModelMessage[];
    system: string;
  } | null = null;

  for (const line of raw.split("\n")) {
    if (!line.trim()) {
      continue;
    }
    let entry: Record<string, unknown>;
    try {
      entry = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }

    if (entry.type === "llm_request") {
      const requestTs = Date.parse(String(entry.timestamp ?? ""));
      if (steps.length > 0) {
        // The gap between a response that produced tool calls and this next
        // request is (almost) exactly the bash tool execution time.
        const last = steps[steps.length - 1]!;
        if (last.toolCalls.length > 0 && last.toolMs === 0) {
          last.toolMs = requestTs - last.responseTs;
        }
      }
      pending = {
        ts: requestTs,
        messages: (entry.messages ?? []) as ModelMessage[],
        system: String(entry.system ?? ""),
      };
      continue;
    }

    if (entry.type === "llm_response" && pending) {
      const responseTs = Date.parse(String(entry.timestamp ?? ""));
      const turn = (pending.messages ?? []).filter(
        (m) => m.role === "user",
      ).length;
      steps.push({
        turn,
        requestTs: pending.ts,
        responseTs,
        llmMs: responseTs - pending.ts,
        toolMs: 0,
        text: String(entry.text ?? ""),
        toolCalls: (entry.toolCalls ?? []) as LlmToolCall[],
        usage: (entry.usage ?? null) as LlmUsage | null,
        requestMessages: pending.messages,
        system: pending.system,
      });
      pending = null;
      continue;
    }
  }

  return steps.length > 0 ? steps : null;
}

/** Rebuild from the llm JSONL: exact usage when logged, tokenizer estimates otherwise. */
function rebuildFromLog(
  steps: LogStep[],
  model: ModelId,
  cache: Map<string, number>,
): RebuiltSessionStats {
  const turnMap = new Map<number, TurnStats>();
  let anyEstimated = false;
  let lastContextUsed = 0;

  for (const step of steps) {
    let turn = turnMap.get(step.turn);
    if (!turn) {
      turn = emptyTurnStats();
      turnMap.set(step.turn, turn);
    }
    turn.steps += 1;
    const pricing = getModelPricing(model, new Date(step.responseTs || step.requestTs));

    if (step.usage) {
      turn.inputTokens += step.usage.inputTokens;
      turn.outputTokens += step.usage.outputTokens;
      turn.cacheReadTokens += step.usage.cacheReadTokens;
      turn.cacheMissTokens += step.usage.cacheMissTokens;
      turn.reasoningTokens += step.usage.reasoningTokens;
      turn.llmMs += step.usage.llmMs;
      turn.thinkingMs += step.usage.thinkingMs;
      turn.answeringMs += step.usage.answeringMs;
      turn.toolMs += Math.max(0, step.toolMs);
      turn.costYuan += costYuan(step.usage, pricing);
      lastContextUsed = step.usage.inputTokens;
    } else {
      anyEstimated = true;
      const wire = toWireMessages(step.requestMessages);
      const inputTokens =
        estimateRequestTokens(wire, [BASH_TOOL_SCHEMA]) +
        cachedBoundedTokens(step.system, cache);
      const outputTokens =
        cachedBoundedTokens(step.text, cache) +
        cachedBoundedTokens(JSON.stringify(step.toolCalls), cache);
      turn.inputTokens += inputTokens;
      turn.outputTokens += outputTokens;
      turn.llmMs += Math.max(0, step.llmMs);
      turn.toolMs += Math.max(0, step.toolMs);
      // Cache split is unknown for legacy logs; charge input at the miss rate
      // (what real sessions showed) but keep cacheRead/Miss at 0 so the
      // displayed hit ratio stays "n/a" instead of a fake 0%.
      turn.costYuan +=
        (inputTokens / 1_000_000) * pricing.cacheMissCnyPerM +
        (outputTokens / 1_000_000) * pricing.outputCnyPerM;
      lastContextUsed = inputTokens;
    }
  }

  const turnStats = [...turnMap.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, turn]) => turn);
  const usage = rollup(turnStats);
  usage.estimated = anyEstimated;
  return { turnStats, usage, lastContextUsed, estimated: anyEstimated };
}

/**
 * Fallback when the llm JSONL is gone: estimate turns/steps/tokens from
 * `llmMessages` alone (times cannot be recovered, so they stay 0).
 */
function rebuildFromMessages(
  messages: ModelMessage[],
  system: string,
  model: ModelId,
  cache: Map<string, number>,
): RebuiltSessionStats {
  const wire = toWireMessages(messages);
  const turnStats: TurnStats[] = [];
  let current = emptyTurnStats();
  // BOS-ish constant (2) + system prompt + tool schemas, mirroring
  // estimateRequestTokens + the separate system message DeepSeek receives.
  let prefixTokens =
    2 +
    cachedBoundedTokens(system, cache) +
    estimateRequestTokens([], [BASH_TOOL_SCHEMA]);
  let lastContextUsed = 0;

  for (const m of wire) {
    const tokens = messageTokens(m, cache);
    if (m.role === "assistant") {
      current.steps += 1;
      current.inputTokens += prefixTokens;
      current.outputTokens += tokens;
      lastContextUsed = prefixTokens;
    } else if (m.role === "user" && current.steps > 0) {
      turnStats.push(current);
      current = emptyTurnStats();
    }
    prefixTokens += tokens;
  }
  if (current.steps > 0) {
    turnStats.push(current);
  }

  const usage = rollup(turnStats);
  usage.estimated = true;
  // No timing data: cost at current rates from estimated tokens.
  const pricing = getModelPricing(model);
  for (const turn of turnStats) {
    turn.costYuan =
      (turn.inputTokens / 1_000_000) * pricing.cacheMissCnyPerM +
      (turn.outputTokens / 1_000_000) * pricing.outputCnyPerM;
    usage.costYuan += turn.costYuan;
  }
  return { turnStats, usage, lastContextUsed, estimated: true };
}

/**
 * Rebuild session stats from persistent history. Returns null when there is
 * nothing to rebuild from.
 */
export async function rebuildSessionStats(
  cwd: string,
  sessionId: string,
  messages: ModelMessage[],
  system: string,
  model: ModelId,
): Promise<RebuiltSessionStats | null> {
  if (messages.length === 0) {
    return null;
  }
  const cache = boundedTokenCache();
  const steps = await parseLlmLog(cwd, sessionId);
  if (steps && steps.length > 0) {
    return rebuildFromLog(steps, model, cache);
  }
  return rebuildFromMessages(messages, system, model, cache);
}

/** One-line summary shown after resume when stats were recovered. */
export function formatRecoveredSummary(stats: RebuiltSessionStats): string {
  const { usage, estimated } = stats;
  const turnPlural = usage.turns === 1 ? "turn" : "turns";
  const stepPlural = usage.steps === 1 ? "step" : "steps";
  const parts = [
    `${estimated ? "Recovered session stats (estimated)" : "Session stats"}: ${usage.turns} ${turnPlural} · ${usage.steps} ${stepPlural}`,
  ];
  if (usage.llmMs > 0) {
    parts.push(`llm ${formatMs(usage.llmMs)}`);
  }
  if (usage.toolMs > 0) {
    parts.push(`tools ${formatMs(usage.toolMs)}`);
  }
  return `${parts.join(" · ")}\n` +
    `in ${formatTokens(usage.inputTokens)} · out ${formatTokens(usage.outputTokens)} · cache ${cacheHitPercent(usage)} · ¥${formatYuan(usage.costYuan)}`;
}
