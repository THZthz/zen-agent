import { createOpenAI } from "@ai-sdk/openai";
import { isStepCount, streamText, tool } from "ai";
import { z } from "zod";
import type { LanguageModelUsage, ModelMessage } from "ai";
import type { ModelId, ThinkingEffort } from "../storage.js";

export interface LlmToolCall {
  id: string;
  name: string;
  input: unknown;
}

export interface LlmUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cacheReadTokens: number;
  cacheMissTokens: number;
  reasoningTokens: number;
  /** Wall-clock time of the whole LLM request in ms. */
  llmMs: number;
  /** Time from request start until the first answer token arrived (thinking time) in ms. */
  thinkingMs: number;
  /** Time spent streaming answer text in ms. */
  answeringMs: number;
}

export interface LlmStepResult {
  text: string;
  toolCalls: LlmToolCall[];
  finishReason: string;
  /** Token usage and timing for this LLM step, if reported by the provider. */
  usage: LlmUsage | null;
}

export const SYSTEM_PROMPT = `You are an experiened software engineer.

You have exactly one tool: bash.
You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.
There is no approval gate: every command you run is executed immediately.
Prefer small, targeted bash commands. Avoid large output from using bash tool.
When modifying files, use shell tools such as cat, sed, awk, or tee. Prefer using trash (npm install --global trash-cli), rg, fdfind (fd), jq, uv if they exist.

> Always use utf-8, no emojis unless needed by your task.`;

export const bashTool = tool({
  description: "Execute a bash command in current OS. The command is completely unrestricted. Your command will be wrapped inside `script -q -e -c \"bash <script file containing your command>\" \"<log path>\"`. If output is large, this tool will tell you to check the log file instead of showing all.",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute."),
  }),
});

export interface ModelPricing {
  /** CNY per 1M input tokens served from cache. */
  cacheHitCnyPerM: number;
  /** CNY per 1M input tokens not served from cache. */
  cacheMissCnyPerM: number;
  /** CNY per 1M output tokens. */
  outputCnyPerM: number;
}

interface ModelRateTable {
  /** CNY per 1M input tokens served from cache. */
  cacheHit: { peak: number; offPeak: number };
  /** CNY per 1M input tokens not served from cache. */
  cacheMiss: { peak: number; offPeak: number };
  /** CNY per 1M output tokens. */
  output: { peak: number; offPeak: number };
}

/**
 * Official DeepSeek pricing (CNY per 1M tokens) for the V4 models.
 * Off-peak price is half the peak price. Peak hours are Beijing time
 * 09:00-12:00 and 14:00-18:00; all other hours are off-peak.
 * Values can be overridden with DEEPSEEK_PRICE_* environment variables.
 */
const MODEL_RATE_TABLE: Record<ModelId, ModelRateTable> = {
  "deepseek-v4-flash": {
    cacheHit: { peak: 0.1, offPeak: 0.05 },
    cacheMiss: { peak: 3.0, offPeak: 1.5 },
    output: { peak: 9.0, offPeak: 4.5 },
  },
  "deepseek-v4-pro": {
    cacheHit: { peak: 0.3, offPeak: 0.15 },
    cacheMiss: { peak: 9.0, offPeak: 4.5 },
    output: { peak: 27.0, offPeak: 13.5 },
  },
};

function parseEnvNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

/**
 * Whether `now` (UTC) falls in DeepSeek's peak pricing window.
 * Peak hours are Beijing time (UTC+8) 09:00-12:00 and 14:00-18:00;
 * 12:00 and 18:00 themselves are off-peak.
 */
export function isPeakTime(now: Date = new Date()): boolean {
  // Beijing time is always UTC+8 (no DST); read the shifted UTC clock directly.
  const beijing = new Date(now.getTime() + 8 * 3_600_000);
  const minutes = beijing.getUTCHours() * 60 + beijing.getUTCMinutes();
  return (minutes >= 9 * 60 && minutes < 12 * 60) || (minutes >= 14 * 60 && minutes < 18 * 60);
}

/** Effective pricing for a model at `now`, including peak/off-peak selection. */
export function getModelPricing(model: ModelId, now: Date = new Date()): ModelPricing {
  const base = MODEL_RATE_TABLE[model] ?? MODEL_RATE_TABLE["deepseek-v4-flash"];
  const peak = isPeakTime(now);
  return {
    cacheHitCnyPerM: parseEnvNumber(
      "DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK",
      peak ? base.cacheHit.peak : base.cacheHit.offPeak,
    ),
    cacheMissCnyPerM: parseEnvNumber(
      "DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK",
      peak ? base.cacheMiss.peak : base.cacheMiss.offPeak,
    ),
    outputCnyPerM: parseEnvNumber(
      "DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK",
      peak ? base.output.peak : base.output.offPeak,
    ),
  };
}

/** Session context window size in tokens, used for usage_update.size. */
export function getContextWindowTokens(): number {
  const raw = process.env.DEEPSEEK_CONTEXT_WINDOW;
  if (!raw) {
    return 1_000_000;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1_000_000;
}

/** Cost in CNY for a single LLM step's token usage. */
export function costYuan(usage: LlmUsage, pricing: ModelPricing): number {
  return (
    (usage.cacheReadTokens / 1_000_000) * pricing.cacheHitCnyPerM +
    (usage.cacheMissTokens / 1_000_000) * pricing.cacheMissCnyPerM +
    (usage.outputTokens / 1_000_000) * pricing.outputCnyPerM
  );
}

export function createDeepseekModel(model?: ModelId | string) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is required");
  }

  const baseURL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";
  const modelName = model ?? process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";

  const provider = createOpenAI({
    name: "deepseek",
    apiKey,
    baseURL,
  });

  return provider.chat(modelName);
}

interface RawUsage {
  prompt_cache_hit_tokens?: unknown;
  prompt_cache_miss_tokens?: unknown;
  completion_tokens_details?: { reasoning_tokens?: unknown };
}

function toNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export /**
 * Normalizes the AI SDK's LanguageModelUsage into our LlmUsage.
 *
 * DeepSeek's OpenAI-compatible API reports cache tokens in its own fields
 * (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` in `usage.raw`),
 * which `@ai-sdk/openai` does NOT map into `inputTokenDetails` — it only
 * reads OpenAI-style `prompt_tokens_details.cached_tokens`. So we read the
 * raw fields first and fall back to the SDK's mapping when they are absent.
 * Reasoning tokens likewise come from `completion_tokens_details.reasoning_tokens`
 * (DeepSeek's `reasoning_content` mode) or the SDK's `outputTokenDetails`.
 */
function extractUsage(
  usage: LanguageModelUsage | undefined,
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (!usage) {
    return null;
  }
  const raw = usage.raw as RawUsage | undefined;

  const inputTokens = usage.inputTokens ?? 0;
  const outputTokens = usage.outputTokens ?? 0;
  const rawCacheHit = raw?.prompt_cache_hit_tokens;
  const cacheReadTokens =
    typeof rawCacheHit === "number"
      ? rawCacheHit
      : usage.inputTokenDetails?.cacheReadTokens ?? 0;
  const rawCacheMiss = raw?.prompt_cache_miss_tokens;
  const cacheMissTokens =
    typeof rawCacheMiss === "number"
      ? rawCacheMiss
      : Math.max(0, inputTokens - cacheReadTokens);
  const rawReasoning = raw?.completion_tokens_details?.reasoning_tokens;
  const reasoningTokens =
    usage.outputTokenDetails?.reasoningTokens ??
    (typeof rawReasoning === "number" ? rawReasoning : 0);

  if (inputTokens === 0 && outputTokens === 0 && cacheReadTokens === 0 && cacheMissTokens === 0) {
    return null;
  }

  return {
    inputTokens,
    outputTokens,
    totalTokens: usage.totalTokens ?? inputTokens + outputTokens,
    cacheReadTokens,
    cacheMissTokens,
    reasoningTokens,
    llmMs: timing.llmMs,
    thinkingMs: timing.thinkingMs,
    answeringMs: timing.answeringMs,
  };
}

export async function runLlmStep(options: {
  messages: ModelMessage[];
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  model?: ModelId;
  thinkingEffort?: ThinkingEffort;
  system?: string;
}): Promise<LlmStepResult> {
  const model = createDeepseekModel(options.model);

  // Timing: "thinking" is the wall time from request start until the first
  // answer (non-reasoning) token arrives — i.e. TTFB, dominated by the
  // model's reasoning phase when thinking is enabled. "answering" is the
  // remaining stream time. If the step only produced reasoning (no text,
  // e.g. a tool-call-only step), firstTextAt stays null and thinking covers
  // the whole request.
  const requestStart = Date.now();
  let firstTextAt: number | null = null;
  let lastReasoningAt: number | null = null;

  let text = "";
  let finishReason = "unknown";
  let stepUsage: LanguageModelUsage | undefined;
  const toolCalls: LlmToolCall[] = [];

  const providerOptions =
    options.thinkingEffort && options.thinkingEffort !== "off"
      ? {
          openai: {
            reasoningEffort: options.thinkingEffort,
          },
        }
      : undefined;

  const result = streamText({
    model,
    system: options.system ?? SYSTEM_PROMPT,
    messages: options.messages,
    tools: {
      bash: bashTool,
    },
    stopWhen: isStepCount(1),
    abortSignal: options.signal,
    providerOptions,
    includeRawChunks: true,
  });

  for await (const part of result.fullStream) {
    switch (part.type) {
      case "text-delta": {
        if (firstTextAt === null) {
          firstTextAt = Date.now();
        }
        text += part.text;
        await options.onTextDelta?.(part.text);
        break;
      }
      case "reasoning-delta": {
        lastReasoningAt = Date.now();
        await options.onReasoningDelta?.(part.text);
        break;
      }
      case "tool-call": {
        toolCalls.push({
          id: part.toolCallId,
          name: part.toolName,
          input: part.input,
        });
        break;
      }
      case "finish-step": {
        finishReason = part.finishReason;
        stepUsage = part.usage;
        break;
      }
      case "finish": {
        if (!stepUsage) {
          stepUsage = part.totalUsage;
        }
        break;
      }
      case "raw": {
        const raw = part.rawValue as {
          choices?: Array<{
            delta?: {
              reasoning_content?: unknown;
            };
          }>;
        };
        const reasoningContent = raw.choices?.[0]?.delta?.reasoning_content;
        if (typeof reasoningContent === "string" && reasoningContent.length > 0) {
          lastReasoningAt = Date.now();
          await options.onReasoningDelta?.(reasoningContent);
        }
        break;
      }
      default:
        break;
    }

    // Yield to the event loop so throttled ACP output can be flushed
    // while the LLM stream is still being consumed.
    await new Promise<void>((resolve) => setImmediate(resolve));
  }

  const totalUsage = stepUsage ?? (await result.usage) ?? undefined;
  const llmMs = Date.now() - requestStart;
  const thinkingMs = (firstTextAt ?? lastReasoningAt ?? requestStart + llmMs) - requestStart;
  const answeringMs = firstTextAt !== null ? requestStart + llmMs - firstTextAt : 0;

  return {
    text,
    toolCalls,
    finishReason,
    usage: extractUsage(totalUsage, { llmMs, thinkingMs, answeringMs }),
  };
}
