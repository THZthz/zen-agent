import type { ModelMessage } from "ai";
import type { ModelId, ThinkingEffort } from "../storage.js";
import { SYSTEM_PROMPT } from "../system-prompt.js";
export { SYSTEM_PROMPT } from "../system-prompt.js";

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

/**
 * The bash tool exposed to the model, in OpenAI function-calling wire format.
 * Kept as a plain object (no AI SDK `tool()` wrapper) because we talk to
 * DeepSeek's OpenAI-compatible API directly (see runLlmStep) — the description
 * and schema must stay byte-identical to what the AI SDK previously sent so
 * the model's behavior does not change.
 */
const BASH_TOOL_SCHEMA = {
  type: "function",
  function: {
    name: "bash",
    description:
      'Execute a bash command in current OS. The command is completely unrestricted. Your command will be wrapped inside `script -q -e -c "bash <script file containing your command>" "<log path>"`. If output is large, this tool will tell you to check the log file instead of showing all.',
    parameters: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The bash command to execute.",
        },
      },
      required: ["command"],
      additionalProperties: false,
    },
  },
} as const;

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

/**
 * DeepSeek's streaming usage object. We read the cache tokens from the raw
 * chunk usage: DeepSeek reports `prompt_cache_hit_tokens` /
 * `prompt_cache_miss_tokens` which OpenAI's schema (and the AI SDK's zod
 * parser) strips, so they must come straight from the wire.
 */
interface DeepSeekUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_cache_hit_tokens?: number;
  prompt_cache_miss_tokens?: number;
  completion_tokens_details?: { reasoning_tokens?: number };
}

function toNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function parseDeepSeekUsage(
  usage: DeepSeekUsage | undefined,
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (!usage) {
    return null;
  }
  const inputTokens = toNumber(usage.prompt_tokens);
  const outputTokens = toNumber(usage.completion_tokens);
  const cacheReadTokens = toNumber(usage.prompt_cache_hit_tokens);
  // DeepSeek reports prompt_cache_miss_tokens explicitly; when absent (some
  // proxies/models), fall back to input - cache-hit.
  const cacheMissTokens =
    usage.prompt_cache_miss_tokens !== undefined
      ? toNumber(usage.prompt_cache_miss_tokens)
      : Math.max(0, inputTokens - cacheReadTokens);
  const reasoningTokens = toNumber(usage.completion_tokens_details?.reasoning_tokens);

  if (inputTokens === 0 && outputTokens === 0 && cacheReadTokens === 0 && cacheMissTokens === 0) {
    return null;
  }

  return {
    inputTokens,
    outputTokens,
    totalTokens: toNumber(usage.total_tokens) || inputTokens + outputTokens,
    cacheReadTokens,
    cacheMissTokens,
    reasoningTokens,
    llmMs: timing.llmMs,
    thinkingMs: timing.thinkingMs,
    answeringMs: timing.answeringMs,
  };
}

/** A single parsed SSE chunk from the DeepSeek chat completions stream. */
interface DeepSeekChunk {
  choices?: Array<{
    index: number;
    delta?: {
      role?: string;
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: DeepSeekUsage;
  error?: { message?: string };
}

/** Partial tool call accumulated from streaming `delta.tool_calls` fragments. */
interface PartialToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** Convert our stored AI-SDK ModelMessage history to the OpenAI wire format. */
function toOpenAiMessages(messages: ModelMessage[]): unknown[] {
  const out: unknown[] = [];

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
        const assistantMessage: Record<string, unknown> = {
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
          out.push({
            role: "tool",
            tool_call_id: (part as { toolCallId: string }).toolCallId,
            content: text,
          });
        }
        break;
      }
      case "system": {
        out.push({
          role: "system",
          content: typeof message.content === "string" ? message.content : JSON.stringify(message.content),
        });
        break;
      }
    }
  }

  return out;
}

function mapFinishReason(raw: string | null | undefined): string {
  switch (raw) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
      return "tool-calls";
    case "content_filter":
      return "content-filter";
    case "insufficient_system_resource":
      return "error";
    default:
      return raw ? "other" : "unknown";
  }
}

/**
 * Calls DeepSeek's OpenAI-compatible chat completions API DIRECTLY and parses
 * the SSE stream ourselves.
 *
 * WHY NOT the AI SDK (`streamText` + `@ai-sdk/openai`): the provider's
 * `throwIfOpenAIStreamErrorBeforeOutput` reads ahead from the response until
 * it sees the first "output" chunk (non-empty `delta.content` / tool call).
 * DeepSeek's thinking mode only sends `delta.reasoning_content` during the
 * reasoning phase, which is invisible to that check, so the SDK swallows the
 * ENTIRE reasoning phase and hands the consumer a buffered burst once the
 * answer starts — the thinking block appears to not stream at all. By parsing
 * the SSE directly we get every reasoning delta live.
 *
 * This also lets us read DeepSeek's raw cache-token fields (see
 * parseDeepSeekUsage), which the SDK's zod schema strips.
 */
export async function runLlmStep(options: {
  messages: ModelMessage[];
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  model?: ModelId;
  thinkingEffort?: ThinkingEffort;
  system?: string;
}): Promise<LlmStepResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is required");
  }
  const baseURL = (process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com").replace(/\/+$/, "");
  const modelName = options.model ?? process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";

  const wireMessages = toOpenAiMessages(options.messages);
  const body: Record<string, unknown> = {
    model: modelName,
    messages: [
      { role: "system", content: options.system ?? SYSTEM_PROMPT },
      ...wireMessages.filter(
        (message) => (message as { role?: string }).role !== "system",
      ),
    ],
    tools: [BASH_TOOL_SCHEMA],
    stream: true,
  };
  if (options.thinkingEffort && options.thinkingEffort !== "off") {
    // Same wire field the AI SDK's providerOptions.openai.reasoningEffort mapped to.
    body.reasoning_effort = options.thinkingEffort;
  }

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
  let sawFinishReason = false;
  let sawOutput = false;
  let rawUsage: DeepSeekUsage | undefined;
  const toolCallsByIndex = new Map<number, PartialToolCall>();

  const fetchWithRetry = async (): Promise<Response> => {
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await fetch(`${baseURL}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify(body),
          signal: options.signal,
        });
        if (response.ok) {
          return response;
        }
        const errorBody = await response.text().catch(() => "");
        lastError = new Error(
          `DeepSeek API error ${response.status}: ${errorBody.slice(0, 500)}`,
        );
        if (
          response.status !== 429 &&
          response.status < 500 &&
          !(options.signal?.aborted ?? false)
        ) {
          throw lastError;
        }
      } catch (error) {
        if (options.signal?.aborted) {
          throw error;
        }
        lastError = error;
      }
      // Retryable (429/5xx/network): back off before the next attempt.
      await new Promise((resolve) => setTimeout(resolve, 500 * (attempt + 1)));
    }
    throw lastError ?? new Error("DeepSeek request failed");
  };

  const response = await fetchWithRetry();
  if (!response.body) {
    throw new Error("DeepSeek response has no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processEvent = async (rawEvent: string): Promise<boolean> => {
    // SSE spec: consecutive `data:` lines are joined with \n. Some servers
    // emit one data line per event; DeepSeek uses a single line.
    const dataLines: string[] = [];
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length === 0) {
      return false;
    }
    const data = dataLines.join("\n");
    if (data === "[DONE]") {
      return true;
    }

    let chunk: DeepSeekChunk;
    try {
      chunk = JSON.parse(data) as DeepSeekChunk;
    } catch {
      return false;
    }

    if (chunk.error) {
      throw new Error(`DeepSeek stream error: ${chunk.error.message ?? "unknown"}`);
    }
    if (chunk.usage) {
      rawUsage = chunk.usage;
    }

    const choice = chunk.choices?.[0];
    if (!choice) {
      return false;
    }

    if (choice.finish_reason != null) {
      finishReason = mapFinishReason(choice.finish_reason);
      sawFinishReason = true;
    }

    const delta = choice.delta;
    if (!delta) {
      return false;
    }

    // Reasoning content (DeepSeek thinking mode) — forwarded LIVE, which is
    // the whole point of bypassing the AI SDK here.
    if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0) {
      lastReasoningAt = Date.now();
      sawOutput = true;
      await options.onReasoningDelta?.(delta.reasoning_content);
    }

    // Answer text.
    if (typeof delta.content === "string" && delta.content.length > 0) {
      if (firstTextAt === null) {
        firstTextAt = Date.now();
      }
      text += delta.content;
      sawOutput = true;
      await options.onTextDelta?.(delta.content);
    }

    // Streaming tool calls (accumulate fragments per index).
    if (delta.tool_calls) {
      for (const toolCallDelta of delta.tool_calls) {
        const index = toolCallDelta.index ?? 0;
        const partial = toolCallsByIndex.get(index) ?? {
          id: "",
          name: "",
          arguments: "",
        };
        if (toolCallDelta.id) {
          partial.id = toolCallDelta.id;
        }
        if (toolCallDelta.function?.name) {
          partial.name = toolCallDelta.function.name;
        }
        if (toolCallDelta.function?.arguments) {
          partial.arguments += toolCallDelta.function.arguments;
        }
        toolCallsByIndex.set(index, partial);
        sawOutput = true;
      }
    }

    return false;
  };

  let done = false;
  while (!done) {
    const { value, done: streamDone } = await reader.read();
    if (streamDone) {
      // Flush any trailing event without a closing blank line.
      if (buffer.trim().length > 0) {
        done = await processEvent(buffer);
        buffer = "";
      }
      break;
    }
    // Normalize CRLF so events split on \n\n regardless of server style.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let separator: number;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      done = await processEvent(rawEvent);
      if (done) {
        break;
      }
    }
  }

  if (!sawOutput && !sawFinishReason) {
    throw new Error("No output generated. The model stream ended without a finish chunk.");
  }

  const toolCalls: LlmToolCall[] = [];
  for (const partial of toolCallsByIndex.values()) {
    if (!partial.id) {
      continue;
    }
    let input: unknown;
    try {
      input = JSON.parse(partial.arguments || "{}");
    } catch {
      input = { command: partial.arguments };
    }
    toolCalls.push({
      id: partial.id,
      name: partial.name || "bash",
      input,
    });
  }

  const llmMs = Date.now() - requestStart;
  const thinkingMs = (firstTextAt ?? lastReasoningAt ?? requestStart + llmMs) - requestStart;
  const answeringMs = firstTextAt !== null ? requestStart + llmMs - firstTextAt : 0;

  return {
    text,
    toolCalls,
    finishReason: finishReason === "unknown" && toolCalls.length > 0 ? "tool-calls" : finishReason,
    usage: parseDeepSeekUsage(rawUsage, { llmMs, thinkingMs, answeringMs }),
  };
}
