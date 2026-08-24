import type { LlmMessage, ModelId, ThinkingEffort } from "./storage.js";
import { SYSTEM_PROMPT } from "./system-prompt.js";

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
  /** Full reasoning emitted by the model during this step. */
  reasoning: string;
  toolCalls: LlmToolCall[];
  finishReason: string;
  /** Token usage and timing for this LLM step, if reported by the provider. */
  usage: LlmUsage | null;
}

export interface LlmStepOptions {
  messages: LlmMessage[];
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  model?: ModelId;
  thinkingEffort?: ThinkingEffort;
  system?: string;
}

/** Provider-neutral pricing in the provider's billing currency. */
export interface GenericPricing {
  currency: "CNY" | "USD";
  /** Price per 1M input tokens served from cache. */
  cacheHitPerM: number;
  /** Price per 1M input tokens not served from cache. */
  cacheMissPerM: number;
  /** Price per 1M output tokens. */
  outputPerM: number;
}

/** Cost of a single LLM step's token usage under the given pricing. */
export function costFromUsage(usage: LlmUsage, pricing: GenericPricing): number {
  return (
    (usage.cacheReadTokens / 1_000_000) * pricing.cacheHitPerM +
    (usage.cacheMissTokens / 1_000_000) * pricing.cacheMissPerM +
    (usage.outputTokens / 1_000_000) * pricing.outputPerM
  );
}

/**
 * The bash tool exposed to the model, in OpenAI function-calling wire format.
 * Kept as a plain object (no AI SDK `tool()` wrapper) because we talk to
 * OpenAI-compatible endpoints directly (see runChatCompletions) — the
 * description and schema must stay byte-identical so the model's behavior
 * does not change across providers.
 */
export const BASH_TOOL_SCHEMA = {
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

/**
 * Convert our stored AI-SDK ModelMessage history to the OpenAI wire format.
 * `reasoningMessageField` is the provider-specific field used to send stored
 * reasoning back in assistant history messages ("reasoning_content" for
 * DeepSeek, "reasoning" for OpenRouter).
 */
export function toOpenAiMessages(
  messages: LlmMessage[],
  reasoningMessageField: string,
): unknown[] {
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
        const userMessage: Record<string, unknown> = { role: "user", content };
        if ("name" in message && typeof message.name === "string" && message.name.length > 0) {
          userMessage.name = message.name;
        }
        out.push(userMessage);
        break;
      }
      case "assistant": {
        const parts = Array.isArray(message.content) ? message.content : [];
        const text = parts
          .filter((part) => part.type === "text")
          .map((part) => (part as { text: string }).text)
          .join("");
        const reasoning = parts
          .filter((part) => part.type === "reasoning")
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
        if (toolCalls.length > 0 && parts.some((part) => part.type === "reasoning")) {
          assistantMessage[reasoningMessageField] = reasoning;
        }
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

/** A single parsed SSE chunk from an OpenAI-compatible chat completions stream. */
interface WireChunk {
  choices?: Array<{
    index: number;
    delta?: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
      [field: string]: unknown;
    };
    finish_reason?: string | null;
  }>;
  usage?: unknown;
  error?: { message?: string };
}

/** Partial tool call accumulated from streaming `delta.tool_calls` fragments. */
interface PartialToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface ChatCompletionsOptions {
  /** Base URL without trailing slash, e.g. "https://api.deepseek.com". */
  baseUrl: string;
  apiKey: string;
  /** Provider name used in error messages, e.g. "DeepSeek". */
  label: string;
  model: string;
  messages: LlmMessage[];
  system?: string;
  thinkingEffort?: ThinkingEffort;
  /**
   * Wire field used to send stored reasoning back in assistant history
   * messages ("reasoning_content" for DeepSeek, "reasoning" for OpenRouter).
   */
  reasoningMessageField: string;
  /** Delta fields that may carry reasoning tokens while streaming. */
  reasoningDeltaFields: readonly string[];
  /** Extra request body fields for this provider (e.g. OpenRouter's stream_options). */
  extraBody?: Record<string, unknown>;
  /** Extra request headers for this provider (e.g. OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string>;
  /** Maps a thinking effort to extra body fields; return undefined to omit them. */
  effortBody?: (effort: ThinkingEffort) => Record<string, unknown> | undefined;
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  /**
   * Provider-specific usage parser. Cache/reasoning token fields differ per
   * provider (DeepSeek's prompt_cache_hit_tokens vs OpenRouter's normalized
   * shape), so the raw usage chunk is handed to the provider.
   */
  parseUsage: (
    raw: unknown,
    timing: { llmMs: number; thinkingMs: number; answeringMs: number },
  ) => LlmUsage | null;
}

/**
 * Calls an OpenAI-compatible chat completions API DIRECTLY and parses the SSE
 * stream ourselves.
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
 * This also lets us read provider-specific raw usage fields (cache tokens,
 * reasoning tokens) which the SDK's zod schema strips.
 */
export async function runChatCompletions(
  options: ChatCompletionsOptions,
): Promise<LlmStepResult> {
  const wireMessages = toOpenAiMessages(options.messages, options.reasoningMessageField);
  const body: Record<string, unknown> = {
    model: options.model,
    messages: [
      { role: "system", content: options.system ?? SYSTEM_PROMPT },
      ...(wireMessages.filter(
        (message) => (message as { role?: string }).role !== "system",
      ) as Array<Record<string, unknown>>),
    ],
    tools: [BASH_TOOL_SCHEMA],
    stream: true,
    ...options.extraBody,
    ...(options.effortBody?.(options.thinkingEffort ?? "off") ?? {}),
  };

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
  let reasoning = "";
  let finishReason = "unknown";
  let sawFinishReason = false;
  let sawOutput = false;
  let rawUsage: unknown;
  const toolCallsByIndex = new Map<number, PartialToolCall>();

  const fetchWithRetry = async (): Promise<Response> => {
    let lastError: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await fetch(`${options.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${options.apiKey}`,
            ...options.extraHeaders,
          },
          body: JSON.stringify(body),
          signal: options.signal,
        });
        if (response.ok) {
          return response;
        }
        const errorBody = await response.text().catch(() => "");
        lastError = new Error(
          `${options.label} API error ${response.status}: ${errorBody.slice(0, 500)}`,
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
    throw lastError ?? new Error(`${options.label} request failed`);
  };

  const response = await fetchWithRetry();
  if (!response.body) {
    throw new Error(`${options.label} response has no body`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processEvent = async (rawEvent: string): Promise<boolean> => {
    // SSE spec: consecutive `data:` lines are joined with \n. Some servers
    // emit one data line per event; DeepSeek and OpenRouter use single lines.
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

    let chunk: WireChunk;
    try {
      chunk = JSON.parse(data) as WireChunk;
    } catch {
      return false;
    }

    if (chunk.error) {
      throw new Error(`${options.label} stream error: ${chunk.error.message ?? "unknown"}`);
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

    // Reasoning tokens (DeepSeek's `reasoning_content`, OpenRouter's
    // `reasoning`) — forwarded LIVE, which is the whole point of bypassing
    // the AI SDK here.
    for (const field of options.reasoningDeltaFields) {
      const reasoningDelta = delta[field];
      if (typeof reasoningDelta === "string" && reasoningDelta.length > 0) {
        reasoning += reasoningDelta;
        lastReasoningAt = Date.now();
        sawOutput = true;
        await options.onReasoningDelta?.(reasoningDelta);
        break;
      }
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
    reasoning,
    toolCalls,
    finishReason: finishReason === "unknown" && toolCalls.length > 0 ? "tool-calls" : finishReason,
    usage: options.parseUsage(rawUsage, { llmMs, thinkingMs, answeringMs }),
  };
}
