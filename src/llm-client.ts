import type { LlmMessage, ModelId, ThinkingEffort, UserContentPart } from './storage.js';
import { healMessages } from './heal.js';
import { waitForChatRateLimit } from './rate-limit.js';
import { fetchWithRetry, type RetryOptions } from './retry.js';
import { SYSTEM_PROMPT } from './system-prompt.js';
import { envPositiveInt } from './env.js';

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

/**
 * Shared LlmUsage assembly for the provider-specific usage parsers: fills in
 * timing, derives totalTokens, and treats an all-zero report as "the
 * provider sent no usable usage" (null).
 */
export function buildLlmUsage(
  totals: {
    inputTokens: number;
    outputTokens: number;
    /** Provider-reported total; falls back to input+output. */
    totalTokens?: number;
    cacheReadTokens: number;
    cacheMissTokens: number;
    reasoningTokens: number;
  },
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (
    totals.inputTokens === 0 &&
    totals.outputTokens === 0 &&
    totals.cacheReadTokens === 0 &&
    totals.cacheMissTokens === 0
  ) {
    return null;
  }
  return {
    inputTokens: totals.inputTokens,
    outputTokens: totals.outputTokens,
    totalTokens:
      totals.totalTokens && totals.totalTokens > 0
        ? totals.totalTokens
        : totals.inputTokens + totals.outputTokens,
    cacheReadTokens: totals.cacheReadTokens,
    cacheMissTokens: totals.cacheMissTokens,
    reasoningTokens: totals.reasoningTokens,
    llmMs: timing.llmMs,
    thinkingMs: timing.thinkingMs,
    answeringMs: timing.answeringMs,
  };
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
  /**
   * Optional debug-log sink (wired to the session's log.jsonl by ZenAgent).
   * Used for events worth correlating with a session, e.g. long client-side
   * rate-limit waits. Fire-and-forget from the caller's perspective.
   */
  logRuntime?: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    details?: Record<string, unknown>,
  ) => void;
  /**
   * Tool schemas offered to the model. Defaults to the bash tool; sessions
   * on multimodal models append read_media (see ZenAgent.sessionTools). Must
   * stay stable within a session: the list is part of the cached prefix.
   */
  tools?: unknown[];
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  model?: ModelId;
  thinkingEffort?: ThinkingEffort;
  system?: string;
}

/** Provider-neutral pricing in the provider's billing currency. */
export interface GenericPricing {
  currency: 'CNY' | 'USD';
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
  type: 'function',
  function: {
    name: 'bash',
    description:
      'Execute a bash command in current OS. The command is completely unrestricted. Your command will be wrapped inside `script -q -e -c "bash <script file containing your command>" "<log path>"`. If output is large, this tool will tell you to check the log file instead of showing all.',
    parameters: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'The bash command to execute.',
        },
      },
      required: ['command'],
      additionalProperties: false,
    },
  },
} as const;

/** Audio formats accepted by OpenAI-compatible `input_audio` parts. */
const INPUT_AUDIO_FORMATS: Record<string, string> = {
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/wave': 'wav',
  'audio/vnd.wave': 'wav',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
  'audio/mpeg3': 'mp3',
};

function inputAudioFormat(mimeType: string): string | null {
  return INPUT_AUDIO_FORMATS[mimeType.toLowerCase()] ?? null;
}

/**
 * Multi-part user content to OpenAI-compatible wire parts: text stays text,
 * images become `image_url` data URIs and audio becomes `input_audio`
 * (base64 + format; URLs are not supported for audio by the API).
 */
export function userPartsToOpenAi(parts: UserContentPart[]): unknown[] {
  const out: unknown[] = [];
  for (const part of parts) {
    switch (part.type) {
      case 'text':
        out.push({ type: 'text', text: part.text });
        break;
      case 'image':
        out.push({
          type: 'image_url',
          image_url: { url: `data:${part.mimeType};base64,${part.data}` },
        });
        break;
      case 'audio': {
        const format = inputAudioFormat(part.mimeType);
        if (format === null) {
          // Unsupported container: degrade instead of failing the request.
          out.push({
            type: 'text',
            text: `[audio attached (${part.mimeType}) omitted: unsupported format]`,
          });
          break;
        }
        out.push({
          type: 'input_audio',
          input_audio: { data: part.data, format },
        });
        break;
      }
    }
  }
  return out;
}

/**
 * The read_media tool: lets the model view an image or audio file by itself.
 * Only offered when the active model accepts the modality; the harness reads
 * the file and injects the payload as media parts in the following user
 * message (the OpenAI tool role only allows text content).
 */
export const READ_MEDIA_TOOL_SCHEMA = {
  type: 'function',
  function: {
    name: 'read_media',
    description:
      'Load a local image or audio file so you can see/hear its content yourself (no user description needed). Use for screenshots, photos, diagrams, recordings, or any media file the user references by path. Returns the media attached to the conversation; a short metadata line confirms what was loaded.',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description:
            'Path to the media file (absolute, or relative to the working directory). Must be an image (png/jpeg/webp/gif) or audio (wav/mp3) file.',
        },
      },
      required: ['path'],
      additionalProperties: false,
    },
  },
} as const;

/**
 * Convert our stored message history to the OpenAI wire format.
 * `reasoningMessageField` is the provider-specific field used to send stored
 * reasoning back in assistant history messages ("reasoning_content" for
 * DeepSeek, "reasoning" for OpenRouter).
 */
export function toOpenAiMessages(messages: LlmMessage[], reasoningMessageField: string): unknown[] {
  const out: unknown[] = [];

  for (const message of messages) {
    switch (message.role) {
      case 'user': {
        const content =
          typeof message.content === 'string'
            ? message.content
            : userPartsToOpenAi(message.content);
        const userMessage: Record<string, unknown> = { role: 'user', content };
        if ('name' in message && typeof message.name === 'string' && message.name.length > 0) {
          userMessage.name = message.name;
        }
        out.push(userMessage);
        break;
      }
      case 'assistant': {
        const parts = Array.isArray(message.content) ? message.content : [];
        const text = parts
          .filter((part) => part.type === 'text')
          .map((part) => (part as { text: string }).text)
          .join('');
        const reasoning = parts
          .filter((part) => part.type === 'reasoning')
          .map((part) => (part as { text: string }).text)
          .join('');
        const toolCalls = parts
          .filter((part) => part.type === 'tool-call')
          .map((part) => ({
            id: (part as { toolCallId: string }).toolCallId,
            type: 'function',
            function: {
              name: (part as { toolName: string }).toolName,
              arguments: JSON.stringify((part as { input: unknown }).input),
            },
          }));
        const assistantMessage: Record<string, unknown> = {
          role: 'assistant',
          content: text || null,
        };
        // Reasoning is replayed only alongside tool-call continuations:
        // thinking-mode endpoints expect their reasoning field echoed back
        // while they continue a tool exchange. Final-answer assistants omit
        // it here; runChatCompletions backfills an empty reasoning field for
        // thinking-mode sessions so the wire shape stays valid either way.
        if (toolCalls.length > 0 && parts.some((part) => part.type === 'reasoning')) {
          assistantMessage[reasoningMessageField] = reasoning;
        }
        if (toolCalls.length > 0) {
          assistantMessage.tool_calls = toolCalls;
        }
        out.push(assistantMessage);
        break;
      }
      case 'tool': {
        const parts = Array.isArray(message.content) ? message.content : [];
        for (const part of parts) {
          if (part.type !== 'tool-result') {
            continue;
          }
          const output = (part as { output: unknown }).output;
          const text =
            typeof output === 'string'
              ? output
              : typeof output === 'object' &&
                  output !== null &&
                  'value' in (output as Record<string, unknown>) &&
                  typeof (output as Record<string, unknown>).value === 'string'
                ? ((output as Record<string, unknown>).value as string)
                : JSON.stringify(output);
          out.push({
            role: 'tool',
            tool_call_id: (part as { toolCallId: string }).toolCallId,
            content: text,
          });
        }
        break;
      }
    }
  }

  return out;
}

function mapFinishReason(raw: string | null | undefined): string {
  switch (raw) {
    case 'stop':
      return 'stop';
    case 'length':
      return 'length';
    case 'tool_calls':
      return 'tool-calls';
    case 'content_filter':
      return 'content-filter';
    case 'insufficient_system_resource':
      return 'error';
    default:
      return raw ? 'other' : 'unknown';
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
  /** Tool schemas offered to the model; defaults to the bash tool. */
  tools?: unknown[];
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
  /** Retry configuration for the initial chat request. Pass `{ maxAttempts: 1 }` to disable retries. */
  retry?: RetryOptions;
  /** Debug-log sink for provider-internal diagnostics (see LlmStepOptions). */
  logRuntime?: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    details?: Record<string, unknown>,
  ) => void;
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
 * Hard cap on a single chat request in ms. The provider's own timeout usually
 * closes the connection first (clean EOF → natural retry); this is the safety
 * net for genuinely hung sockets. Override with ZEN_AGENT_CHAT_TIMEOUT_MS.
 */
/** Rate-limit waits longer than this are recorded via logRuntime (log.jsonl). */
const RATE_LIMIT_WAIT_LOG_THRESHOLD_MS = 1_000;

function parseChatTimeoutMs(): number {
  return envPositiveInt('ZEN_AGENT_CHAT_TIMEOUT_MS', 660_000);
}

/**
 * Locates the next SSE event terminator in the ASSEMBLED buffer ("\n\n", or
 * CRLF "\r\n\r\n"). Terminators must be searched after assembly: normalizing
 * each network chunk separately misses a "\r\n\r\n" that is split across two
 * reads, which would glue consecutive events together.
 */
function findEventSeparator(buffer: string): { index: number; length: number } | null {
  const lf = buffer.indexOf('\n\n');
  const crlf = buffer.indexOf('\r\n\r\n');
  if (crlf !== -1 && (lf === -1 || crlf < lf)) {
    return { index: crlf, length: 4 };
  }
  if (lf !== -1) {
    return { index: lf, length: 2 };
  }
  return null;
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
export async function runChatCompletions(options: ChatCompletionsOptions): Promise<LlmStepResult> {
  // Heal the history before sending: drop unpaired assistant tool calls and
  // stray tool results (DeepSeek 400s on either shape) without mutating the
  // session's stored messages.
  const healed = healMessages(options.messages);
  if (healed.droppedAssistants > 0 || healed.droppedTools > 0) {
    // Healing silently loses history by design; the drop must be visible in
    // the session's log.jsonl (via the agent's logRuntime sink) rather than
    // on stdout/stderr, which Zed swallows.
    void options.logRuntime?.('warn', 'healed message history before LLM request', {
      label: options.label,
      model: options.model,
      droppedAssistants: healed.droppedAssistants,
      droppedTools: healed.droppedTools,
    });
  }
  const wireMessages = toOpenAiMessages(healed.messages, options.reasoningMessageField);

  // Thinking-mode DeepSeek 400s on assistant messages without
  // `reasoning_content`. Back-fill "" — thinking-mode sessions only, because
  // on non-thinking sessions the extra field would churn the prefix cache.
  if (
    (options.thinkingEffort ?? 'off') !== 'off' &&
    options.reasoningMessageField === 'reasoning_content'
  ) {
    for (const message of wireMessages) {
      const wire = message as { role?: string; [key: string]: unknown };
      if (wire.role === 'assistant' && typeof wire[options.reasoningMessageField] !== 'string') {
        wire[options.reasoningMessageField] = '';
      }
    }
  }
  const body: Record<string, unknown> = {
    model: options.model,
    messages: [
      { role: 'system', content: options.system ?? SYSTEM_PROMPT },
      ...(wireMessages.filter(
        (message) => (message as { role?: string }).role !== 'system',
      ) as Array<Record<string, unknown>>),
    ],
    tools: options.tools ?? [BASH_TOOL_SCHEMA],
    stream: true,
    ...options.extraBody,
    ...(options.effortBody?.(options.thinkingEffort ?? 'off') ?? {}),
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

  let text = '';
  let reasoning = '';
  let finishReason = 'unknown';
  let sawFinishReason = false;
  let sawOutput = false;
  let rawUsage: unknown;
  const toolCallsByIndex = new Map<number, PartialToolCall>();

  // Hard cap on a single request: the provider's own timeout usually closes
  // the connection first (clean EOF → natural retry); this timer is the
  // safety net for genuinely hung sockets.
  const timeoutMs = parseChatTimeoutMs();
  const timeoutCtrl = new AbortController();
  // Combine — `options.signal ?? timeoutCtrl.signal` orphans the timer when
  // the caller passes a signal, so timeoutMs never reaches fetch.
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeoutCtrl.signal])
    : timeoutCtrl.signal;

  let timer: NodeJS.Timeout | undefined;
  try {
    // Rate-limit waiting happens BEFORE the chat timeout starts ticking:
    // queueing behind ZEN_AGENT_CHAT_RPM is not the request's time, and
    // arming earlier would kill queued requests before they were ever sent.
    // Only the caller's signal can interrupt the wait itself.
    const rateLimitStart = Date.now();
    await waitForChatRateLimit(options.signal);
    const rateLimitedMs = Date.now() - rateLimitStart;
    if (rateLimitedMs >= RATE_LIMIT_WAIT_LOG_THRESHOLD_MS) {
      void options.logRuntime?.('info', 'chat request delayed by client-side rate limit', {
        label: options.label,
        model: options.model,
        waitedMs: rateLimitedMs,
      });
    }
    timer = setTimeout(() => {
      timeoutCtrl.abort(new Error(`${options.label} request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    timer.unref?.();

    const response = await fetchWithRetry(
      fetch,
      `${options.baseUrl}/chat/completions`,
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${options.apiKey}`,
          ...options.extraHeaders,
        },
        body: JSON.stringify(body),
        signal,
      },
      { ...options.retry, signal },
    );
    if (!response.ok) {
      // Only the initial fetch is retried (see fetchWithRetry); a non-2xx here
      // means the status is non-retryable or attempts ran out — surface it.
      const errorBody = await response.text().catch(() => '');
      throw new Error(`${options.label} API error ${response.status}: ${errorBody.slice(0, 500)}`);
    }
    if (!response.body) {
      throw new Error(`${options.label} response has no body`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const processEvent = async (rawEvent: string): Promise<boolean> => {
      // SSE spec: consecutive `data:` lines are joined with \n. Some servers
      // emit one data line per event; DeepSeek and OpenRouter use single lines.
      const dataLines: string[] = [];
      for (const line of rawEvent.split(/\r?\n/)) {
        if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (dataLines.length === 0) {
        return false;
      }
      const data = dataLines.join('\n');
      if (data === '[DONE]') {
        return true;
      }

      let chunk: WireChunk;
      try {
        chunk = JSON.parse(data) as WireChunk;
      } catch {
        return false;
      }

      if (chunk.error) {
        throw new Error(`${options.label} stream error: ${chunk.error.message ?? 'unknown'}`);
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
        if (typeof reasoningDelta === 'string' && reasoningDelta.length > 0) {
          reasoning += reasoningDelta;
          lastReasoningAt = Date.now();
          sawOutput = true;
          await options.onReasoningDelta?.(reasoningDelta);
          break;
        }
      }

      // Answer text.
      if (typeof delta.content === 'string' && delta.content.length > 0) {
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
            id: '',
            name: '',
            arguments: '',
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
          buffer = '';
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      // Split on event terminators found in the assembled buffer (see
      // findEventSeparator): works for LF and CRLF servers regardless of how
      // the bytes land in chunks.
      for (;;) {
        const separator = findEventSeparator(buffer);
        if (!separator) {
          break;
        }
        const rawEvent = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator.length);
        done = await processEvent(rawEvent);
        if (done) {
          break;
        }
      }
    }

    if (!sawOutput && !sawFinishReason) {
      throw new Error('No output generated. The model stream ended without a finish chunk.');
    }

    const toolCalls: LlmToolCall[] = [];
    for (const partial of toolCallsByIndex.values()) {
      if (!partial.id) {
        continue;
      }
      let input: unknown;
      try {
        input = JSON.parse(partial.arguments || '{}');
      } catch {
        // Never parsed: keep the raw string under an explicit marker instead
        // of guessing the tool's shape (executeLlmToolCall reports it).
        input = { malformed_arguments: partial.arguments };
      }
      toolCalls.push({
        id: partial.id,
        name: partial.name || 'bash',
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
      finishReason:
        finishReason === 'unknown' && toolCalls.length > 0 ? 'tool-calls' : finishReason,
      usage: options.parseUsage(rawUsage, { llmMs, thinkingMs, answeringMs }),
    };
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}
