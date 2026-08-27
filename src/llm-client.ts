import type { LlmMessage, ModelId, ThinkingEffort, UserContentPart } from './storage.js';

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
    description: 'Read a local image or audio file.',
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
