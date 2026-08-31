import type { LlmMessage, ModelId, ThinkingEffort } from './storage.js';

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
  /**
   * Opaque provider replay data for the reasoning block. This may contain a
   * provider signature or serialized structured reasoning details; persist it
   * unchanged alongside `reasoning` when storage supports it.
   */
  reasoningSignature?: string;
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
  /**
   * Stable per-session identifier for the LLM provider. OpenRouter uses it
   * as its `session_id` to pin provider sticky routing (and Z.AI's session
   * affinity key) from the very first request: without it, routing only
   * sticks after a cache hit has been observed, so a session whose request
   * shape changes mid-turn (e.g. read_media injecting image/audio content)
   * can be re-routed to another provider and lose the whole cached prefix.
   */
  sessionId?: string;
}

/** Provider-neutral pricing in the provider's billing currency. */
export interface GenericPricing {
  currency: string;
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
 * Kept as the OpenAI-compatible source schema. The pi-ai adapter converts it
 * to Pi's tool representation without changing its name, description, or
 * JSON Schema across providers.
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
