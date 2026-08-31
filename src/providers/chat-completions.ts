import {
  type AssistantMessage as PiAssistantMessage,
  type Model as PiModel,
} from '@earendil-works/pi-ai';
import { stream as streamOpenAiCompletions } from '@earendil-works/pi-ai/api/openai-completions';
import type { LlmMessage, ThinkingEffort } from '../session/storage.js';
import { mapFinishReason, patchPayload, toLlmUsage, toPiContext, toPiTools } from './convert.js';
import { healMessages } from './heal.js';
import { waitForChatRateLimit } from './rate-limit.js';
import { SYSTEM_PROMPT } from '../session/system-prompt.js';
import { envPositiveInt } from '../util/env.js';
import type { LlmStepResult, LlmToolCall } from './llm-client.js';

export interface ChatCompletionsOptions {
  /**
   * Pi model for the provider/model, supplied by the provider registry
   * (carries baseUrl, compat, thinkingLevelMap and per-model metadata).
   */
  model: PiModel<'openai-completions'>;
  /** API key sent to the endpoint; resolved from the provider definition. */
  apiKey: string;
  /** Provider name used in Zen's error messages, e.g. "DeepSeek". */
  label: string;
  messages: LlmMessage[];
  /** Tool schemas offered to the model; defaults to the bash tool. */
  tools?: unknown[];
  system?: string;
  thinkingEffort?: ThinkingEffort;
  /** Extra request fields for this provider (e.g. OpenRouter routing). */
  extraBody?: Record<string, unknown>;
  /** Extra request headers for this provider (e.g. OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string>;
  /** Stable provider-side cache/routing session identity. */
  sessionId?: string;
  /**
   * Maximum total attempts for the initial chat request, including the first.
   * Default 4. Pi (the provider library) owns retryable-status selection,
   * `Retry-After` handling, and never re-sending a failed mid-stream request.
   */
  maxAttempts?: number;
  /** Upper bound on any single retry delay. */
  maxRetryDelayMs?: number;
  /** Debug-log sink for provider-internal diagnostics (see LlmStepOptions). */
  logRuntime?: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    details?: Record<string, unknown>,
  ) => void;
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
}

/**
 * Pi takes a retry COUNT (0 = never retry); `maxAttempts` is the total number
 * of attempts including the first, so the mapping is `attempts - 1`, clamped
 * at 0. Default: 4 attempts -> 3 retries.
 */
export function piMaxRetries(maxAttempts: number | undefined): number {
  return Math.max(0, (maxAttempts ?? 4) - 1);
}

/**
 * Hard cap on a single chat request in ms. The provider's own timeout usually
 * closes the connection first; this is the safety net for genuinely hung
 * sockets. Override with ZEN_AGENT_CHAT_TIMEOUT_MS.
 */
const RATE_LIMIT_WAIT_LOG_THRESHOLD_MS = 1_000;

function parseChatTimeoutMs(): number {
  return envPositiveInt('ZEN_AGENT_CHAT_TIMEOUT_MS', 6_600_000);
}

function piStreamError(label: string, message: string): Error {
  const detailed = /^(\d{3}):\s*([\s\S]*)$/.exec(message);
  if (detailed) {
    return new Error(`${label} API error ${detailed[1]}: ${detailed[2]}`);
  }
  const statusOnly = /^(\d{3})\s+status code(?:\s+\(no body\))?$/.exec(message);
  if (statusOnly) {
    return new Error(`${label} API error ${statusOnly[1]}: `);
  }
  return new Error(message);
}

/**
 * Calls an OpenAI-compatible chat-completions endpoint through pi-ai.
 * Pi owns payload construction, provider compatibility handling, retries, and
 * SSE parsing; this adapter only bridges Zen's persisted message/result shape.
 */
export async function runChatCompletions(options: ChatCompletionsOptions): Promise<LlmStepResult> {
  const healed = healMessages(options.messages);
  if (healed.droppedAssistants > 0 || healed.droppedTools > 0) {
    void options.logRuntime?.('warn', 'healed message history before LLM request', {
      label: options.label,
      model: options.model.id,
      droppedAssistants: healed.droppedAssistants,
      droppedTools: healed.droppedTools,
    });
  }

  const model = options.model;
  const piTools = toPiTools(options.tools);
  const context = toPiContext(healed.messages, model, options.system ?? SYSTEM_PROMPT, piTools);

  const requestStart = Date.now();
  let firstTextAt: number | null = null;
  let lastReasoningAt: number | null = null;
  let text = '';
  let reasoning = '';
  let finalMessage: PiAssistantMessage | undefined;

  const timeoutMs = parseChatTimeoutMs();
  const timeoutCtrl = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeoutCtrl.signal])
    : timeoutCtrl.signal;

  let timer: NodeJS.Timeout | undefined;
  try {
    const rateLimitStart = Date.now();
    await waitForChatRateLimit(options.signal);
    const rateLimitedMs = Date.now() - rateLimitStart;
    if (rateLimitedMs >= RATE_LIMIT_WAIT_LOG_THRESHOLD_MS) {
      void options.logRuntime?.('info', 'chat request delayed by client-side rate limit', {
        label: options.label,
        model: options.model.id,
        waitedMs: rateLimitedMs,
      });
    }

    timer = setTimeout(() => {
      timeoutCtrl.abort(new Error(`${options.label} request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    timer.unref?.();

    const stream = streamOpenAiCompletions(model, context, {
      apiKey: options.apiKey,
      headers: options.extraHeaders,
      signal,
      sessionId: options.sessionId,
      reasoningEffort:
        options.thinkingEffort && options.thinkingEffort !== 'off'
          ? options.thinkingEffort
          : undefined,
      maxRetries: piMaxRetries(options.maxAttempts),
      maxRetryDelayMs: options.maxRetryDelayMs,
      samplingParams: options.extraBody,
      onPayload: (payload) => patchPayload(payload, healed.messages, options.tools?.length === 0),
    });

    for await (const event of stream) {
      switch (event.type) {
        case 'text_delta':
          if (firstTextAt === null) {
            firstTextAt = Date.now();
          }
          text += event.delta;
          await options.onTextDelta?.(event.delta);
          break;
        case 'thinking_delta':
          reasoning += event.delta;
          lastReasoningAt = Date.now();
          await options.onReasoningDelta?.(event.delta);
          break;
        case 'done':
          finalMessage = event.message;
          break;
        case 'error':
          if (timeoutCtrl.signal.aborted && timeoutCtrl.signal.reason instanceof Error) {
            throw timeoutCtrl.signal.reason;
          }
          throw piStreamError(options.label, event.error.errorMessage ?? 'Unknown provider error');
      }
    }

    if (!finalMessage) {
      throw new Error('No output generated. The model stream ended without a completion event.');
    }
    const completedMessage = finalMessage;

    // Pi's final message is authoritative even if a provider emitted no delta
    // event for a content block.
    if (text.length === 0) {
      text = completedMessage.content
        .filter(
          (part): part is Extract<(typeof completedMessage.content)[number], { type: 'text' }> =>
            part.type === 'text',
        )
        .map((part) => part.text)
        .join('');
    }
    const thinkingParts = completedMessage.content.filter(
      (part): part is Extract<(typeof completedMessage.content)[number], { type: 'thinking' }> =>
        part.type === 'thinking',
    );
    if (reasoning.length === 0) {
      reasoning = thinkingParts.map((part) => part.thinking).join('');
    }
    const reasoningSignature = thinkingParts.find(
      (part) => typeof part.thinkingSignature === 'string' && part.thinkingSignature.length > 0,
    )?.thinkingSignature;

    const toolCalls: LlmToolCall[] = completedMessage.content
      .filter(
        (part): part is Extract<(typeof completedMessage.content)[number], { type: 'toolCall' }> =>
          part.type === 'toolCall',
      )
      .map((part) => ({ id: part.id, name: part.name || 'bash', input: part.arguments }));

    const llmMs = Date.now() - requestStart;
    const thinkingMs = (firstTextAt ?? lastReasoningAt ?? requestStart + llmMs) - requestStart;
    const answeringMs = firstTextAt !== null ? requestStart + llmMs - firstTextAt : 0;

    return {
      text,
      reasoning,
      ...(reasoningSignature !== undefined ? { reasoningSignature } : {}),
      toolCalls,
      finishReason: mapFinishReason(completedMessage.stopReason),
      usage: toLlmUsage(completedMessage.usage, { llmMs, thinkingMs, answeringMs }),
    };
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}
