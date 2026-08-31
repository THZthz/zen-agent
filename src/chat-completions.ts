import {
  type AssistantMessage as PiAssistantMessage,
  type Context as PiContext,
  type Model as PiModel,
  type OpenAICompletionsCompat,
  type Tool as PiTool,
} from '@earendil-works/pi-ai';
import { stream as streamOpenAiCompletions } from '@earendil-works/pi-ai/api/openai-completions';
import type { LlmMessage, ThinkingEffort, UserContentPart } from './storage.js';
import { healMessages } from './heal.js';
import { waitForChatRateLimit } from './rate-limit.js';
import type { RetryOptions } from './retry.js';
import { SYSTEM_PROMPT } from './system-prompt.js';
import { envPositiveInt } from './env.js';
import {
  BASH_TOOL_SCHEMA,
  buildLlmUsage,
  type LlmStepResult,
  type LlmToolCall,
  type LlmUsage,
} from './llm-client.js';

export interface ChatCompletionsOptions {
  /** Base URL without trailing slash, e.g. "https://api.deepseek.com". */
  baseUrl: string;
  apiKey: string;
  /** Pi provider id, used for compatibility detection and message replay. */
  provider: string;
  /** Provider name used in Zen's error messages, e.g. "DeepSeek". */
  label: string;
  model: string;
  messages: LlmMessage[];
  /** Tool schemas offered to the model; defaults to the bash tool. */
  tools?: unknown[];
  system?: string;
  thinkingEffort?: ThinkingEffort;
  /** Provider/model-specific Pi compatibility configuration. */
  compat?: OpenAICompletionsCompat;
  /** Pi thinking-level mapping for this provider/model. */
  thinkingLevelMap?: Partial<Record<ThinkingEffort, string | null>>;
  /** Extra request fields for this provider (e.g. OpenRouter routing). */
  extraBody?: Record<string, unknown>;
  /** Extra request headers for this provider (e.g. OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string>;
  /** Stable provider-side cache/routing session identity. */
  sessionId?: string;
  /** Retry configuration for the initial chat request. */
  retry?: RetryOptions;
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
 * Hard cap on a single chat request in ms. The provider's own timeout usually
 * closes the connection first; this is the safety net for genuinely hung
 * sockets. Override with ZEN_AGENT_CHAT_TIMEOUT_MS.
 */
const RATE_LIMIT_WAIT_LOG_THRESHOLD_MS = 1_000;

function parseChatTimeoutMs(): number {
  return envPositiveInt('ZEN_AGENT_CHAT_TIMEOUT_MS', 6_600_000);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toolFromSchema(schema: unknown): PiTool {
  const functionSpec = isRecord(schema) && isRecord(schema.function) ? schema.function : schema;
  if (
    !isRecord(functionSpec) ||
    typeof functionSpec.name !== 'string' ||
    typeof functionSpec.description !== 'string' ||
    !isRecord(functionSpec.parameters)
  ) {
    throw new Error('Tool schemas must use the OpenAI function-calling format.');
  }
  // Pi's Tool generic is TypeBox-oriented, but its OpenAI adapter accepts the
  // JSON Schema objects Zen already persists and sends.
  return {
    name: functionSpec.name,
    description: functionSpec.description,
    parameters: functionSpec.parameters,
  } as PiTool;
}

function toPiTools(tools: unknown[] | undefined): PiTool[] | undefined {
  if (tools !== undefined && tools.length === 0) {
    return undefined;
  }
  return (tools ?? [BASH_TOOL_SCHEMA]).map(toolFromSchema);
}

/** Pi's public chat context has text/image blocks; model audio stays OpenAI-compatible at payload time. */
function toPiUserContent(
  content: string | UserContentPart[],
):
  | string
  | Array<{ type: 'text'; text: string } | { type: 'image'; mimeType: string; data: string }> {
  if (typeof content === 'string') {
    return content;
  }
  return content.map((part) => {
    if (part.type === 'text') {
      return { type: 'text' as const, text: part.text };
    }
    // Pi currently models images, not OpenAI `input_audio`. Keep audio as an
    // image-shaped transport block and restore its OpenAI wire representation
    // in patchPayload before the request leaves the process.
    return { type: 'image' as const, mimeType: part.mimeType, data: part.data };
  });
}

function reasoningReplayField(model: PiModel<'openai-completions'>): string | undefined {
  if (model.compat?.requiresReasoningContentOnAssistantMessages) {
    return 'reasoning_content';
  }
  return model.compat?.thinkingFormat === 'openrouter' ? 'reasoning' : undefined;
}

type StoredReasoningPart = { type: 'reasoning'; text: string };
type StoredReasoningPartWithSignature = StoredReasoningPart & {
  reasoningSignature?: unknown;
  thinkingSignature?: unknown;
  signature?: unknown;
};

/** Read forward-compatible replay metadata without widening storage's current type. */
function storedReasoningSignature(part: StoredReasoningPart): string | undefined {
  const signed = part as StoredReasoningPartWithSignature;
  for (const candidate of [signed.reasoningSignature, signed.thinkingSignature, signed.signature]) {
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }
  return undefined;
}

function toolResultText(output: unknown): string {
  if (typeof output === 'string') {
    return output;
  }
  if (isRecord(output) && typeof output.value === 'string') {
    return output.value;
  }
  return JSON.stringify(output);
}

function toPiContext(
  messages: LlmMessage[],
  model: PiModel<'openai-completions'>,
  systemPrompt: string,
  tools: PiTool[] | undefined,
): PiContext {
  const piMessages: unknown[] = messages.flatMap((message): unknown[] => {
    switch (message.role) {
      case 'user':
        return [{ role: 'user', content: toPiUserContent(message.content), timestamp: 0 }];
      case 'assistant':
        return [
          {
            role: 'assistant',
            content: message.content.map((part) => {
              switch (part.type) {
                case 'text':
                  return { type: 'text' as const, text: part.text };
                case 'reasoning':
                  return {
                    type: 'thinking' as const,
                    thinking: part.text,
                    thinkingSignature:
                      storedReasoningSignature(part) ?? reasoningReplayField(model),
                  };
                case 'tool-call':
                  return {
                    type: 'toolCall' as const,
                    id: part.toolCallId,
                    name: part.toolName,
                    arguments: part.input as Record<string, unknown>,
                  };
              }
            }),
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: 'stop' as const,
            timestamp: 0,
          },
        ];
      case 'tool':
        return message.content.map((part) => ({
          role: 'toolResult' as const,
          toolCallId: part.toolCallId,
          toolName: part.toolName ?? '',
          content: [{ type: 'text' as const, text: toolResultText(part.output) }],
          isError: false,
          timestamp: 0,
        }));
    }
  });
  return {
    systemPrompt,
    messages: piMessages as PiContext['messages'],
    tools,
  };
}

function userContentSignature(content: string | UserContentPart[]): string {
  const piContent = toPiUserContent(content);
  if (typeof piContent === 'string') {
    return `string:${piContent}`;
  }
  return `parts:${JSON.stringify(
    piContent.map((part) =>
      part.type === 'text'
        ? part
        : { type: 'image_url', image_url: { url: `data:${part.mimeType};base64,${part.data}` } },
    ),
  )}`;
}

function payloadUserContentSignature(content: unknown): string | null {
  if (typeof content === 'string') {
    return `string:${content}`;
  }
  return Array.isArray(content) ? `parts:${JSON.stringify(content)}` : null;
}

function audioFormat(mimeType: string): string | null {
  switch (mimeType.toLowerCase()) {
    case 'audio/wav':
    case 'audio/x-wav':
    case 'audio/wave':
    case 'audio/vnd.wave':
      return 'wav';
    case 'audio/mpeg':
    case 'audio/mp3':
    case 'audio/mpeg3':
      return 'mp3';
    default:
      return null;
  }
}

/** Restores Zen's named-user and audio extensions after Pi builds the standard payload. */
function patchPayload(
  payload: unknown,
  messages: LlmMessage[],
  toolsWereDisabled: boolean,
): unknown {
  if (!isRecord(payload)) {
    return payload;
  }

  if (toolsWereDisabled) {
    delete payload.tools;
  }

  const namesByContent = new Map<string, Array<string | undefined>>();
  for (const message of messages) {
    if (message.role !== 'user') {
      continue;
    }
    const signature = userContentSignature(message.content);
    const names = namesByContent.get(signature) ?? [];
    names.push(message.name);
    namesByContent.set(signature, names);
  }

  if (!Array.isArray(payload.messages)) {
    return payload;
  }
  for (const message of payload.messages) {
    if (!isRecord(message) || message.role !== 'user') {
      continue;
    }

    const signature = payloadUserContentSignature(message.content);
    const names = signature === null ? undefined : namesByContent.get(signature);
    const name = names?.shift();
    if (name) {
      message.name = name;
    }

    if (!Array.isArray(message.content)) {
      continue;
    }
    message.content = message.content.map((part) => {
      if (!isRecord(part) || part.type !== 'image_url' || !isRecord(part.image_url)) {
        return part;
      }
      const url = part.image_url.url;
      if (typeof url !== 'string') {
        return part;
      }
      const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (!match || !match[1]!.toLowerCase().startsWith('audio/')) {
        return part;
      }
      const mimeType = match[1]!;
      const format = audioFormat(mimeType);
      return format === null
        ? { type: 'text', text: `[audio attached (${mimeType}) omitted: unsupported format]` }
        : { type: 'input_audio', input_audio: { data: match[2]!, format } };
    });
  }
  return payload;
}

function toLlmUsage(
  usage: PiAssistantMessage['usage'],
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  return buildLlmUsage(
    {
      // Pi separates cache reads/writes from normal input. Zen's historical
      // input count is the provider's full prompt token count.
      inputTokens: usage.input + usage.cacheRead + usage.cacheWrite,
      outputTokens: usage.output,
      totalTokens: usage.totalTokens,
      cacheReadTokens: usage.cacheRead,
      // Zen does not expose a cache-write bucket; count it as non-read input.
      cacheMissTokens: usage.input + usage.cacheWrite,
      reasoningTokens: usage.reasoning ?? 0,
    },
    timing,
  );
}

function mapFinishReason(reason: string): string {
  switch (reason) {
    case 'toolUse':
      return 'tool-calls';
    case 'length':
      return 'length';
    case 'stop':
      return 'stop';
    default:
      return 'other';
  }
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
      model: options.model,
      droppedAssistants: healed.droppedAssistants,
      droppedTools: healed.droppedTools,
    });
  }

  const model: PiModel<'openai-completions'> = {
    id: options.model,
    name: options.model,
    api: 'openai-completions',
    provider: options.provider,
    baseUrl: options.baseUrl,
    reasoning: true,
    // Zen already gates media against its model catalog. Keep Pi from
    // downgrading valid OpenAI-compatible image/audio payloads in transit.
    input: ['text', 'image'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_000_000,
    maxTokens: 384_000,
    compat: options.compat,
    thinkingLevelMap: options.thinkingLevelMap,
  };
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
        model: options.model,
        waitedMs: rateLimitedMs,
      });
    }

    timer = setTimeout(() => {
      timeoutCtrl.abort(new Error(`${options.label} request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    timer.unref?.();

    const attempts = options.retry?.maxAttempts ?? 4;
    const stream = streamOpenAiCompletions(model, context, {
      apiKey: options.apiKey,
      headers: options.extraHeaders,
      signal,
      sessionId: options.sessionId,
      reasoningEffort:
        options.thinkingEffort && options.thinkingEffort !== 'off'
          ? options.thinkingEffort
          : undefined,
      maxRetries: Math.max(0, attempts - 1),
      maxRetryDelayMs: options.retry?.maxBackoffMs,
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
