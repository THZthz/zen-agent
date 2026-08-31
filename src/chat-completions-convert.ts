import {
  type AssistantMessage as PiAssistantMessage,
  type Context as PiContext,
  type Model as PiModel,
  type Tool as PiTool,
} from '@earendil-works/pi-ai';
import { BASH_TOOL_SCHEMA, buildLlmUsage, type LlmUsage } from './llm-client.js';
import { isRecord } from './is-record.js';
import type { LlmMessage, UserContentPart } from './storage.js';

/**
 * Pure wire-conversion layer for the OpenAI-compatible chat-completions
 * adapter (see the ownership map in agent.ts): Zen's persisted message shape
 * <-> Pi's chat context, plus the payload patch-ups Pi does not model
 * (named-user messages, OpenAI `input_audio`). No I/O and no request state —
 * everything here is exported for direct unit testing; the streaming loop
 * that consumes it lives in chat-completions.ts.
 */

export function toolFromSchema(schema: unknown): PiTool {
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

export function toPiTools(tools: unknown[] | undefined): PiTool[] | undefined {
  if (tools !== undefined && tools.length === 0) {
    return undefined;
  }
  return (tools ?? [BASH_TOOL_SCHEMA]).map(toolFromSchema);
}

/** Pi's public chat context has text/image blocks; model audio stays OpenAI-compatible at payload time. */
export function toPiUserContent(
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

export function reasoningReplayField(model: PiModel<'openai-completions'>): string | undefined {
  if (model.compat?.requiresReasoningContentOnAssistantMessages) {
    return 'reasoning_content';
  }
  return model.compat?.thinkingFormat === 'openrouter' ? 'reasoning' : undefined;
}

export type StoredReasoningPart = { type: 'reasoning'; text: string };
type StoredReasoningPartWithSignature = StoredReasoningPart & {
  reasoningSignature?: unknown;
  thinkingSignature?: unknown;
  signature?: unknown;
};

/** Read forward-compatible replay metadata without widening storage's current type. */
export function storedReasoningSignature(part: StoredReasoningPart): string | undefined {
  const signed = part as StoredReasoningPartWithSignature;
  for (const candidate of [signed.reasoningSignature, signed.thinkingSignature, signed.signature]) {
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }
  return undefined;
}

export function toolResultText(output: unknown): string {
  if (typeof output === 'string') {
    return output;
  }
  if (isRecord(output) && typeof output.value === 'string') {
    return output.value;
  }
  return JSON.stringify(output);
}

export function toPiContext(
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

export function userContentSignature(content: string | UserContentPart[]): string {
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

export function payloadUserContentSignature(content: unknown): string | null {
  if (typeof content === 'string') {
    return `string:${content}`;
  }
  return Array.isArray(content) ? `parts:${JSON.stringify(content)}` : null;
}

export function audioFormat(mimeType: string): string | null {
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
export function patchPayload(
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

export function toLlmUsage(
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

export function mapFinishReason(reason: string): string {
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
