import { afterEach, describe, expect, it, vi } from 'vitest';
import type { LlmMessage } from './storage.js';
import { healMessages } from './heal.js';

// Fixture builders — content arrays are cast because the AI SDK part types
// require fields that corrupted sessions legitimately lack.
function assistant(parts: unknown[]): LlmMessage {
  return { role: 'assistant', content: parts as never };
}

function textPart(text: string): unknown {
  return { type: 'text', text };
}

function toolCallPart(id: string | undefined, name = 'bash'): unknown {
  return { type: 'tool-call', toolCallId: id, toolName: name, input: { command: 'ls' } };
}

function toolResultPart(id: string): unknown {
  return {
    type: 'tool-result',
    toolCallId: id,
    toolName: 'bash',
    output: { type: 'text', value: 'ok' },
  };
}

function toolMessage(results: unknown[]): LlmMessage {
  return { role: 'tool', content: results as never };
}

function userMessage(text: string): LlmMessage {
  return { role: 'user', content: text };
}

describe('healMessages', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('passes a healthy history through unchanged', () => {
    const messages: LlmMessage[] = [
      userMessage('hello'),
      assistant([textPart('sure'), toolCallPart('call-1')]),
      toolMessage([toolResultPart('call-1')]),
      assistant([textPart('done')]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(0);
    expect(result.droppedTools).toBe(0);
    expect(result.messages).toEqual(messages);
  });

  it('drops an assistant whose tool calls have no results', () => {
    const messages: LlmMessage[] = [
      userMessage('hello'),
      assistant([toolCallPart('call-1')]),
      assistant([textPart('next turn')]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(1);
    expect(result.droppedTools).toBe(0);
    expect(result.messages).toEqual([userMessage('hello'), assistant([textPart('next turn')])]);
  });

  it('drops an assistant when the tool message only covers some calls', () => {
    const messages: LlmMessage[] = [
      assistant([toolCallPart('call-1'), toolCallPart('call-2')]),
      toolMessage([toolResultPart('call-1')]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(1);
    expect(result.droppedTools).toBe(1);
    expect(result.messages).toEqual([]);
  });

  it('drops stray tool messages that follow a healthy group', () => {
    const messages: LlmMessage[] = [
      assistant([toolCallPart('call-1')]),
      toolMessage([toolResultPart('call-1')]),
      toolMessage([toolResultPart('call-orphan')]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(0);
    expect(result.droppedTools).toBe(1);
    expect(result.messages).toEqual([
      assistant([toolCallPart('call-1')]),
      toolMessage([toolResultPart('call-1')]),
    ]);
  });

  it('treats a tool message with a stray result as unpaired', () => {
    const messages: LlmMessage[] = [
      assistant([toolCallPart('call-1')]),
      toolMessage([toolResultPart('call-1'), toolResultPart('call-stray')]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(1);
    expect(result.droppedTools).toBe(1);
    expect(result.messages).toEqual([]);
  });

  it('stamps a missing tool-call id so the pair can match', () => {
    vi.setSystemTime(1_700_000_000_000);
    const messages: LlmMessage[] = [
      assistant([toolCallPart(undefined)]),
      toolMessage([toolResultPart(`zen-1700000000000-0`)]),
    ];
    const result = healMessages(messages);
    expect(result.droppedAssistants).toBe(0);
    expect(result.droppedTools).toBe(0);
    const kept = result.messages[0]! as { content: Array<{ type: string; toolCallId: string }> };
    expect(kept.content[0]!.toolCallId).toBe('zen-1700000000000-0');
  });

  it('does not mutate the input messages', () => {
    const messages: LlmMessage[] = [assistant([toolCallPart(undefined)])];
    const original = JSON.stringify(messages);
    healMessages(messages);
    expect(JSON.stringify(messages)).toBe(original);
  });
});
