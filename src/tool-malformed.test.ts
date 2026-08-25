import { describe, expect, it } from 'vitest';
import * as acp from '@agentclientprotocol/sdk';
import { executeLlmToolCall, type ToolExecutorContext } from './tool-execution.js';
import type { StoredSession } from './storage.js';

function makeContext(overrides: Partial<ToolExecutorContext> = {}): ToolExecutorContext {
  const session = {
    sessionId: 'sess_x',
    cwd: '/tmp',
    createdAt: '',
    updatedAt: '',
    title: null,
    events: [],
    llmMessages: [],
    config: {
      provider: 'deepseek',
      model: 'm',
      thinkingEffort: 'off',
      systemPrompt: '',
      sandbox: false,
    },
    usage: {},
    turnStats: [],
  } as unknown as StoredSession;
  return {
    session,
    mediaModalities: { image: true, audio: true },
    sandbox: false,
    clientCapabilities: { terminal: true },
    emit: async () => {},
    logRuntime: async () => {},
    ...overrides,
  };
}

describe('malformed streamed tool-call arguments', () => {
  it('reports malformed JSON arguments instead of a per-tool type error', async () => {
    const updates: acp.SessionUpdate[] = [];
    const logged: Array<{ level: string; message: string }> = [];
    const context = makeContext({
      emit: async (update) => {
        updates.push(update);
      },
      logRuntime: async (level, message) => {
        logged.push({ level, message });
      },
    });

    const result = await executeLlmToolCall(
      context,
      {} as unknown as acp.AgentContext,
      { id: 'c9', name: 'read_media', input: { malformed_arguments: '{"path": "shot.pn' } },
      new AbortController().signal,
    );

    expect(result.toolName).toBe('read_media');
    expect(result.output.value).toContain('malformed JSON arguments');
    expect(result.output.value).toContain('read_media');
    const finalUpdate = updates.at(-1) as { status?: string };
    expect(finalUpdate?.status).toBe('failed');
    expect(logged[0]?.message).toBe('tool call had malformed JSON arguments');
  });
});
