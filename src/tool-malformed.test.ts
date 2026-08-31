import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
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
      toolsEnabled: true,
    },
    usage: {},
    turnStats: [],
  } as unknown as StoredSession;
  return {
    session,
    mediaModalities: { image: true, audio: true },
    sandbox: false,
    toolsEnabled: true,
    clientCapabilities: { terminal: true },
    emit: async () => {},
    logRuntime: async () => {},
    ...overrides,
  };
}

describe('malformed streamed tool-call arguments', () => {
  it.each(['bash', 'read_media'] as const)(
    'returns a failed %s result for null JSON arguments',
    async (name) => {
      const updates: acp.SessionUpdate[] = [];
      const result = await executeLlmToolCall(
        makeContext({
          emit: async (update) => {
            updates.push(update);
          },
        }),
        {} as unknown as acp.AgentContext,
        { id: 'null-input', name, input: null },
        new AbortController().signal,
      );

      expect(result.output.value).toMatch(/requires a non-empty string/);
      expect((updates.at(-1) as { status?: string })?.status).toBe('failed');
    },
  );

  it('kills a terminal when abort races terminal creation', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-tool-abort-'));
    try {
      let finishCreate!: (value: { terminalId: string }) => void;
      const create = new Promise<{ terminalId: string }>((resolve) => {
        finishCreate = resolve;
      });
      const request = vi.fn((method: string) => {
        if (method === acp.methods.client.terminal.create) return create;
        if (method === acp.methods.client.terminal.kill) return Promise.resolve({});
        if (method === acp.methods.client.terminal.release) return Promise.resolve({});
        return Promise.reject(new Error(`unexpected client request: ${method}`));
      });
      const session = { ...makeContext().session, cwd };
      const controller = new AbortController();
      const execution = executeLlmToolCall(
        makeContext({ session }),
        { request } as unknown as acp.AgentContext,
        { id: 'abort-race', name: 'bash', input: { command: 'sleep 60' } },
        controller.signal,
      );

      await vi.waitFor(() =>
        expect(request).toHaveBeenCalledWith(acp.methods.client.terminal.create, expect.anything()),
      );
      controller.abort(new Error('cancelled'));
      finishCreate({ terminalId: 'terminal-race' });

      const result = await execution;
      expect(result.output.value).toContain('cancelled');
      expect(request).toHaveBeenCalledWith(acp.methods.client.terminal.kill, {
        sessionId: session.sessionId,
        terminalId: 'terminal-race',
      });
      expect(request).not.toHaveBeenCalledWith(
        acp.methods.client.terminal.waitForExit,
        expect.anything(),
      );
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

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
