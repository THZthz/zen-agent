import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('../../providers/index.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from '../../agent/index.js';
import { runLlmStep, type LlmStepResult } from '../../providers/index.js';
import { coalesceReplayEvents, prepareReplayEvents } from '../replay.js';
import { sessionPath } from '../storage.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);

function bashThenAnswer(): LlmStepResult[] {
  return [
    {
      text: '',
      reasoning: '',
      toolCalls: [{ id: 'c1', name: 'bash', input: { command: 'echo hi' } }],
      finishReason: 'tool-calls',
      usage: null,
    },
    { text: 'done', reasoning: '', toolCalls: [], finishReason: 'stop', usage: null },
  ];
}

describe('transcript does not duplicate terminal output', () => {
  let cwd: string;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-transcript-dedupe-'));
    process.env.DEEPSEEK_API_KEY = 'test-key';
    mockedRunLlmStep.mockReset();
  });

  it('keeps _meta.terminal_output on the wire but derives it from rawOutput on replay', async () => {
    const agent = new ZenAgent();
    await agent.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: { terminal: true },
    } as acp.InitializeRequest);
    const request = vi.fn((method: string) => {
      switch (method) {
        case acp.methods.client.terminal.create:
          return Promise.resolve({ terminalId: 't1' });
        case acp.methods.client.terminal.waitForExit:
          return Promise.resolve({ exitCode: 0, signal: null });
        case acp.methods.client.terminal.output:
          return Promise.resolve({ output: 'the full command output', truncated: false });
        case acp.methods.client.terminal.release:
          return Promise.resolve({});
        default:
          return Promise.reject(new Error(`unexpected client request: ${method}`));
      }
    });
    const notifications: Array<{ sessionId: string; update: acp.SessionUpdate }> = [];
    const cx = {
      request,
      notify: vi.fn(async (_method: string, params: (typeof notifications)[number]) => {
        notifications.push(params);
      }),
    } as unknown as acp.AgentContext;

    const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
    for (const step of bashThenAnswer()) {
      mockedRunLlmStep.mockResolvedValueOnce(step);
    }
    await agent.prompt(
      { sessionId: created.sessionId, prompt: [{ type: 'text', text: 'run' }] },
      cx,
    );

    // Live wire: the final update still carries _meta.terminal_output for Zed.
    const wireUpdate = notifications
      .map((n) => n.update)
      .filter(
        (u) =>
          u.sessionUpdate === 'tool_call_update' &&
          ((u as { status?: string }).status === 'completed' ||
            (u as { status?: string }).status === 'failed'),
      )
      .at(-1) as { _meta?: Record<string, unknown> };
    expect(wireUpdate._meta?.terminal_output).toBeDefined();

    // Persisted transcript: no duplicated payload, but rawOutput.output kept.
    const stored = JSON.parse(await readFile(sessionPath(cwd, created.sessionId), 'utf8')) as {
      events: Array<{
        sessionUpdate: string;
        status?: string;
        rawOutput?: unknown;
        _meta?: unknown;
      }>;
    };
    const storedUpdate = stored.events
      .filter((e) => e.sessionUpdate === 'tool_call_update' && e.status === 'completed')
      .at(-1)!;
    expect(storedUpdate._meta).toBeUndefined();
    expect((storedUpdate.rawOutput as { output?: string }).output).toBe('the full command output');

    // Replay reconstructs the display-only terminal streaming byte-identically
    // from rawOutput.output — the legacy-session derivation path.
    const replayed = coalesceReplayEvents(
      prepareReplayEvents(stored.events as acp.SessionUpdate[], cwd),
    );
    const replayedFinal = replayed
      .filter((e) => e.sessionUpdate === 'tool_call_update')
      .map((e) => e as unknown as { _meta?: { terminal_output?: { data?: string } } })
      .find((e) => e._meta?.terminal_output);
    expect(replayedFinal?._meta?.terminal_output?.data).toBe('the full command output');
  });
});
