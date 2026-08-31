import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync } from 'node:fs';

import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('../../providers/index.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from '../index.js';
import { runLlmStep, type LlmStepResult, type LlmStepOptions } from '../../providers/index.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);

function textStep(text: string): LlmStepResult {
  return { text, reasoning: '', toolCalls: [], finishReason: 'stop', usage: null };
}

function makeCx() {
  return {
    notify: vi.fn(async () => {}),
    request: vi.fn(() => Promise.reject(new Error('unexpected client request'))),
  } as unknown as acp.AgentContext;
}

describe('concurrent prompts are serialized per session', () => {
  let cwd: string;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-concurrency-'));
    process.env.DEEPSEEK_API_KEY = 'test-key';
    mockedRunLlmStep.mockReset();
  });

  it('waits for the aborted turn to settle before running the new prompt', async () => {
    const agent = new ZenAgent();
    await agent.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: {},
    } as acp.InitializeRequest);
    const cx = makeCx();
    const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
    const sessionId = created.sessionId;
    const active = (
      agent as unknown as {
        sessions: Map<
          string,
          { turnPromise: Promise<unknown> | null; abortController: AbortController | null }
        >;
      }
    ).sessions.get(sessionId)!;

    // Turn A blocks in its LLM step until hard-aborted (like a real stream).
    mockedRunLlmStep.mockImplementationOnce(async (_provider, options: LlmStepOptions) => {
      const signal = options.signal!;
      if (!signal.aborted) {
        await new Promise<void>((resolve) => {
          signal.addEventListener('abort', () => resolve(), { once: true });
        });
      }
      throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    });
    // Turn B runs normally once it gets the floor.
    mockedRunLlmStep.mockImplementationOnce(async (_provider, _options: LlmStepOptions) => {
      // B's step must only start after A fully settled (its turn promise
      // cleared). A's own entry is gone from the history by now.
      expect(active.turnPromise).not.toBeNull();
      return textStep('second answer');
    });

    const promptA = agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'first' }] }, cx);
    await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalledTimes(1));

    const promptB = agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'second' }] }, cx);

    const [responseA, responseB] = await Promise.all([
      promptA.catch((error: Error) => ({ error })),
      promptB,
    ]);

    // A was cancelled (its controller was aborted by B's arrival)…
    expect(responseA).toEqual({ stopReason: 'cancelled' });
    // …and B completed as a clean turn.
    expect(responseB.stopReason).toBe('end_turn');

    // History is strictly ordered: A never interleaved its (failed) step
    // after B's user message.
    const session = (
      agent as unknown as {
        sessions: Map<
          string,
          { session: { llmMessages: Array<{ role: string; content: unknown }> } }
        >;
      }
    ).sessions.get(sessionId)!.session;
    const texts = session.llmMessages.map((m) =>
      typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
    );
    expect(texts.filter((t) => t === 'first')).toHaveLength(1);
    expect(texts.indexOf('first')).toBeLessThan(texts.indexOf('second'));
    expect(texts.at(-1)).toContain('second answer');

    // The old prompt's cleanup did not clobber the new state.
    expect(active.abortController).toBeNull();
  });

  it('serializes prompts that arrive together while the session is idle', async () => {
    const agent = new ZenAgent();
    await agent.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: {},
    } as acp.InitializeRequest);
    const cx = makeCx();
    const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);

    let releaseFirst!: () => void;
    const firstStep = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    mockedRunLlmStep
      .mockImplementationOnce(async () => {
        await firstStep;
        return textStep('first answer');
      })
      .mockResolvedValueOnce(textStep('second answer'));

    const promptA = agent.prompt(
      { sessionId: created.sessionId, prompt: [{ type: 'text', text: 'first' }] },
      cx,
    );
    const promptB = agent.prompt(
      { sessionId: created.sessionId, prompt: [{ type: 'text', text: 'second' }] },
      cx,
    );

    await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(mockedRunLlmStep).toHaveBeenCalledTimes(1);

    releaseFirst();
    const [responseA, responseB] = await Promise.all([promptA, promptB]);
    expect(responseA.stopReason).toBe('end_turn');
    expect(responseB.stopReason).toBe('end_turn');
    expect(mockedRunLlmStep).toHaveBeenCalledTimes(2);

    const secondMessages = mockedRunLlmStep.mock.calls[1]?.[1].messages ?? [];
    expect(JSON.stringify(secondMessages)).toContain('first answer');
  });
});
