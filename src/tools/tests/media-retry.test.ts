import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('../../providers/index.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return {
    ...actual,
    runLlmStep: vi.fn(),
    // Controlled per test; null simulates "lookup failed / catalog unknown".
    getModelModalities: vi.fn(),
  };
});

import { ZenAgent } from '../../agent/index.js';
import { runLlmStep, getModelModalities, type LlmStepResult } from '../../providers/index.js';
import { BASH_TOOL_SCHEMA, READ_MEDIA_TOOL_SCHEMA } from '../../providers/llm-client.js';
import { clientLogPath } from '../../session/storage.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);
const mockedGetModelModalities = vi.mocked(getModelModalities);

function textStep(text: string): LlmStepResult {
  return { text, reasoning: '', toolCalls: [], finishReason: 'stop', usage: null };
}

function makeCx() {
  return {
    notify: vi.fn(async () => {}),
    request: vi.fn(() => Promise.reject(new Error('unexpected client request'))),
  } as unknown as acp.AgentContext;
}

describe('model modality lookup retries until definitive', () => {
  let cwd: string;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-media-retry-'));
    process.env.DEEPSEEK_API_KEY = 'test-key';
    mockedRunLlmStep.mockReset();
    mockedGetModelModalities.mockReset();
  });

  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it('retries an unknown lookup instead of caching a text-only answer for the session', async () => {
    // First lookup fails (e.g. offline start); the catalog then recovers.
    mockedGetModelModalities
      .mockResolvedValueOnce(null)
      .mockResolvedValue({ image: true, audio: false });

    const agent = new ZenAgent();
    await agent.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: {},
    } as acp.InitializeRequest);
    const cx = makeCx();
    const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
    const sessionId = created.sessionId;

    const toolsPerStep: Array<unknown[]> = [];
    mockedRunLlmStep.mockImplementation(async (_provider, options) => {
      toolsPerStep.push(options.tools ?? []);
      return textStep(`answer ${toolsPerStep.length}`);
    });

    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'a' }] }, cx);
    // Unknown at this point: conservative bash-only tool list for the request…
    expect(toolsPerStep[0]).toEqual([BASH_TOOL_SCHEMA]);

    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'b' }] }, cx);
    // …but NOT memoized: once the lookup succeeds, read_media is offered.
    expect(toolsPerStep[1]).toEqual([BASH_TOOL_SCHEMA, READ_MEDIA_TOOL_SCHEMA]);

    // The transient unknown is visible in log.jsonl (warn + recovery info).
    const startupKey = (agent as unknown as { startupLogKey: string }).startupLogKey;
    const logFile = clientLogPath(cwd, startupKey);
    await vi.waitFor(async () => {
      const raw = await readFile(logFile, 'utf8');
      expect(raw).toContain('model input modalities unknown');
      expect(raw).toContain('model input modalities resolved');
    });
  });
});
