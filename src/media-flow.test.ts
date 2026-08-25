import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('./provider.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return {
    ...actual,
    runLlmStep: vi.fn(),
    // Deterministic modality answers; overridden per test via mockedModalities.
    getModelModalities: vi.fn(async () => mockedModalities),
  };
});

import { ZenAgent } from './agent.js';
import { runLlmStep, type LlmStepResult } from './provider.js';
import { READ_MEDIA_TOOL_SCHEMA } from './llm-client.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47]).toString('base64');

let mockedModalities: { image: boolean; audio: boolean } = { image: false, audio: false };

/**
 * The real runLlmStep receives the LIVE llmMessages array (the turn keeps
 * mutating it afterwards), so each step snapshots the request at call time.
 */
const recordedSteps: Array<{ tools?: unknown[]; messages: unknown[] }> = [];

function recordStep(result: LlmStepResult): void {
  mockedRunLlmStep.mockImplementationOnce(async (_provider, options) => {
    recordedSteps.push({
      tools: options.tools,
      messages: JSON.parse(JSON.stringify(options.messages)),
    });
    return result;
  });
}

type RecordedMessage = {
  role: string;
  content: string | Array<{ type: string; [k: string]: unknown }>;
};

function makeCx() {
  const notifications: Array<{ sessionId: string; update: acp.SessionUpdate }> = [];
  const cx = {
    request: vi.fn((method: string) => {
      switch (method) {
        case acp.methods.client.terminal.create:
          return Promise.resolve({ terminalId: 't1' });
        case acp.methods.client.terminal.waitForExit:
          return Promise.resolve({ exitCode: 0, signal: null });
        case acp.methods.client.terminal.output:
          return Promise.resolve({ output: 'done', truncated: false });
        case acp.methods.client.terminal.release:
          return Promise.resolve({});
        default:
          return Promise.reject(new Error(`unexpected client request: ${method}`));
      }
    }),
    notify: vi.fn(
      async (method: string, params: { sessionId: string; update: acp.SessionUpdate }) => {
        if (method === acp.methods.client.session.update) notifications.push(params);
      },
    ),
  } as unknown as acp.AgentContext;
  return { cx, notifications };
}

async function newSession(agent: ZenAgent, cwd: string): Promise<string> {
  const response = await agent.newSession(
    { cwd, mcpServers: [] } as acp.NewSessionRequest,
    makeCx().cx,
  );
  return response.sessionId;
}

function textStep(text: string): LlmStepResult {
  return {
    text,
    reasoning: '',
    toolCalls: [],
    finishReason: 'stop',
    usage: null,
  };
}

describe('media prompt flow', () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-media-flow-'));
    process.env.DEEPSEEK_API_KEY = 'test-key';
    mockedRunLlmStep.mockReset();
    recordedSteps.length = 0;
    mockedModalities = { image: false, audio: false };
  });

  it('declares image/audio prompt capabilities in initialize', async () => {
    const agent = new ZenAgent();
    const response = await agent.initialize({
      protocolVersion: acp.PROTOCOL_VERSION,
    } as acp.InitializeRequest);
    expect(response.agentCapabilities?.promptCapabilities).toEqual({
      image: true,
      audio: true,
    });
  });

  it('degrades attached images to placeholder text on text-only models and omits read_media', async () => {
    recordStep(textStep('ok'));
    const agent = new ZenAgent();
    const sessionId = await newSession(agent, cwd);
    const { cx } = makeCx();

    await agent.prompt(
      {
        sessionId,
        prompt: [
          { type: 'text', text: 'what is this?' },
          { type: 'image', data: PNG, mimeType: 'image/png' },
        ],
      },
      cx,
    );

    const options = recordedSteps[0]!;
    expect(options.tools).toEqual([
      expect.objectContaining({
        function: expect.objectContaining({ name: 'bash' }),
      }),
    ]);
    const last = options.messages.at(-1) as RecordedMessage;
    expect(last.role).toBe('user');
    expect(Array.isArray(last.content)).toBe(true);
    const parts = last.content as Array<{ type: string; text?: string }>;
    expect(parts.some((part) => part.type === 'image')).toBe(false);
    expect(
      parts.some(
        (part) => part.type === 'text' && part.text!.includes('does not support image input'),
      ),
    ).toBe(true);
  });

  it('forwards supported images as parts and offers read_media on vision models', async () => {
    mockedModalities = { image: true, audio: false };
    recordStep(textStep('a cat'));
    const agent = new ZenAgent();
    const sessionId = await newSession(agent, cwd);

    await agent.prompt(
      {
        sessionId,
        prompt: [
          { type: 'text', text: 'what is this?' },
          { type: 'image', data: PNG, mimeType: 'image/png' },
        ],
      },
      makeCx().cx,
    );

    const options = recordedSteps[0]!;
    expect(
      options.tools?.map((tool) => (tool as typeof READ_MEDIA_TOOL_SCHEMA).function.name),
    ).toEqual(['bash', 'read_media']);
    const last = options.messages.at(-1) as RecordedMessage;
    const parts = last.content as Array<{ type: string; mimeType?: string }>;
    expect(parts).toEqual([
      { type: 'text', text: 'what is this?' },
      { type: 'image', mimeType: 'image/png', data: PNG },
    ]);
  });

  it('read_media tool call injects the payload as a synthetic user message', async () => {
    mockedModalities = { image: true, audio: false };
    writeFileSync(join(cwd, 'shot.png'), Buffer.from(PNG, 'base64'));
    recordStep({
      text: '',
      reasoning: '',
      toolCalls: [{ id: 'm1', name: 'read_media', input: { path: 'shot.png' } }],
      finishReason: 'tool-calls',
      usage: null,
    });
    recordStep(textStep('it is a tiny png'));

    const agent = new ZenAgent();
    const sessionId = await newSession(agent, cwd);
    const response = await agent.prompt(
      { sessionId, prompt: [{ type: 'text', text: 'look at shot.png' }] },
      makeCx().cx,
    );
    expect(response.stopReason).toBe('end_turn');

    const secondCallOptions = recordedSteps[1]!;
    const messages = secondCallOptions.messages as RecordedMessage[];
    // assistant(tool-call), tool(result), synthetic user with the payload.
    const toolMessage = messages.findLast((message) => message.role === 'tool');
    expect(toolMessage).toBeTruthy();
    const syntheticUser = messages.at(-1)!;
    expect(syntheticUser.role).toBe('user');
    const content = syntheticUser.content as Array<{ type: string; [k: string]: unknown }>;
    expect(content.some((part) => part.type === 'image' && part.data === PNG)).toBe(true);
    expect(content[0]).toMatchObject({ type: 'text' });
    expect(content[0].text).toContain('shot.png');
  });

  it('read_media failures produce a failed tool result without media injection', async () => {
    mockedModalities = { image: true, audio: false };
    recordStep({
      text: '',
      reasoning: '',
      toolCalls: [{ id: 'm2', name: 'read_media', input: { path: 'nope.png' } }],
      finishReason: 'tool-calls',
      usage: null,
    });
    recordStep(textStep('not found'));

    const agent = new ZenAgent();
    const sessionId = await newSession(agent, cwd);
    await agent.prompt(
      { sessionId, prompt: [{ type: 'text', text: 'look at nope.png' }] },
      makeCx().cx,
    );

    const secondCallOptions = recordedSteps[1]!;
    const last = (secondCallOptions.messages as RecordedMessage[]).at(-1)!;
    expect(Array.isArray(last.content) === false || (last.content as unknown[]).length > 0).toBe(
      true,
    );
    const roles = (secondCallOptions.messages as RecordedMessage[]).map((message) => message.role);
    expect(roles.filter((role) => role === 'tool').length).toBeGreaterThan(0);
    const syntheticUser = (secondCallOptions.messages as RecordedMessage[]).at(-1)!;
    if (syntheticUser.role === 'user') {
      const content = syntheticUser.content as Array<{ type: string }>;
      expect(content.some((part) => part.type === 'image')).toBe(false);
    }
  });
});
