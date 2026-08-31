import type { Model } from '@earendil-works/pi-ai';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { runChatCompletions } from '../chat-completions.js';
import { resetChatRateLimit } from '../rate-limit.js';
import type { LlmMessage } from '../../session/storage.js';
import { testServerBaseUrl } from '../../test-server.js';

const originalEnv = { ...process.env };

/** Minimal pi model for adapter-level tests (mirrors registry-built models). */
function testModel(
  baseUrl: string,
  compat: Record<string, unknown> = {},
): Model<'openai-completions'> {
  return {
    id: 'test-model',
    name: 'Test Model',
    api: 'openai-completions',
    provider: 'test',
    baseUrl,
    reasoning: true,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: 384_000,
    compat: compat as Model<'openai-completions'>['compat'],
    thinkingLevelMap: {},
  };
}
let server: import('node:http').Server | undefined;

function startServer(
  handler: (
    req: import('node:http').IncomingMessage,
    res: import('node:http').ServerResponse,
  ) => void,
): Promise<number> {
  return new Promise((resolve) => {
    const srv = require('node:http').createServer(handler);
    server = srv;
    srv.listen(0, () => {
      const addr = srv.address() as import('node:net').AddressInfo;
      resolve(addr.port);
    });
  });
}

describe('runChatCompletions', () => {
  beforeEach(() => {
    process.env.ZEN_AGENT_CHAT_TIMEOUT_MS = '100';
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    resetChatRateLimit();
    server?.closeAllConnections?.();
    server?.close();
    server = undefined;
  });

  it('times out a hung request with a clear message', async () => {
    const port = await startServer(() => {
      /* never respond — the stream stalls on reader.read() */
    });
    await expect(
      runChatCompletions({
        model: testModel(testServerBaseUrl(server, port)),
        apiKey: 'test',
        label: 'Test',
        messages: [],
        system: 'system',
        thinkingEffort: 'off',
      }),
    ).rejects.toThrow('timed out after 100ms');
  });

  it('does not burn the chat timeout budget while queued behind the rate limit', async () => {
    // One request per 1200ms; the chat timeout is shorter than the wait and
    // the wait clears the >=1s rate-limit logging threshold.
    process.env.ZEN_AGENT_CHAT_TIMEOUT_MS = '500';
    process.env.ZEN_AGENT_CHAT_RPM = '50';
    resetChatRateLimit();
    const logRuntime = vi.fn();

    const port = await startServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      const chunk = JSON.stringify({
        choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
    });

    const options = {
      model: testModel(testServerBaseUrl(server, port)),
      apiKey: 'test',
      label: 'Test',
      messages: [],
      system: 'system',
      thinkingEffort: 'off' as const,
      logRuntime,
    };

    // First call consumes the immediate slot.
    await runChatCompletions(options);
    // Second call queues ~1000ms — longer than the 500ms timeout. Before the
    // fix the timeout timer was armed before the wait and this request died
    // with "timed out after 500ms" before ever being sent.
    const result = await runChatCompletions(options);
    expect(result.text).toBe('hi');
    // The long wait is recorded for debugging (log.jsonl via the agent).
    expect(logRuntime).toHaveBeenCalledWith(
      'info',
      'chat request delayed by client-side rate limit',
      expect.objectContaining({ label: 'Test', waitedMs: expect.any(Number) }),
    );
  });

  it('preserves streamed reasoning details and replays them on the next tool step', async () => {
    process.env.ZEN_AGENT_CHAT_TIMEOUT_MS = '2000';
    const requestBodies: Array<Record<string, unknown>> = [];
    const reasoningDetails = [
      {
        type: 'reasoning.text',
        text: 'I should inspect the directory.',
        signature: 'reasoning-signature',
        id: 'reasoning-1',
        format: 'anthropic-claude-v1',
        index: 0,
      },
      {
        type: 'reasoning.encrypted',
        data: 'opaque-reasoning-data',
        id: 'reasoning-1',
        format: 'anthropic-claude-v1',
        index: 1,
      },
    ];
    const port = await startServer((req, res) => {
      let raw = '';
      req.on('data', (chunk) => (raw += chunk));
      req.on('end', () => {
        requestBodies.push(JSON.parse(raw));
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const delta =
          requestBodies.length === 1
            ? {
                reasoning: 'I should inspect the directory.',
                reasoning_details: reasoningDetails,
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_1',
                    type: 'function',
                    function: { name: 'bash', arguments: '{"command":"pwd"}' },
                  },
                ],
              }
            : { content: 'done' };
        const finishReason = requestBodies.length === 1 ? 'tool_calls' : 'stop';
        const chunk = JSON.stringify({
          id: `completion-${requestBodies.length}`,
          model: 'test-model',
          choices: [{ index: 0, delta, finish_reason: finishReason }],
        });
        res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
      });
    });

    const baseOptions = {
      model: testModel(testServerBaseUrl(server, port), {
        supportsDeveloperRole: false,
        thinkingFormat: 'openrouter',
      }),
      apiKey: 'test',
      label: 'OpenRouter',
      system: 'system',
      thinkingEffort: 'high' as const,
    };
    const first = await runChatCompletions({
      ...baseOptions,
      messages: [{ role: 'user', content: 'Inspect this project.' }],
    });

    expect(first.toolCalls).toEqual([{ id: 'call_1', name: 'bash', input: { command: 'pwd' } }]);
    expect(first.reasoningSignature).toBe(JSON.stringify(reasoningDetails));

    type ReasoningPartWithSignature = {
      type: 'reasoning';
      text: string;
      reasoningSignature?: string;
    };
    const reasoningPart: ReasoningPartWithSignature = {
      type: 'reasoning',
      text: first.reasoning,
      reasoningSignature: first.reasoningSignature,
    };
    const replayMessages: LlmMessage[] = [
      { role: 'user', content: 'Inspect this project.' },
      {
        role: 'assistant',
        content: [
          reasoningPart,
          {
            type: 'tool-call',
            toolCallId: 'call_1',
            toolName: 'bash',
            input: { command: 'pwd' },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'call_1', toolName: 'bash', output: '/project' },
        ],
      },
    ];
    await runChatCompletions({ ...baseOptions, messages: replayMessages });

    expect(requestBodies).toHaveLength(2);
    const replayedAssistant = (requestBodies[1]?.messages as Array<Record<string, unknown>>).find(
      (message) => message.role === 'assistant',
    );
    expect(replayedAssistant?.reasoning_details).toStrictEqual(reasoningDetails);
    expect(replayedAssistant?.reasoning).toBeUndefined();
  });

  it('splits CRLF events whose terminator spans two network chunks', async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      const chunk = JSON.stringify({
        choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: 'stop' }],
      });
      // First write ends in the MIDDLE of the "\r\n\r\n" terminator.
      res.write(`data: ${chunk}\r\n\r`);
      setTimeout(() => {
        res.write(`\ndata: [DONE]\r\n\r\n`);
        res.end();
      }, 30);
    });

    const result = await runChatCompletions({
      model: testModel(testServerBaseUrl(server, port)),
      apiKey: 'test',
      label: 'Test',
      messages: [],
      system: 'system',
      thinkingEffort: 'off',
    });
    expect(result.text).toBe('hi');
  });
});

describe('runChatCompletions tools field', () => {
  beforeEach(() => {
    process.env.ZEN_AGENT_CHAT_TIMEOUT_MS = '2000';
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    resetChatRateLimit();
    server?.closeAllConnections?.();
    server?.close();
    server = undefined;
  });

  /** Starts a server that records the request body, then runs one request. */
  async function captureBody(tools: unknown[] | undefined): Promise<Record<string, unknown>> {
    let resolveBody!: (body: Record<string, unknown>) => void;
    const bodyPromise = new Promise<Record<string, unknown>>((resolve) => {
      resolveBody = resolve;
    });
    const port = await startServer((req, res) => {
      let raw = '';
      req.on('data', (chunk) => (raw += chunk));
      req.on('end', () => {
        resolveBody(JSON.parse(raw));
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const chunk = JSON.stringify({
          choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: 'stop' }],
        });
        res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
      });
    });
    await runChatCompletions({
      model: testModel(testServerBaseUrl(server, port)),
      apiKey: 'test',
      label: 'Test',
      messages: [],
      system: 'system',
      thinkingEffort: 'off',
      tools,
    });
    return bodyPromise;
  }

  it('defaults to the bash tool schema when tools are not specified', async () => {
    const body = await captureBody(undefined);
    expect((body.tools as unknown[]).length).toBe(1);
  });

  it('omits the tools field entirely when the list is empty (/tools off)', async () => {
    const body = await captureBody([]);
    expect(body.tools).toBeUndefined();
  });
});
