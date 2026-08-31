import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { runChatCompletions } from './chat-completions.js';
import { resetChatRateLimit } from './rate-limit.js';

const originalEnv = { ...process.env };
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
        baseUrl: `http://127.0.0.1:${port}`,
        apiKey: 'test',
        provider: 'test',
        label: 'Test',
        model: 'test-model',
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
      baseUrl: `http://127.0.0.1:${port}`,
      apiKey: 'test',
      provider: 'test',
      label: 'Test',
      model: 'test-model',
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
      baseUrl: `http://127.0.0.1:${port}`,
      apiKey: 'test',
      provider: 'test',
      label: 'Test',
      model: 'test-model',
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
      baseUrl: `http://127.0.0.1:${port}`,
      apiKey: 'test',
      provider: 'test',
      label: 'Test',
      model: 'test-model',
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
