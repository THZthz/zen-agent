import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { formatLlmError } from './llm-errors.js';
import { testServerBaseUrl } from './test-server.js';

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

function apiError(status: number, body: string, label = 'DeepSeek'): Error {
  return new Error(`${label} API error ${status}: ${body}`);
}

describe('formatLlmError', () => {
  beforeEach(() => {
    process.env.DEEPSEEK_API_KEY = 'test';
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
  });

  it('returns non-API errors unchanged', async () => {
    await expect(formatLlmError(new Error('boom'), { provider: 'deepseek' })).resolves.toBe('boom');
    await expect(formatLlmError('not an error', { provider: 'deepseek' })).resolves.toBe(
      'not an error',
    );
  });

  it('points at the right key env for 401', async () => {
    const ds = await formatLlmError(apiError(401, 'unauthorized'), { provider: 'deepseek' });
    expect(ds).toContain('DEEPSEEK_API_KEY');
    const or = await formatLlmError(apiError(401, 'unauthorized', 'OpenRouter'), {
      provider: 'openrouter',
    });
    expect(or).toContain('OPENROUTER_API_KEY');
  });

  it('tells DeepSeek users to top up on 402', async () => {
    const msg = await formatLlmError(apiError(402, 'insufficient balance'), {
      provider: 'deepseek',
    });
    expect(msg).toContain('balance');
  });

  it('explains context overflow with the requested token count', async () => {
    const msg = await formatLlmError(
      apiError(
        400,
        '{"error":{"message":"This model\'s maximum context length is 1000000 tokens. However, you requested 1234567 tokens"}}',
      ),
      { provider: 'deepseek' },
    );
    expect(msg).toContain('Context window exceeded');
    expect(msg).toContain('1234567');
  });

  it('extracts the inner message for plain 400/422', async () => {
    const bad = await formatLlmError(apiError(400, '{"error":{"message":"Invalid format"}}'), {
      provider: 'deepseek',
    });
    expect(bad).toBe('Bad request (400): Invalid format');
    const unprocessable = await formatLlmError(
      apiError(422, '{"error":{"message":"Bad parameters"}}'),
      { provider: 'deepseek' },
    );
    expect(unprocessable).toBe('Request rejected (422): Bad parameters');
  });

  it('mentions the automatic retry on 429', async () => {
    const msg = await formatLlmError(apiError(429, 'rate limit'), { provider: 'deepseek' });
    expect(msg).toContain('Rate limit');
    expect(msg).toContain('retried automatically');
  });

  it('distinguishes reachable vs unreachable DeepSeek on 5xx via the balance probe', async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          is_available: true,
          balance_infos: [{ currency: 'CNY', total_balance: '110.00' }],
        }),
      );
    });
    process.env.DEEPSEEK_BASE_URL = testServerBaseUrl(server, port);

    const reachable = await formatLlmError(apiError(503, 'service unavailable'), {
      provider: 'deepseek',
    });
    expect(reachable).toContain('reachable');
    expect(reachable).toContain('503');
  });

  it('reports DeepSeek as unreachable when the probe fails', async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(500);
      res.end('boom');
    });
    process.env.DEEPSEEK_BASE_URL = testServerBaseUrl(server, port);

    const msg = await formatLlmError(apiError(503, 'service unavailable'), {
      provider: 'deepseek',
    });
    expect(msg).toContain('Cannot reach DeepSeek');
  });

  it('uses generic wording for OpenRouter 5xx', async () => {
    const msg = await formatLlmError(apiError(500, 'boom', 'OpenRouter'), {
      provider: 'openrouter',
    });
    expect(msg).toContain('OpenRouter returned 500');
  });

  it('leaves unclassified statuses unchanged', async () => {
    const msg = await formatLlmError(apiError(403, 'forbidden'), { provider: 'deepseek' });
    expect(msg).toBe('DeepSeek API error 403: forbidden');
  });
});
