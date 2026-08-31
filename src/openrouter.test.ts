import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getModelPricing,
  runLlmStep,
} from './provider.js';
import { resetPiModels } from './provider-pi.js';
import { resetCatalogCache } from './provider-catalog.js';
import { resetProviderRegistry } from './provider-registry.js';
import { testServerBaseUrl } from './test-server.js';

/**
 * OpenRouter wire-level tests through the provider facade. The registry +
 * pi Models collection are rebuilt per test (env carries the fake base URL);
 * every SSE server must answer the discovery GET /models first.
 */

const originalEnv = { ...process.env };
let xdgHome: string;
let server: import('node:http').Server | undefined;

function serveCatalog(res: import('node:http').ServerResponse, entries: unknown[]): void {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ data: entries }));
}

const makeChunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
  `data: ${JSON.stringify({
    id: '1',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'or-model',
    choices: [{ index: 0, delta, finish_reason: null }],
    ...extra,
  })}\n\n`;

const reasoningEffort = (body: Record<string, unknown>): unknown =>
  (body.reasoning as { effort?: unknown } | undefined)?.effort;

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-openrouter-test-'));
  process.env.XDG_DATA_HOME = xdgHome;
  process.env.OPENROUTER_API_KEY = 'test';
  delete process.env.OPENROUTER_MODEL;
  delete process.env.OPENROUTER_SITE_URL;
  delete process.env.OPENROUTER_APP_NAME;
  delete process.env.OPENROUTER_PROVIDER_SORT;
});

afterEach(() => {
  process.env = { ...originalEnv };
  resetProviderRegistry();
  resetPiModels();
  resetCatalogCache();
  server?.close();
  server = undefined;
  if (xdgHome) {
    rmSync(xdgHome, { recursive: true, force: true });
  }
});

describe('runLlmStep openrouter (live SSE)', () => {
  it('streams reasoning (delta.reasoning) LIVE and parses the final usage chunk', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          res.write(makeChunk({ role: 'assistant', content: '' }));
          setTimeout(() => {
            res.write(makeChunk({ reasoning: 'Let me think. ' }));
          }, 150);
          setTimeout(() => {
            res.write(makeChunk({ reasoning: 'Second thought.' }));
          }, 300);
          setTimeout(() => {
            res.write(makeChunk({ content: 'The answer.' }));
          }, 450);
          setTimeout(() => {
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write(
              makeChunk(
                {},
                {
                  choices: [],
                  usage: {
                    prompt_tokens: 100,
                    completion_tokens: 20,
                    total_tokens: 120,
                    completion_tokens_details: { reasoning_tokens: 8 },
                  },
                },
              ),
            );
            res.write('data: [DONE]\n\n');
            res.end();
          }, 600);
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    const t0 = Date.now();
    const reasoningArrivals: number[] = [];
    const result = await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      system: 'sys',
      thinkingEffort: 'high',
      onReasoningDelta: async () => {
        reasoningArrivals.push(Date.now() - t0);
      },
    });

    expect(result.text).toBe('The answer.');
    expect(result.reasoning).toBe('Let me think. Second thought.');
    expect(result.finishReason).toBe('stop');
    expect(reasoningArrivals.length).toBe(2);
    expect(reasoningArrivals[0]).toBeLessThan(450);
    expect(result.usage?.inputTokens).toBe(100);
    expect(result.usage?.outputTokens).toBe(20);
    expect(result.usage?.cacheReadTokens).toBe(0);
    expect(result.usage?.reasoningTokens).toBe(8);
  });

  it('accepts delta.reasoning_content passthrough (DeepSeek routes)', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          res.write(makeChunk({ reasoning_content: 'thinking via passthrough' }));
          res.write(makeChunk({ content: 'done' }));
          res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
          res.write('data: [DONE]\n\n');
          res.end();
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    const result = await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });
    expect(result.reasoning).toBe('thinking via passthrough');
    expect(result.text).toBe('done');
  });

  it("uses Pi's reasoning object and includes usage when the allowlist is unknown", async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'max',
    });
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    expect(bodies).toHaveLength(2);
    expect(reasoningEffort(bodies[0]!)).toBe('max');
    expect(bodies[0]?.stream_options).toEqual({ include_usage: true });
    expect(bodies[0]?.provider).toEqual({ sort: 'price' });
    expect(bodies[0]?.stream).toBe(true);
    expect((bodies[0]?.tools as unknown[]).length).toBe(1);
    expect(bodies[0]?.model).toBe('openrouter/free');
    expect(reasoningEffort(bodies[1]!)).toBeUndefined();
    expect(bodies[1]?.provider).toEqual({ sort: 'price' });
  });

  it("maps Pi's reasoning effort to the model's supported_efforts allowlist", async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, [
              {
                id: 'vendor/glm',
                reasoning: {
                  supported_efforts: ['max', 'high', 'low'],
                  default_effort: 'max',
                  mandatory: true,
                },
              },
            ]);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/glm';

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'max',
    });
    // medium is unsupported on this model → nearest is high (tie breaks up).
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'medium',
    });
    // off has no `none` tier here → lowest supported effort (low).
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    expect(bodies.map(reasoningEffort)).toEqual(['max', 'high', 'low']);
  });

  it('sends Pi reasoning none for off when the model supports it', async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, [
              {
                id: 'vendor/full',
                reasoning: {
                  supported_efforts: ['max', 'high', 'medium', 'low', 'minimal', 'none'],
                  default_effort: 'medium',
                },
              },
            ]);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/full';

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'minimal',
    });

    expect(bodies.map(reasoningEffort)).toEqual(['none', 'minimal']);
  });

  it('keeps Pi reasoning stable across consecutive steps (cache-safe)', async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, [
              {
                id: 'vendor/stable',
                reasoning: {
                  supported_efforts: ['max', 'high', 'medium', 'low', 'minimal', 'none'],
                  default_effort: 'medium',
                },
              },
            ]);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/stable';

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'max',
    });

    expect(bodies).toHaveLength(3);
    expect(reasoningEffort(bodies[0]!)).toBe('none');
    expect(reasoningEffort(bodies[1]!)).toBe('none');
    expect(reasoningEffort(bodies[2]!)).toBe('max');
  });

  it('sends HTTP-Referer and X-Title when configured, and honors OPENROUTER_MODEL', async () => {
    let headers: import('node:http').IncomingHttpHeaders | undefined;
    let body: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          headers = req.headers;
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_SITE_URL = 'https://zed.dev';
    process.env.OPENROUTER_APP_NAME = 'Zen Agent';
    process.env.OPENROUTER_MODEL = 'openai/gpt-5';

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    expect(headers?.['http-referer']).toBe('https://zed.dev');
    expect(headers?.['x-title']).toBe('Zen Agent');
    expect(body?.model).toBe('openai/gpt-5');
  });

  it('honors OPENROUTER_PROVIDER_SORT and allows opting out', async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;

    process.env.OPENROUTER_PROVIDER_SORT = 'latency';
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    process.env.OPENROUTER_PROVIDER_SORT = '';
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    expect(bodies).toHaveLength(2);
    expect(bodies[0]?.provider).toEqual({ sort: 'latency' });
    expect(bodies[1]?.provider).toBeUndefined();
  });

  it('sends session_id so Z.AI pins the context cache to the conversation', async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            serveCatalog(res, []);
            return;
          }
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            res.write(makeChunk({ content: 'ok' }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
            res.write('data: [DONE]\n\n');
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;

    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
      sessionId: 'sess_0123456789abcdef',
    });
    await runLlmStep('openrouter', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });

    expect(bodies).toHaveLength(2);
    expect(bodies[0]?.session_id).toBe('sess_0123456789abcdef');
    expect(bodies[1]?.session_id).toBeUndefined();
  });

  it('requires OPENROUTER_API_KEY', async () => {
    delete process.env.OPENROUTER_API_KEY;
    await expect(
      runLlmStep('openrouter', { messages: [{ role: 'user', content: 'hi' }] }),
    ).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});

describe('openrouter model info through the facade', () => {
  it('falls back to the static table without an API key', async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getContextWindowTokens('openrouter', 'openrouter/free')).toBe(128_000);
    expect(await getModelPricing('openrouter', 'openrouter/free')).toEqual({
      currency: 'USD',
      cacheHitPerM: 0,
      cacheMissPerM: 0,
      outputPerM: 0,
    });
  });

  it('falls back to generic defaults for unknown models', async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getContextWindowTokens('openrouter', 'some/unknown-model')).toBe(200_000);
    const pricing = await getModelPricing('openrouter', 'some/unknown-model');
    expect(pricing.cacheMissPerM).toBeGreaterThan(0);
    expect(pricing.outputPerM).toBeGreaterThan(0);
  });

  it('prefers the live /models catalog when available', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (_req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: [
                {
                  id: 'vendor/model',
                  context_length: 123456,
                  pricing: { prompt: '0.0000005', completion: '0.0000015' },
                },
              ],
            }),
          );
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    expect(await getContextWindowTokens('openrouter', 'vendor/model')).toBe(123456);
    expect(await getModelPricing('openrouter', 'vendor/model')).toEqual({
      currency: 'USD',
      cacheHitPerM: 0.5,
      cacheMissPerM: 0.5,
      outputPerM: 1.5,
    });
  });
});

describe('fetchBalanceSnapshot (openrouter)', () => {
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

  it('reads key usage and limit as USD credits', async () => {
    let authorization: string | undefined;
    const port = await startServer((req, res) => {
      authorization = req.headers.authorization;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          data: { label: 'test', usage: 4.2, limit: 100, is_free_tier: false },
        }),
      );
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    const balance = await fetchBalanceSnapshot('openrouter');

    expect(authorization).toBe('Bearer test');
    expect(balance).toEqual({
      isAvailable: true,
      currency: 'USD',
      total: 95.8,
      details: {
        usageUsd: 4.2,
        limitUsd: 100,
        isFreeTier: false,
      },
    });
  });

  it('throws on a non-OK response', async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(401);
      res.end('{"error":{"message":"invalid key"}}');
    });

    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    await expect(fetchBalanceSnapshot('openrouter')).rejects.toThrow(/401/);
  });

  it('throws without OPENROUTER_API_KEY', async () => {
    delete process.env.OPENROUTER_API_KEY;
    await expect(fetchBalanceSnapshot('openrouter')).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});
