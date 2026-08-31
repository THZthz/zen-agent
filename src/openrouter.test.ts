import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  getOpenRouterModelInfo,
  getOpenRouterModelOptions,
  getOpenRouterReasoning,
  parseReasoning,
  resetOpenRouterModelsCache,
} from './openrouter-models.js';
import {
  DEFAULT_OPENROUTER_MODEL,
  fetchOpenRouterBalance,
  mapOpenRouterEffort,
  runOpenRouterStep,
} from './openrouter.js';

describe('runOpenRouterStep (live SSE)', () => {
  const originalEnv = { ...process.env };
  let server: import('node:http').Server | undefined;

  const makeChunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
    `data: ${JSON.stringify({
      id: '1',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'or-model',
      choices: [{ index: 0, delta, finish_reason: null }],
      ...extra,
    })}\n\n`;

  /**
   * runOpenRouterStep now consults GET /models for the model's
   * reasoning-effort allowlist (cached in memory). The SSE servers must
   * answer that request with a JSON catalog instead of a stream.
   */
  const serveCatalog = (res: import('node:http').ServerResponse, entries: unknown[]) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ data: entries }));
  };

  const reasoningEffort = (body: Record<string, unknown>): unknown =>
    (body.reasoning as { effort?: unknown } | undefined)?.effort;

  beforeEach(() => {
    process.env.OPENROUTER_API_KEY = 'test';
    delete process.env.OPENROUTER_MODEL;
    delete process.env.OPENROUTER_SITE_URL;
    delete process.env.OPENROUTER_APP_NAME;
    delete process.env.OPENROUTER_PROVIDER_SORT;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
  });

  it('streams reasoning (delta.reasoning) LIVE and parses the final usage chunk', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          // Answer the reasoning-allowlist lookup immediately so it does not
          // consume (and delay) the timed SSE stream below.
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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const t0 = Date.now();
    const reasoningArrivals: number[] = [];
    const result = await runOpenRouterStep({
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
    // Reasoning arrived BEFORE the answer started (450ms) — live streaming,
    // not a buffered burst.
    expect(reasoningArrivals.length).toBe(2);
    expect(reasoningArrivals[0]).toBeLessThan(450);
    // Usage came from the final chunk (no cache fields reported → 0).
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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const result = await runOpenRouterStep({
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
            // Empty catalog: `openrouter/free` is unknown, so the gateway
            // allowlist is assumed to accept every effort value.
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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'max' });
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

    expect(bodies).toHaveLength(2);
    expect(reasoningEffort(bodies[0]!)).toBe('max');
    expect(bodies[0]?.stream_options).toEqual({ include_usage: true });
    expect(bodies[0]?.provider).toEqual({ sort: 'price' });
    expect(bodies[0]?.stream).toBe(true);
    expect((bodies[0]?.tools as unknown[]).length).toBe(1);
    expect(bodies[0]?.model).toBe(DEFAULT_OPENROUTER_MODEL);
    expect(reasoningEffort(bodies[1]!)).toBeUndefined();
    expect(bodies[1]?.provider).toEqual({ sort: 'price' });
  });

  it("maps Pi's reasoning effort to the model's supported_efforts allowlist", async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            // e.g. a GLM-style model: max/high/low, no medium/minimal/none.
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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/glm';

    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'max' });
    // medium is unsupported on this model → nearest is high (tie breaks up).
    await runOpenRouterStep({
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'medium',
    });
    // off has no `none` tier here → lowest supported effort (low).
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/full';

    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });
    await runOpenRouterStep({
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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    process.env.OPENROUTER_MODEL = 'vendor/stable';

    // Two steps at the same effort: the allowlist lookup is cached in memory,
    // so the request body (including Pi's reasoning object) is deterministic. The
    // provider's context cache keys on system + tools + messages, none of
    // which vary between steps.
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'max' });

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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    process.env.OPENROUTER_SITE_URL = 'https://zed.dev';
    process.env.OPENROUTER_APP_NAME = 'Zen Agent';
    process.env.OPENROUTER_MODEL = 'openai/gpt-5';

    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    process.env.OPENROUTER_PROVIDER_SORT = 'latency';
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

    process.env.OPENROUTER_PROVIDER_SORT = '';
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    await runOpenRouterStep({
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
      sessionId: 'sess_0123456789abcdef',
    });
    await runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }], thinkingEffort: 'off' });

    expect(bodies).toHaveLength(2);
    expect(bodies[0]?.session_id).toBe('sess_0123456789abcdef');
    expect(bodies[1]?.session_id).toBeUndefined();
  });

  it('requires OPENROUTER_API_KEY', async () => {
    delete process.env.OPENROUTER_API_KEY;
    await expect(
      runOpenRouterStep({ messages: [{ role: 'user', content: 'hi' }] }),
    ).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});

describe('mapOpenRouterEffort', () => {
  it('passes through unknown allowlists (offline/unknown model) and omits off', () => {
    expect(mapOpenRouterEffort('max', null)).toBe('max');
    expect(mapOpenRouterEffort('xhigh', null)).toBe('xhigh');
    expect(mapOpenRouterEffort('low', null)).toBe('low');
    expect(mapOpenRouterEffort('off', null)).toBeNull();
  });

  it('sends exact values present in the allowlist', () => {
    const supported = ['max', 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'];
    for (const effort of ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const) {
      expect(mapOpenRouterEffort(effort, supported)).toBe(effort === 'off' ? 'none' : effort);
    }
  });

  it('maps max/xhigh to the nearest supported effort instead of collapsing to high', () => {
    // Model caps at high (e.g. an OpenAI-style model): max and xhigh → high.
    const capped = ['high', 'medium', 'low'];
    expect(mapOpenRouterEffort('max', capped)).toBe('high');
    expect(mapOpenRouterEffort('xhigh', capped)).toBe('high');
    // Model supports xhigh but not max (e.g. Claude 5 Sonnet): max → xhigh.
    const xhighOnly = ['xhigh', 'high', 'medium', 'low'];
    expect(mapOpenRouterEffort('max', xhighOnly)).toBe('xhigh');
  });

  it('resolves medium on a max/high/low model to high (tie breaks upward)', () => {
    expect(mapOpenRouterEffort('medium', ['max', 'high', 'low'])).toBe('high');
  });

  it('maps off to none when supported, else to the lowest effort only for mandatory-reasoning models', () => {
    // Model with a none tier: off = none.
    expect(mapOpenRouterEffort('off', ['max', 'high', 'low', 'none'])).toBe('none');
    // Mandatory-reasoning model (no none tier): off = lowest supported effort.
    expect(mapOpenRouterEffort('off', ['max', 'high', 'low'], true)).toBe('low');
    expect(mapOpenRouterEffort('off', ['high', 'medium', 'low', 'minimal'], true)).toBe('minimal');
    // Non-mandatory model without a none tier: omit the field so the
    // provider's default (usually off/low) applies.
    expect(mapOpenRouterEffort('off', ['max', 'high', 'low'])).toBeNull();
    expect(mapOpenRouterEffort('off', ['xhigh', 'high'], false)).toBeNull();
  });
});

describe('parseReasoning', () => {
  it('parses supported_efforts and default_effort from the catalog reasoning object', () => {
    expect(
      parseReasoning({
        supported_efforts: ['xhigh', 'high', 'medium', 'low', 'minimal'],
        default_effort: 'medium',
        mandatory: true,
      }),
    ).toEqual({
      supportedEfforts: ['xhigh', 'high', 'medium', 'low', 'minimal'],
      defaultEffort: 'medium',
      mandatory: true,
    });
  });

  it('returns null allowlist when reasoning metadata is absent or malformed', () => {
    expect(parseReasoning(undefined)).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
    expect(parseReasoning(null)).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
    expect(parseReasoning('nope')).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
    expect(parseReasoning({})).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
    expect(parseReasoning({ supported_efforts: [], default_effort: '' })).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
  });
});

describe('getOpenRouterReasoning', () => {
  const originalEnv = { ...process.env };
  let server: import('node:http').Server | undefined;

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
    server?.close();
    server = undefined;
  });

  it('returns the catalog allowlist for a known model and null fallback for unknown ones', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (_req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: [
                {
                  id: 'vendor/model',
                  reasoning: {
                    supported_efforts: ['max', 'high', 'low'],
                    default_effort: 'max',
                  },
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

    process.env.OPENROUTER_API_KEY = 'test';
    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    expect(await getOpenRouterReasoning('vendor/model')).toEqual({
      supportedEfforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
      mandatory: null,
    });
    expect(await getOpenRouterReasoning('vendor/other')).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
  });

  it('degrades to the full-gateway allowlist when the catalog fetch fails', async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getOpenRouterReasoning('vendor/model')).toEqual({
      supportedEfforts: null,
      defaultEffort: null,
      mandatory: null,
    });
  });

  it('falls back to the persisted catalog file for offline starts when cwd is given', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'zen-agent-reasoning-'));
    try {
      await mkdir(join(dir, '.sessions', 'client'), { recursive: true });
      await writeFile(
        join(dir, '.sessions', 'client', 'models.openrouter.json'),
        JSON.stringify({
          version: 3,
          fetchedAt: new Date().toISOString(),
          baseUrl: 'https://openrouter.ai/api/v1',
          models: [
            {
              id: 'cached/model',
              name: 'Cached',
              inputPerM: 1,
              outputPerM: 2,
              contextLength: 1000,
              supportsTools: true,
              reasoning: {
                supportedEfforts: ['max', 'high', 'low'],
                defaultEffort: 'max',
                mandatory: true,
              },
            },
          ],
        }),
        'utf8',
      );
      delete process.env.OPENROUTER_API_KEY;

      expect(await getOpenRouterReasoning('cached/model', dir)).toEqual({
        supportedEfforts: ['max', 'high', 'low'],
        defaultEffort: 'max',
        mandatory: true,
      });
      // Unknown to the persisted file: full-gateway allowlist.
      expect(await getOpenRouterReasoning('vendor/other', dir)).toEqual({
        supportedEfforts: null,
        defaultEffort: null,
        mandatory: null,
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('getOpenRouterModelInfo', () => {
  const originalEnv = { ...process.env };
  let server: import('node:http').Server | undefined;

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
    server?.close();
    server = undefined;
  });

  it('falls back to the static table without an API key', async () => {
    delete process.env.OPENROUTER_API_KEY;
    const info = await getOpenRouterModelInfo('openrouter/free');
    expect(info).toEqual({ inputPerM: 0, outputPerM: 0, contextLength: 128_000 });
  });

  it('falls back to generic defaults for unknown models', async () => {
    delete process.env.OPENROUTER_API_KEY;
    const info = await getOpenRouterModelInfo('some/unknown-model');
    expect(info.contextLength).toBe(200_000);
    expect(info.inputPerM).toBeGreaterThan(0);
    expect(info.outputPerM).toBeGreaterThan(0);
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
                  pricing: { prompt: '0.5', completion: '1.5' },
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

    process.env.OPENROUTER_API_KEY = 'test';
    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const info = await getOpenRouterModelInfo('vendor/model');
    expect(info).toEqual({ inputPerM: 0.5, outputPerM: 1.5, contextLength: 123456 });
  });

  it('retries the /models fetch after a failed attempt instead of caching the rejection', async () => {
    let requests = 0;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (_req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          requests += 1;
          if (requests === 1) {
            res.writeHead(500);
            res.end('boom');
            return;
          }
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: [
                {
                  id: 'vendor/model',
                  context_length: 123456,
                  pricing: { prompt: '0.5', completion: '1.5' },
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

    process.env.OPENROUTER_API_KEY = 'test';
    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    // First call hits the failure and falls back WITHOUT poisoning the cache.
    const fallback = await getOpenRouterModelInfo('vendor/model');
    expect(fallback.contextLength).toBe(200_000);

    // Second call retries the live fetch and sees the catalog.
    const info = await getOpenRouterModelInfo('vendor/model');
    expect(info).toEqual({ inputPerM: 0.5, outputPerM: 1.5, contextLength: 123456 });
    expect(requests).toBe(2);
  });
});

describe('getOpenRouterModelOptions', () => {
  const originalEnv = { ...process.env };
  let server: import('node:http').Server | undefined;
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'zen-agent-models-'));
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
    server?.close();
    server = undefined;
    rmSync(dir, { recursive: true, force: true });
  });

  it('fetches the live catalog, persists it, and returns tool-capable models sorted with openrouter/free first', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (_req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: [
                {
                  id: 'vendor/zeta',
                  name: 'Zeta',
                  context_length: 1000,
                  pricing: { prompt: '1', completion: '2' },
                  supported_parameters: ['tools'],
                  reasoning: {
                    supported_efforts: ['max', 'xhigh', 'high', 'medium', 'low'],
                    default_effort: 'medium',
                  },
                },
                {
                  id: 'vendor/alpha',
                  name: 'Alpha',
                  context_length: 200000,
                  pricing: { prompt: '0.5', completion: '1.5' },
                },
                {
                  id: 'vendor/no-tools',
                  name: 'No Tools',
                  context_length: 1000,
                  pricing: { prompt: '1', completion: '2' },
                  supported_parameters: ['reasoning'],
                },
                {
                  id: 'openrouter/free',
                  name: 'OpenRouter Free',
                  context_length: 128000,
                  pricing: { prompt: '0', completion: '0' },
                  supported_parameters: ['tools'],
                  reasoning: {
                    supported_efforts: ['high', 'medium', 'low', 'minimal', 'none'],
                    default_effort: 'medium',
                  },
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

    process.env.OPENROUTER_API_KEY = 'test';
    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    const options = await getOpenRouterModelOptions(dir);
    expect(options?.map((o) => o.value)).toEqual([
      'openrouter/free',
      'vendor/alpha',
      'vendor/zeta',
    ]);
    expect(options?.[0].description).toContain('128K');

    // The catalog is persisted for offline restarts.
    const raw = await readFile(join(dir, '.sessions', 'client', 'models.openrouter.json'), 'utf8');
    const file = JSON.parse(raw) as {
      version: number;
      models: Array<{ id: string }>;
    };
    expect(file.version).toBe(3);
    expect(file.models.map((m) => m.id)).toEqual([
      'vendor/zeta',
      'vendor/alpha',
      'vendor/no-tools',
      'openrouter/free',
    ]);
  });

  it('falls back to the persisted file when the live fetch fails', async () => {
    await mkdir(join(dir, '.sessions', 'client'), { recursive: true });
    await writeFile(
      join(dir, '.sessions', 'client', 'models.openrouter.json'),
      JSON.stringify({
        version: 3,
        fetchedAt: new Date().toISOString(),
        baseUrl: 'https://openrouter.ai/api/v1',
        models: [
          {
            id: 'cached/model',
            name: 'Cached',
            inputPerM: 1,
            outputPerM: 2,
            contextLength: 1000,
            supportsTools: true,
            reasoning: {
              supportedEfforts: ['max', 'high', 'low'],
              defaultEffort: 'max',
            },
          },
        ],
      }),
      'utf8',
    );
    delete process.env.OPENROUTER_API_KEY;

    const options = await getOpenRouterModelOptions(dir);
    expect(options?.map((o) => o.value)).toEqual(['openrouter/free', 'cached/model']);
  });

  it('returns null when neither the live fetch nor the file is available', async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getOpenRouterModelOptions(dir)).toBeNull();
  });
});

describe('fetchOpenRouterBalance', () => {
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

  beforeEach(() => {
    process.env.OPENROUTER_API_KEY = 'test';
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
  });

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

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const balance = await fetchOpenRouterBalance();

    expect(authorization).toBe('Bearer test');
    expect(balance).toEqual({
      isAvailable: true,
      currency: 'USD',
      remainingUsd: 95.8,
      usageUsd: 4.2,
      limitUsd: 100,
      isFreeTier: false,
    });
  });

  it('throws on a non-OK response', async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(401);
      res.end('{"error":{"message":"invalid key"}}');
    });

    process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    await expect(fetchOpenRouterBalance()).rejects.toThrow(/401/);
  });

  it('throws without OPENROUTER_API_KEY', async () => {
    delete process.env.OPENROUTER_API_KEY;
    await expect(fetchOpenRouterBalance()).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});
