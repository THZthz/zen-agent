import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getDefaultModel,
  getModelPricing,
  getProviderCurrency,
  getThinkingEfforts,
  resolveModelModalities,
  runLlmStep,
} from '../index.js';
import { resetPiModels } from '../pi.js';
import { resetCatalogCache } from '../catalog.js';
import { resetProviderRegistry } from '../registry.js';
import { testServerBaseUrl } from '../../test-server.js';

const originalEnv = { ...process.env };
let xdgHome: string;
let server: import('node:http').Server | undefined;

const STATIC_PROVIDER = JSON.stringify([
  {
    id: 'groq',
    name: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    apiKeyEnv: 'GROQ_API_KEY',
    defaultModel: 'llama-3.3-70b',
    currency: 'USD',
    models: [
      {
        id: 'llama-3.3-70b',
        name: 'Llama 3.3 70B',
        contextLength: 131_072,
        cost: { inputPerM: 0.59, outputPerM: 0.79 },
        modalities: ['image'],
      },
    ],
  },
]);

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-provider-test-'));
  process.env.XDG_DATA_HOME = xdgHome;
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

describe('runLlmStep', () => {
  it('rejects unknown providers with a configuration hint', async () => {
    await expect(runLlmStep('nope', { messages: [] })).rejects.toThrow(
      /Unknown LLM provider "nope"/,
    );
  });

  it('requires the declared API key env var', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    delete process.env.GROQ_API_KEY;
    await expect(
      runLlmStep('groq', { messages: [{ role: 'user', content: 'hi' }] }),
    ).rejects.toThrow(/GROQ_API_KEY/);
  });
});

describe('getDefaultModel', () => {
  it('returns the declared default model', () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    expect(getDefaultModel('groq')).toBe('llama-3.3-70b');
  });
});

describe('getProviderCurrency', () => {
  it('returns the declared billing currency', () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    expect(getProviderCurrency('groq')).toBe('USD');
  });
});

describe('getModelPricing', () => {
  it('uses the declared model cost', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    const pricing = await getModelPricing('groq', 'llama-3.3-70b');
    expect(pricing).toEqual({
      currency: 'USD',
      cacheHitPerM: 0.59,
      cacheMissPerM: 0.59,
      outputPerM: 0.79,
    });
  });

  it('falls back to the provider fallback prices for undeclared models', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    const pricing = await getModelPricing('groq', 'some/other-model');
    expect(pricing).toEqual({
      currency: 'USD',
      cacheHitPerM: 1,
      cacheMissPerM: 1,
      outputPerM: 2,
    });
  });
});

describe('getContextWindowTokens', () => {
  it('uses the declared context length', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    expect(await getContextWindowTokens('groq', 'llama-3.3-70b')).toBe(131_072);
  });
});

describe('getThinkingEfforts', () => {
  it('offers the full ladder when the model declares none', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    expect(await getThinkingEfforts('groq', 'llama-3.3-70b')).toEqual([
      'off',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ]);
  });

  it('offers the declared per-model thinkingEfforts in declared order', async () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'http://x',
        defaultModel: 'glm-5.3-flash',
        models: [
          { id: 'glm-5.3-flash', thinkingEfforts: ['off', 'low', 'high', 'max'] },
          { id: 'glm-5.3', thinkingEfforts: ['low', 'high', 'max'] },
        ],
      },
    ]);
    expect(await getThinkingEfforts('zai', 'glm-5.3-flash')).toEqual(['off', 'low', 'high', 'max']);
    // No off: mandatory reasoning.
    expect(await getThinkingEfforts('zai', 'glm-5.3')).toEqual(['low', 'high', 'max']);
  });
});

describe('getModelModalities', () => {
  it('honors declared modalities for static models', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    expect(await resolveModelModalities('groq', 'llama-3.3-70b')).toEqual({
      image: true,
      audio: false,
    });
  });
});

describe('fetchBalanceSnapshot', () => {
  it('returns an unavailable snapshot for providers without a balance endpoint', async () => {
    process.env.ZEN_AGENT_PROVIDERS = STATIC_PROVIDER;
    const snapshot = await fetchBalanceSnapshot('groq');
    expect(snapshot).toEqual({
      isAvailable: false,
      currency: 'USD',
      total: 0,
      details: {},
    });
  });
});

describe('per-model thinking effort mapping', () => {
  const EFFORT_PROVIDER = (thinkingEfforts: string[]) => ({
    id: 'effort',
    baseUrl: 'http://x',
    apiKeyEnv: 'EFFORT_API_KEY',
    defaultModel: 'm',
    models: [{ id: 'm', thinkingEfforts }],
  });

  async function captureReasoningEffort(effort: string): Promise<string | undefined> {
    let body: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            const chunk = JSON.stringify({
              choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
            });
            res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });
    process.env.EFFORT_API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        ...EFFORT_PROVIDER(['off', 'low', 'high', 'max']),
        baseUrl: testServerBaseUrl(server, port),
      },
    ]);
    await runLlmStep('effort', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: effort as never,
    });
    return (body?.reasoning_effort as string | undefined) ?? undefined;
  }

  it('sends declared values unchanged and omits the field for off', async () => {
    expect(await captureReasoningEffort('high')).toBe('high');
    expect(await captureReasoningEffort('max')).toBe('max');
    expect(await captureReasoningEffort('off')).toBeUndefined();
  });

  it('remaps unsupported session values to the nearest declared effort (ties up)', async () => {
    expect(await captureReasoningEffort('minimal')).toBe('low');
    expect(await captureReasoningEffort('medium')).toBe('high');
    expect(await captureReasoningEffort('xhigh')).toBe('max');
  });

  it('maps off to the lowest effort on mandatory-reasoning models', async () => {
    let body: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            const chunk = JSON.stringify({
              choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
            });
            res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });
    process.env.EFFORT_API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'effort',
        baseUrl: testServerBaseUrl(server, port),
        apiKeyEnv: 'EFFORT_API_KEY',
        defaultModel: 'm',
        models: [{ id: 'm', thinkingEfforts: ['low', 'high', 'max'] }],
      },
    ]);
    await runLlmStep('effort', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: 'off',
    });
    expect(body?.reasoning_effort).toBe('low');
  });
});

describe('deepseek thinking mode', () => {
  async function captureBody(
    provider: Record<string, unknown>,
    effort: string,
  ): Promise<Record<string, unknown>> {
    let body: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            const chunk = JSON.stringify({
              choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
            });
            res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });
    process.env.DS_API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { ...provider, baseUrl: testServerBaseUrl(server, port) },
    ]);
    await runLlmStep('ds', {
      messages: [{ role: 'user', content: 'hi' }],
      thinkingEffort: effort as never,
    });
    return body!;
  }

  const DS_PROVIDER = {
    id: 'ds',
    apiKeyEnv: 'DS_API_KEY',
    defaultModel: 'deepseek-v4-flash',
    thinkingMode: 'deepseek',
    models: [{ id: 'deepseek-v4-flash', thinkingEfforts: ['off', 'low', 'high', 'max'] }],
  };

  it('sends thinking:{type:"disabled"} for off (no reasoning_effort)', async () => {
    const body = await captureBody(DS_PROVIDER, 'off');
    expect(body.thinking).toEqual({ type: 'disabled' });
    expect(body.reasoning_effort).toBeUndefined();
  });

  it('sends thinking:{type:"enabled"} plus the mapped effort for non-off', async () => {
    const body = await captureBody(DS_PROVIDER, 'high');
    expect(body.thinking).toEqual({ type: 'enabled' });
    expect(body.reasoning_effort).toBe('high');
  });

  it('passes efforts through unchanged when no thinkingEfforts are declared (API auto-maps)', async () => {
    const body = await captureBody({ ...DS_PROVIDER, models: ['deepseek-v4-flash'] }, 'medium');
    expect(body.thinking).toEqual({ type: 'enabled' });
    expect(body.reasoning_effort).toBe('medium');
  });

  it('offers the full ladder when the model declares no thinkingEfforts', async () => {
    process.env.DS_API_KEY = 'test';
    const provider = {
      id: 'ds',
      baseUrl: 'http://x',
      apiKeyEnv: 'DS_API_KEY',
      defaultModel: 'm',
      thinkingMode: 'deepseek',
      models: ['m'],
    };
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([provider]);
    const { getThinkingEfforts: getEfforts } = await import('../index.js');
    expect(await getEfforts('ds', 'm')).toEqual([
      'off',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ]);
  });
});

describe('user provider stream', () => {
  it('streams through the facade to the declared base URL', async () => {
    let seenModel: string | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          const chunks: Buffer[] = [];
          req.on('data', (c: Buffer) => chunks.push(c));
          req.on('end', () => {
            seenModel = (JSON.parse(Buffer.concat(chunks).toString('utf8')) as { model?: string })
              .model;
            res.writeHead(200, { 'content-type': 'text/event-stream' });
            const chunk = JSON.stringify({
              choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
            });
            res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.GROQ_API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'groq',
        baseUrl: testServerBaseUrl(server, port),
        apiKeyEnv: 'GROQ_API_KEY',
        defaultModel: 'llama-3.3-70b',
        models: ['llama-3.3-70b'],
      },
    ]);
    const result = await runLlmStep('groq', {
      messages: [{ role: 'user', content: 'hi' }],
    });
    expect(seenModel).toBe('llama-3.3-70b');
    expect(result.text).toBe('ok');
  });
});
