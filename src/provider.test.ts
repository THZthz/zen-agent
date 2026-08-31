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
} from './provider.js';
import { resetPiModels } from './provider-pi.js';
import { resetCatalogCache } from './provider-catalog.js';
import { resetProviderRegistry } from './provider-registry.js';
import { testServerBaseUrl } from './test-server.js';

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
  it('offers the full ladder for user-defined providers', async () => {
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
