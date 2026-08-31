import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { buildEffortMap, getModelOptions, getPiModel, resetPiModels } from './provider-pi.js';
import { resetCatalogCache } from './provider-catalog.js';
import { getProviderDefinition, resetProviderRegistry } from './provider-registry.js';
import { testServerBaseUrl } from './test-server.js';

const originalEnv = { ...process.env };
let xdgHome: string;
let server: import('node:http').Server | undefined;

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-pi-test-'));
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

describe('buildEffortMap', () => {
  it('returns DeepSeek static map for static-map providers', () => {
    const def = getProviderDefinition('deepseek')!;
    expect(buildEffortMap(def, null)).toEqual({
      off: 'disabled',
      minimal: 'low',
      low: 'low',
      medium: 'high',
      high: 'high',
      xhigh: 'high',
      max: 'max',
    });
  });

  it('passes through every value on unknown openrouter allowlists and omits off', () => {
    const def = getProviderDefinition('openrouter')!;
    const map = buildEffortMap(def, null);
    expect(map).toEqual({
      minimal: 'minimal',
      low: 'low',
      medium: 'medium',
      high: 'high',
      xhigh: 'xhigh',
      max: 'max',
      off: null,
    });
  });

  it('remaps openrouter efforts to the model allowlist by ladder distance', () => {
    const def = getProviderDefinition('openrouter')!;
    const map = buildEffortMap(def, {
      id: 'glm',
      name: 'GLM',
      inputPerM: 0,
      outputPerM: 0,
      cacheReadPerM: 0,
      contextLength: 200_000,
      supportsTools: true,
      inputModalities: null,
      reasoning: {
        supportedEfforts: ['max', 'high', 'low'],
        defaultEffort: 'max',
        mandatory: true,
      },
    });
    expect(map).toEqual({
      minimal: 'low',
      low: 'low',
      medium: 'high',
      high: 'high',
      xhigh: 'max',
      max: 'max',
      off: 'low',
    });
  });

  it('maps off to none when the model supports it, null otherwise', () => {
    const def = getProviderDefinition('openrouter')!;
    const withNone = buildEffortMap(def, {
      id: 'full',
      name: 'Full',
      inputPerM: 0,
      outputPerM: 0,
      cacheReadPerM: 0,
      contextLength: 200_000,
      supportsTools: true,
      inputModalities: null,
      reasoning: {
        supportedEfforts: ['max', 'high', 'medium', 'low', 'minimal', 'none'],
        defaultEffort: 'medium',
        mandatory: false,
      },
    });
    expect(withNone?.off).toBe('none');

    const noNone = buildEffortMap(def, {
      id: 'no-none',
      name: 'No None',
      inputPerM: 0,
      outputPerM: 0,
      cacheReadPerM: 0,
      contextLength: 200_000,
      supportsTools: true,
      inputModalities: null,
      reasoning: {
        supportedEfforts: ['max', 'high', 'low'],
        defaultEffort: 'max',
        mandatory: false,
      },
    });
    expect(noNone?.off).toBeNull();
  });
});

describe('getPiModel', () => {
  it('returns the static baseline model for openrouter/free without network', () => {
    delete process.env.OPENROUTER_API_KEY;
    const model = getPiModel('openrouter', 'openrouter/free');
    expect(model.id).toBe('openrouter/free');
    expect(model.contextWindow).toBe(128_000);
    expect(model.cost.input).toBe(0);
    expect(model.thinkingLevelMap?.off).toBeNull();
  });

  it('synthesizes unknown slugs with conservative defaults', () => {
    delete process.env.OPENROUTER_API_KEY;
    const model = getPiModel('openrouter', 'some/unknown-model');
    expect(model.contextWindow).toBe(200_000);
    expect(model.input).toEqual(['text']);
    expect(model.cost.input).toBe(1);
    expect(model.cost.output).toBe(2);
  });

  it('applies curated z.ai specs to GLM 5.3 Flash even without a catalog', () => {
    delete process.env.OPENROUTER_API_KEY;
    const flash = getPiModel('openrouter', 'z-ai/glm-5.3-flash');
    expect(flash.contextWindow).toBe(1_048_576);
    expect(flash.input).toEqual(['text', 'image']);
    expect(flash.maxTokens).toBe(131_072);
    expect(flash.cost.input).toBe(0.075);
    expect(flash.cost.output).toBe(0.25);
    expect(flash.cost.cacheRead).toBe(0.015);

    const glm53 = getPiModel('openrouter', 'z-ai/glm-5.3');
    expect(glm53.contextWindow).toBe(1_048_576);
    expect(glm53.input).toEqual(['text']);
    expect(glm53.thinkingLevelMap?.off).toBe('low'); // mandatory reasoning
  });
});

describe('getModelOptions (discovery)', () => {
  it('fetches the live catalog, filters tool-capable models and pins openrouter/free first', async () => {
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
                  pricing: { prompt: '0.000001', completion: '0.000002' },
                  supported_parameters: ['tools'],
                },
                {
                  id: 'vendor/alpha',
                  name: 'Alpha',
                  context_length: 200000,
                  pricing: { prompt: '0.0000005', completion: '0.0000015' },
                },
                {
                  id: 'vendor/no-tools',
                  name: 'No Tools',
                  supported_parameters: ['reasoning'],
                },
                {
                  id: 'openrouter/free',
                  name: 'OpenRouter Free',
                  context_length: 128000,
                  pricing: { prompt: '0', completion: '0' },
                  supported_parameters: ['tools'],
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
    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;

    const options = await getModelOptions('openrouter');
    expect(options?.map((o) => o.value)).toEqual([
      'openrouter/free',
      'vendor/alpha',
      'vendor/zeta',
    ]);
    expect(options?.[0].description).toContain('128K');
  });

  it('falls back to the static baseline when discovery is unavailable', async () => {
    delete process.env.OPENROUTER_API_KEY;
    const options = await getModelOptions('openrouter');
    expect(options?.map((o) => o.value)).toEqual(['openrouter/free']);
  });

  it('returns the static list for static providers', async () => {
    const options = await getModelOptions('deepseek');
    expect(options?.map((o) => o.value)).toEqual(['deepseek-v4-flash', 'deepseek-v4-pro']);
  });
});

describe('user-defined providers via ZEN_AGENT_PROVIDERS', () => {
  it('auto-discovers models from just an endpoint and an API key', async () => {
    let authorization: string | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            authorization = req.headers.authorization;
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(
              JSON.stringify({
                data: [
                  { id: 'groq/llama-3.3-70b', context_length: 131072 },
                  { id: 'groq/qwen-2.5', context_length: 65536 },
                ],
              }),
            );
            return;
          }
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          const chunk = JSON.stringify({
            choices: [{ index: 0, delta: { content: 'hi' }, finish_reason: 'stop' }],
          });
          res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
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
        name: 'Groq',
        baseUrl: `${testServerBaseUrl(server, port)}/api/v1`,
        apiKeyEnv: 'GROQ_API_KEY',
        defaultModel: 'groq/llama-3.3-70b',
        currency: 'USD',
      },
    ]);

    const options = await getModelOptions('groq');
    expect(options?.map((o) => o.value)).toEqual(['groq/llama-3.3-70b', 'groq/qwen-2.5']);
    expect(authorization).toBe('Bearer test');

    // The discovered model is streamable through the facade.
    const model = getPiModel('groq', 'groq/qwen-2.5');
    expect(model.contextWindow).toBe(65536);
    expect(model.compat?.thinkingFormat).toBe('openai');
  });

  it('rejects malformed provider config with a clear error', () => {
    process.env.ZEN_AGENT_PROVIDERS = 'not json';
    expect(() => getProviderDefinition('anything')).toThrow(/ZEN_AGENT_PROVIDERS/);
  });

  it('rejects user providers colliding with built-ins and explains the alternatives', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'openrouter', baseUrl: 'http://x', defaultModel: 'm', models: ['m'] },
    ]);
    expect(() => getProviderDefinition('openrouter')).toThrow(/collides with a built-in provider/);
    // The message points at the built-in's own env vars and a distinct id.
    expect(() => getProviderDefinition('openrouter')).toThrow(/OPENROUTER_BASE_URL/);
    expect(() => getProviderDefinition('openrouter')).toThrow(/"my-openrouter"/);

    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'deepseek', baseUrl: 'http://x', defaultModel: 'm', models: ['m'] },
    ]);
    expect(() => getProviderDefinition('deepseek')).toThrow(/DEEPSEEK_BASE_URL/);
    expect(() => getProviderDefinition('deepseek')).toThrow(/"my-deepseek"/);
  });

  it('rejects duplicate ids among user providers', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'groq', baseUrl: 'http://a', defaultModel: 'm', models: ['m'] },
      { id: 'groq', baseUrl: 'http://b', defaultModel: 'm2', models: ['m2'] },
    ]);
    expect(() => getProviderDefinition('groq')).toThrow(/duplicate provider id "groq"/);
  });
});
