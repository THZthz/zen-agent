import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { getModelOptions, getPiModel, resetPiModels } from './provider-pi.js';
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

describe('user provider parsing', () => {
  it('parses declared models with metadata (context, cost, modalities)', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'https://api.z.ai/api/paas/v4',
        apiKeyEnv: 'ZAI_API_KEY',
        defaultModel: 'glm-5.3',
        models: [
          { id: 'glm-5.3', name: 'GLM 5.3', contextLength: 1_048_576 },
          {
            id: 'glm-5.3-flash',
            name: 'GLM 5.3 Flash',
            contextLength: 1_048_576,
            cost: { inputPerM: 0.075, outputPerM: 0.25 },
            modalities: ['image'],
          },
        ],
      },
    ]);
    const def = getProviderDefinition('zai')!;
    expect(def.discovery.enabled).toBe(false);
    expect(def.defaultModel).toBe('glm-5.3');
    expect(def.staticModels).toHaveLength(2);
    expect(def.staticModels[1]).toMatchObject({
      value: 'glm-5.3-flash',
      contextLength: 1_048_576,
      modalities: ['image'],
    });
    expect(def.staticModels[1]?.cost).toEqual({ inputPerM: 0.075, outputPerM: 0.25 });
  });

  it('defaults to the first declared model when defaultModel is omitted', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'p', baseUrl: 'http://x', models: ['a', 'b'] },
    ]);
    expect(getProviderDefinition('p')?.defaultModel).toBe('a');
  });

  it('rejects a provider with neither models nor fetchModels', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'empty', baseUrl: 'http://x', defaultModel: 'm' },
    ]);
    expect(() => getProviderDefinition('empty')).toThrow(
      /declares no models — add a "models" list or set "fetchModels": true/,
    );
  });

  it('requires defaultModel when fetchModels is true', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'discover', baseUrl: 'http://x', fetchModels: true },
    ]);
    expect(() => getProviderDefinition('discover')).toThrow(
      /"defaultModel" is required when "fetchModels" is true/,
    );
  });

  it('rejects duplicate ids among user providers', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'groq', baseUrl: 'http://a', models: ['m'] },
      { id: 'groq', baseUrl: 'http://b', models: ['m2'] },
    ]);
    expect(() => getProviderDefinition('groq')).toThrow(/duplicate provider id "groq"/);
  });

  it('rejects malformed provider config with a clear error', () => {
    process.env.ZEN_AGENT_PROVIDERS = 'not json';
    expect(() => getProviderDefinition('anything')).toThrow(/ZEN_AGENT_PROVIDERS/);
  });

  it('parses per-model thinkingEfforts (deduped, in declared order)', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'http://x',
        defaultModel: 'glm-5.3-flash',
        models: [{ id: 'glm-5.3-flash', thinkingEfforts: ['off', 'low', 'high', 'high', 'max'] }],
      },
    ]);
    expect(getProviderDefinition('zai')?.staticModels[0]?.thinkingEfforts).toEqual([
      'off',
      'low',
      'high',
      'max',
    ]);
  });

  it('rejects invalid per-model thinkingEfforts values', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'http://x',
        defaultModel: 'm',
        models: [{ id: 'm', thinkingEfforts: ['off', 'turbo'] }],
      },
    ]);
    expect(() => getProviderDefinition('zai')).toThrow(/invalid value "turbo"/);
  });

  it('rejects an empty per-model thinkingEfforts array', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'http://x',
        defaultModel: 'm',
        models: [{ id: 'm', thinkingEfforts: [] }],
      },
    ]);
    expect(() => getProviderDefinition('zai')).toThrow(/non-empty array/);
  });
});

describe('getPiModel', () => {
  it('builds a pi model from declared metadata', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'zai',
        baseUrl: 'https://api.z.ai/api/paas/v4',
        apiKeyEnv: 'ZAI_API_KEY',
        defaultModel: 'glm-5.3',
        models: [
          { id: 'glm-5.3', contextLength: 1_048_576 },
          {
            id: 'glm-5.3-flash',
            contextLength: 1_048_576,
            cost: { inputPerM: 0.075, outputPerM: 0.25 },
            modalities: ['image'],
          },
        ],
      },
    ]);
    const flash = getPiModel('zai', 'glm-5.3-flash');
    expect(flash.contextWindow).toBe(1_048_576);
    expect(flash.input).toEqual(['text', 'image']);
    expect(flash.cost.input).toBe(0.075);
    expect(flash.cost.output).toBe(0.25);

    const glm53 = getPiModel('zai', 'glm-5.3');
    expect(glm53.contextWindow).toBe(1_048_576);
    expect(glm53.input).toEqual(['text']);
  });

  it('synthesizes unknown slugs with conservative defaults', () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      { id: 'p', baseUrl: 'http://x', models: ['m'] },
    ]);
    const model = getPiModel('p', 'some/unknown-model');
    expect(model.contextWindow).toBe(200_000);
    expect(model.input).toEqual(['text']);
    expect(model.cost.input).toBe(1);
    expect(model.cost.output).toBe(2);
  });
});

describe('getModelOptions', () => {
  it('returns the declared static list for static providers', async () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'deepseek',
        baseUrl: 'http://x',
        models: [
          { id: 'deepseek-v4-flash', name: 'Flash', contextLength: 1_000_000 },
          { id: 'deepseek-v4-pro', name: 'Pro', contextLength: 1_000_000 },
        ],
      },
    ]);
    const options = await getModelOptions('deepseek');
    expect(options?.map((o) => o.value)).toEqual(['deepseek-v4-flash', 'deepseek-v4-pro']);
    expect(options?.[0].description).toContain('1.0M');
  });

  it('fetches the live catalog when fetchModels is true', async () => {
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
                  { id: 'groq/no-tools', supported_parameters: ['reasoning'] },
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
        baseUrl: `${testServerBaseUrl(server, port)}/api/v1`,
        apiKeyEnv: 'GROQ_API_KEY',
        defaultModel: 'groq/llama-3.3-70b',
        fetchModels: true,
      },
    ]);

    const options = await getModelOptions('groq');
    // no-tools is filtered out by supported_parameters.
    expect(options?.map((o) => o.value)).toEqual(['groq/llama-3.3-70b', 'groq/qwen-2.5']);
    expect(authorization).toBe('Bearer test');

    // The discovered model is streamable through the facade.
    const model = getPiModel('groq', 'groq/qwen-2.5');
    expect(model.contextWindow).toBe(65536);
  });

  it('keeps declared models alongside a fetched catalog', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ data: [{ id: 'discovered/model', context_length: 1000 }] }));
            return;
          }
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          res.end('data: [DONE]\n\n');
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'hybrid',
        baseUrl: `${testServerBaseUrl(server, port)}/api/v1`,
        apiKeyEnv: 'API_KEY',
        defaultModel: 'declared/model',
        fetchModels: true,
        models: ['declared/model'],
      },
    ]);

    const options = await getModelOptions('hybrid');
    expect(options?.map((o) => o.value)).toEqual(['declared/model', 'discovered/model']);
  });

  it('does not call /models when fetchModels is absent', async () => {
    let catalogHits = 0;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            catalogHits += 1;
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ data: [] }));
            return;
          }
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          res.end('data: [DONE]\n\n');
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'static-only',
        baseUrl: `${testServerBaseUrl(server, port)}/api/v1`,
        defaultModel: 'm',
        models: ['m'],
      },
    ]);

    const options = await getModelOptions('static-only');
    expect(options?.map((o) => o.value)).toEqual(['m']);
    expect(catalogHits).toBe(0);
  });
});
