import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  getDefaultModel,
  getModelPricing,
  getContextWindowTokens,
  runLlmStep,
  fetchBalanceSnapshot,
  getThinkingEfforts,
} from './provider.js';
import { resetPiModels } from './provider-pi.js';
import { resetCatalogCache } from './provider-catalog.js';
import { resetProviderRegistry } from './provider-registry.js';
import { testServerBaseUrl } from './test-server.js';

const originalEnv = { ...process.env };
let xdgHome: string;
let server: import('node:http').Server | undefined;

function serveChat(res: import('node:http').ServerResponse) {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const chunk = JSON.stringify({
    choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
  });
  res.end(`data: ${chunk}\n\ndata: [DONE]\n\n`);
}

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-envcompat-'));
  process.env.XDG_DATA_HOME = xdgHome;
});
afterEach(() => {
  process.env = { ...originalEnv };
  resetProviderRegistry();
  resetPiModels();
  resetCatalogCache();
  server?.close();
  server = undefined;
  if (xdgHome) rmSync(xdgHome, { recursive: true, force: true });
});

describe('legacy env vars', () => {
  it('DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL + DEEPSEEK_MODEL drive a deepseek step', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((req: any, res: any) => {
        let body = '';
        req.on('data', (d: Buffer) => (body += d.toString()));
        req.on('end', () => {
          const parsed = JSON.parse(body);
          expect(parsed.model).toBe('my-custom-deepseek-model');
          serveChat(res);
        });
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.DEEPSEEK_API_KEY = 'k123';
    process.env.DEEPSEEK_BASE_URL = testServerBaseUrl(server, port);
    process.env.DEEPSEEK_MODEL = 'my-custom-deepseek-model';

    expect(getDefaultModel('deepseek')).toBe('my-custom-deepseek-model');
    const result = await runLlmStep('deepseek', { messages: [{ role: 'user', content: 'hi' }] });
    expect(result.text).toBe('ok');
  });

  it('OPENROUTER_API_KEY + OPENROUTER_BASE_URL + OPENROUTER_MODEL drive an openrouter step', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((req: any, res: any) => {
        if (req.url?.endsWith('/models')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ data: [] }));
          return;
        }
        let body = '';
        req.on('data', (d: Buffer) => (body += d.toString()));
        req.on('end', () => {
          const parsed = JSON.parse(body);
          expect(parsed.model).toBe('my-custom-or-model');
          serveChat(res);
        });
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.OPENROUTER_API_KEY = 'or-key';
    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_MODEL = 'my-custom-or-model';

    expect(getDefaultModel('openrouter')).toBe('my-custom-or-model');
    const result = await runLlmStep('openrouter', { messages: [{ role: 'user', content: 'hi' }] });
    expect(result.text).toBe('ok');
  });

  it('DEEPSEEK_CONTEXT_WINDOW + DEEPSEEK_PRICE_* overrides still apply', async () => {
    process.env.DEEPSEEK_CONTEXT_WINDOW = '98765';
    process.env.DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK = '1.25';
    process.env.DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK = '2.5';
    process.env.DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK = '3.75';
    expect(await getContextWindowTokens('deepseek', 'deepseek-v4-flash')).toBe(98765);
    const pricing = await getModelPricing('deepseek', 'deepseek-v4-flash');
    expect(pricing).toEqual({
      currency: 'CNY',
      cacheHitPerM: 1.25,
      cacheMissPerM: 2.5,
      outputPerM: 3.75,
    });
  });

  it('OPENROUTER_SITE_URL + OPENROUTER_APP_NAME + OPENROUTER_PROVIDER_SORT reach the wire', async () => {
    let seen: { headers: any; body: any } | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((req: any, res: any) => {
        if (req.url?.endsWith('/models')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ data: [] }));
          return;
        }
        let body = '';
        req.on('data', (d: Buffer) => (body += d.toString()));
        req.on('end', () => {
          seen = { headers: req.headers, body: JSON.parse(body) };
          serveChat(res);
        });
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.OPENROUTER_API_KEY = 'or-key';
    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_SITE_URL = 'https://zed.dev';
    process.env.OPENROUTER_APP_NAME = 'Zen Agent';
    process.env.OPENROUTER_PROVIDER_SORT = 'throughput';

    await runLlmStep('openrouter', { messages: [{ role: 'user', content: 'hi' }] });
    expect(seen?.headers['http-referer']).toBe('https://zed.dev');
    expect(seen?.headers['x-title']).toBe('Zen Agent');
    expect(seen?.body.provider).toEqual({ sort: 'throughput' });
  });

  it('OPENROUTER_PROVIDER_SORT="" disables the provider block', async () => {
    let seenBody: any;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((req: any, res: any) => {
        if (req.url?.endsWith('/models')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ data: [] }));
          return;
        }
        let body = '';
        req.on('data', (d: Buffer) => (body += d.toString()));
        req.on('end', () => {
          seenBody = JSON.parse(body);
          serveChat(res);
        });
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.OPENROUTER_API_KEY = 'or-key';
    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, port)}/api/v1`;
    process.env.OPENROUTER_PROVIDER_SORT = '';

    await runLlmStep('openrouter', { messages: [{ role: 'user', content: 'hi' }] });
    expect(seenBody.provider).toBeUndefined();
  });

  it('balance fetch still honors DEEPSEEK_BASE_URL and OPENROUTER_BASE_URL', async () => {
    // DeepSeek balance
    const dsPort = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((_req: any, res: any) => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            is_available: true,
            balance_infos: [{ currency: 'CNY', total_balance: '55.00' }],
          }),
        );
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.DEEPSEEK_API_KEY = 'k';
    process.env.DEEPSEEK_BASE_URL = testServerBaseUrl(server, dsPort);
    const ds = await fetchBalanceSnapshot('deepseek');
    expect(ds).toMatchObject({ currency: 'CNY', total: 55 });

    // OpenRouter balance
    const orPort = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer((_req: any, res: any) => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ data: { usage: 1, limit: 10 } }));
      });
      server = srv;
      srv.listen(0, () => resolve((srv.address() as any).port));
    });
    process.env.OPENROUTER_API_KEY = 'k';
    process.env.OPENROUTER_BASE_URL = `${testServerBaseUrl(server, orPort)}/api/v1`;
    const or = await fetchBalanceSnapshot('openrouter');
    expect(or).toMatchObject({ currency: 'USD', total: 9 });
  });

  it('thinking selector env-independent: deepseek vocabulary unchanged', async () => {
    expect(await getThinkingEfforts('deepseek', 'deepseek-v4-flash')).toEqual([
      'off',
      'low',
      'high',
      'max',
    ]);
  });
});
