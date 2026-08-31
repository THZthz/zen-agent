import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  CATALOG_CACHE_VERSION,
  fetchCatalog,
  parseCatalogEntry,
  parseInputModalities,
  parsePricePerM,
  readCatalogFile,
  resetCatalogCache,
  setCatalog,
  writeCatalogFile,
} from './provider-catalog.js';
import { resetProviderRegistry } from './provider-registry.js';
import { testServerBaseUrl } from './test-server.js';

const originalEnv = { ...process.env };
let xdgHome: string;
let server: import('node:http').Server | undefined;

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-catalog-test-'));
  process.env.XDG_DATA_HOME = xdgHome;
});

afterEach(() => {
  process.env = { ...originalEnv };
  resetProviderRegistry();
  resetCatalogCache();
  server?.close();
  server = undefined;
  if (xdgHome) {
    rmSync(xdgHome, { recursive: true, force: true });
  }
});

describe('parseInputModalities', () => {
  it('normalizes to a string list and returns null for unknown', () => {
    expect(parseInputModalities(['text', 'image', 'audio'])).toEqual(['text', 'image', 'audio']);
    expect(parseInputModalities(['text', 42, ''])).toEqual(['text']);
    expect(parseInputModalities(undefined)).toBeNull();
    expect(parseInputModalities('image')).toBeNull();
    expect(parseInputModalities([])).toBeNull();
  });
});

describe('parsePricePerM', () => {
  it('converts per-token USD prices to per-1M units', () => {
    expect(parsePricePerM('0.0000005')).toBe(0.5);
    expect(parsePricePerM('0')).toBe(0);
    expect(parsePricePerM(undefined)).toBe(0);
    expect(parsePricePerM('bogus')).toBe(0);
  });
});

describe('parseCatalogEntry', () => {
  it('parses an OpenRouter-style entry with all optional fields', () => {
    const entry = parseCatalogEntry({
      id: 'vendor/model',
      name: 'Vendor Model',
      context_length: 123456,
      pricing: { prompt: '0.0000005', completion: '0.0000015' },
      supported_parameters: ['tools', 'reasoning'],
      architecture: { input_modalities: ['text', 'image', 'audio'] },
      reasoning: { supported_efforts: ['max', 'high', 'low'], default_effort: 'max' },
    });
    expect(entry).toEqual({
      id: 'vendor/model',
      name: 'Vendor Model',
      inputPerM: 0.5,
      outputPerM: 1.5,
      cacheReadPerM: 0,
      contextLength: 123456,
      supportsTools: true,
      inputModalities: ['text', 'image', 'audio'],
    });
  });

  it('degrades unknown fields to conservative defaults', () => {
    const entry = parseCatalogEntry({ id: 'minimal' });
    expect(entry).toEqual({
      id: 'minimal',
      name: null,
      inputPerM: 0,
      outputPerM: 0,
      cacheReadPerM: 0,
      contextLength: 200_000,
      supportsTools: true,
      inputModalities: null,
    });
  });

  it('rejects entries without an id', () => {
    expect(parseCatalogEntry({ name: 'no id' })).toBeNull();
    expect(parseCatalogEntry(null)).toBeNull();
    expect(parseCatalogEntry('nope')).toBeNull();
  });

  it('parses input_cache_read pricing (cache hits billed below input)', () => {
    const entry = parseCatalogEntry({
      id: 'vendor/cache-priced',
      pricing: {
        prompt: '0.000001',
        completion: '0.000003',
        input_cache_read: '0.0000002',
      },
    });
    expect(entry?.inputPerM).toBe(1);
    expect(entry?.outputPerM).toBe(3);
    expect(entry?.cacheReadPerM).toBe(0.2);
  });
});

describe('fetchCatalog', () => {
  it('fetches and parses GET /models with a bearer key', async () => {
    let authorization: string | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          authorization = req.headers.authorization;
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: [{ id: 'a/model', context_length: 1000 }, { id: 'b/model' }],
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

    const catalog = await fetchCatalog(
      testServerBaseUrl(server, port),
      'secret',
      new AbortController().signal,
    );
    expect(authorization).toBe('Bearer secret');
    expect([...catalog.keys()]).toEqual(['a/model', 'b/model']);
    expect(catalog.get('a/model')?.contextLength).toBe(1000);
  });

  it('throws on a non-OK response', async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (_req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          res.writeHead(401);
          res.end('nope');
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    await expect(
      fetchCatalog(testServerBaseUrl(server, port), 'bad', new AbortController().signal),
    ).rejects.toThrow(/401/);
  });
});

describe('catalog file persistence', () => {
  it('round-trips a catalog through the global per-provider file', async () => {
    const catalog = new Map();
    catalog.set('vendor/model', {
      id: 'vendor/model',
      name: 'Vendor Model',
      inputPerM: 1,
      outputPerM: 2,
      contextLength: 1000,
      supportsTools: true,
      inputModalities: ['text'],
      reasoning: {
        supportedEfforts: ['max', 'high', 'low'],
        defaultEffort: 'max',
        mandatory: true,
      },
    });
    await writeCatalogFile('openrouter', catalog);
    const restored = await readCatalogFile('openrouter');
    expect(restored?.get('vendor/model')).toEqual(catalog.get('vendor/model'));
  });

  it('returns null for a missing or outdated file', async () => {
    expect(await readCatalogFile('nope')).toBeNull();
    setCatalog('probe', new Map());
    resetCatalogCache();
    expect(await readCatalogFile('probe')).toBeNull();
  });
});

describe('catalog cache version', () => {
  it('is current so stale caches are discarded', () => {
    expect(CATALOG_CACHE_VERSION).toBe(5);
  });
});
