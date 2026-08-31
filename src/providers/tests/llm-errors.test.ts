import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { formatLlmError } from '../llm-errors.js';
import { resetCatalogCache } from '../catalog.js';
import { resetPiModels } from '../pi.js';
import { resetProviderRegistry } from '../registry.js';

const originalEnv = { ...process.env };
let xdgHome: string;

function apiError(status: number, body: string, label = 'Groq'): Error {
  return new Error(`${label} API error ${status}: ${body}`);
}

describe('formatLlmError', () => {
  beforeEach(() => {
    xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-llmerrors-test-'));
    process.env.XDG_DATA_HOME = xdgHome;
    process.env.GROQ_API_KEY = 'test';
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'groq',
        name: 'Groq',
        baseUrl: 'https://api.groq.com/openai/v1',
        apiKeyEnv: 'GROQ_API_KEY',
        defaultModel: 'm',
        models: ['m'],
      },
    ]);
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    resetProviderRegistry();
    resetPiModels();
    resetCatalogCache();
    if (xdgHome) {
      rmSync(xdgHome, { recursive: true, force: true });
    }
  });

  it('returns non-API errors unchanged', async () => {
    await expect(formatLlmError(new Error('boom'), { provider: 'groq' })).resolves.toBe('boom');
    await expect(formatLlmError('not an error', { provider: 'groq' })).resolves.toBe(
      'not an error',
    );
  });

  it('points at the declared key env for 401', async () => {
    const msg = await formatLlmError(apiError(401, 'unauthorized'), { provider: 'groq' });
    expect(msg).toContain('GROQ_API_KEY');
  });

  it('explains context overflow with the requested token count', async () => {
    const msg = await formatLlmError(
      apiError(
        400,
        '{"error":{"message":"This model\'s maximum context length is 1000000 tokens. However, you requested 1234567 tokens"}}',
      ),
      { provider: 'groq' },
    );
    expect(msg).toContain('Context window exceeded');
    expect(msg).toContain('1234567');
  });

  it('extracts the inner message for plain 400/422', async () => {
    const bad = await formatLlmError(apiError(400, '{"error":{"message":"Invalid format"}}'), {
      provider: 'groq',
    });
    expect(bad).toBe('Bad request (400): Invalid format');
    const unprocessable = await formatLlmError(
      apiError(422, '{"error":{"message":"Bad parameters"}}'),
      { provider: 'groq' },
    );
    expect(unprocessable).toBe('Request rejected (422): Bad parameters');
  });

  it('mentions the automatic retry on 429', async () => {
    const msg = await formatLlmError(apiError(429, 'rate limit'), { provider: 'groq' });
    expect(msg).toContain('Rate limit');
    expect(msg).toContain('retried automatically');
  });

  it('uses generic wording for 5xx when the provider has no balance endpoint', async () => {
    const msg = await formatLlmError(apiError(500, 'boom'), { provider: 'groq' });
    expect(msg).toContain('Groq returned 500');
    expect(msg).toContain('temporary server error');
  });

  it('leaves unclassified statuses unchanged', async () => {
    const msg = await formatLlmError(apiError(418, 'teapot'), { provider: 'groq' });
    expect(msg).toBe('Groq API error 418: teapot');
  });
});
