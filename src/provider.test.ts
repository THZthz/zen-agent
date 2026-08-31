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
  isPeakTime,
  runLlmStep,
} from './provider.js';
import { resetPiModels } from './provider-pi.js';
import { resetCatalogCache } from './provider-catalog.js';
import { resetProviderRegistry } from './provider-registry.js';

const originalEnv = { ...process.env };
let xdgHome: string;

beforeEach(() => {
  xdgHome = mkdtempSync(join(tmpdir(), 'zen-agent-provider-test-'));
  process.env.XDG_DATA_HOME = xdgHome;
});

afterEach(() => {
  process.env = { ...originalEnv };
  resetProviderRegistry();
  resetPiModels();
  resetCatalogCache();
  rmSync(xdgHome, { recursive: true, force: true });
});

describe('runLlmStep', () => {
  it('dispatches to the requested provider (key checks prove routing)', async () => {
    delete process.env.DEEPSEEK_API_KEY;
    await expect(runLlmStep('deepseek', { messages: [] })).rejects.toThrow(/DEEPSEEK_API_KEY/);
    delete process.env.OPENROUTER_API_KEY;
    await expect(runLlmStep('openrouter', { messages: [] })).rejects.toThrow(/OPENROUTER_API_KEY/);
  });

  it('rejects unknown providers with a configuration hint', async () => {
    await expect(runLlmStep('nope', { messages: [] })).rejects.toThrow(
      /Unknown LLM provider "nope"/,
    );
  });
});

describe('getDefaultModel', () => {
  it('returns the DeepSeek fallback', () => {
    expect(getDefaultModel('deepseek')).toBe('deepseek-v4-flash');
  });

  it('returns OPENROUTER_MODEL when set, else the curated default', () => {
    delete process.env.OPENROUTER_MODEL;
    expect(getDefaultModel('openrouter')).toBe('openrouter/free');
    process.env.OPENROUTER_MODEL = 'openai/gpt-5';
    expect(getDefaultModel('openrouter')).toBe('openai/gpt-5');
  });
});

describe('getProviderCurrency', () => {
  it('returns the provider billing currency', () => {
    expect(getProviderCurrency('deepseek')).toBe('CNY');
    expect(getProviderCurrency('openrouter')).toBe('USD');
  });
});

describe('isPeakTime', () => {
  const atBeijing = (h: number, m = 0) => new Date(Date.UTC(2026, 7, 19, h - 8, m));

  it('treats 09:00-11:59 and 14:00-17:59 Beijing as peak', () => {
    expect(isPeakTime(atBeijing(9, 0))).toBe(true);
    expect(isPeakTime(atBeijing(11, 59))).toBe(true);
    expect(isPeakTime(atBeijing(14, 0))).toBe(true);
    expect(isPeakTime(atBeijing(17, 59))).toBe(true);
  });

  it('treats 12:00-13:59 and 18:00-08:59 Beijing as off-peak', () => {
    expect(isPeakTime(atBeijing(12, 0))).toBe(false);
    expect(isPeakTime(atBeijing(13, 59))).toBe(false);
    expect(isPeakTime(atBeijing(18, 0))).toBe(false);
    expect(isPeakTime(atBeijing(8, 59))).toBe(false);
  });
});

describe('getModelPricing', () => {
  it('returns CNY DeepSeek pricing honoring peak/off-peak windows', async () => {
    // 2026-08-19 04:00 UTC = 12:00 Beijing = off-peak.
    const offPeak = new Date(Date.UTC(2026, 7, 19, 4));
    const pricing = await getModelPricing('deepseek', 'deepseek-v4-flash', offPeak);
    expect(pricing).toEqual({
      currency: 'CNY',
      cacheHitPerM: 0.05,
      cacheMissPerM: 1.5,
      outputPerM: 4.5,
    });
    // 10:00 Beijing = peak.
    const peak = new Date(Date.UTC(2026, 7, 19, 2));
    const peakPricing = await getModelPricing('deepseek', 'deepseek-v4-flash', peak);
    expect(peakPricing).toEqual({
      currency: 'CNY',
      cacheHitPerM: 0.1,
      cacheMissPerM: 3.0,
      outputPerM: 9.0,
    });
  });

  it('returns USD OpenRouter pricing from the static fallback without network', async () => {
    delete process.env.OPENROUTER_API_KEY;
    const pricing = await getModelPricing('openrouter', 'openrouter/free');
    // openrouter/free routes to free models: billed at $0.
    expect(pricing).toEqual({
      currency: 'USD',
      cacheHitPerM: 0,
      cacheMissPerM: 0,
      outputPerM: 0,
    });
  });
});

describe('getContextWindowTokens', () => {
  it('uses DEEPSEEK_CONTEXT_WINDOW for deepseek', async () => {
    process.env.DEEPSEEK_CONTEXT_WINDOW = '123456';
    expect(await getContextWindowTokens('deepseek', 'deepseek-v4-flash')).toBe(123456);
  });

  it("uses the model's context length for openrouter", async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getContextWindowTokens('openrouter', 'openrouter/free')).toBe(128_000);
  });
});

describe('getThinkingEfforts', () => {
  it('offers DeepSeek vocabulary for deepseek models', async () => {
    expect(await getThinkingEfforts('deepseek', 'deepseek-v4-flash')).toEqual([
      'off',
      'low',
      'high',
      'max',
    ]);
  });

  it('offers the full ladder for unknown openrouter models', async () => {
    delete process.env.OPENROUTER_API_KEY;
    expect(await getThinkingEfforts('openrouter', 'openrouter/free')).toEqual([
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

describe('fetchBalanceSnapshot', () => {
  it("throws with a provider hint when the provider's key is missing", async () => {
    delete process.env.DEEPSEEK_API_KEY;
    await expect(fetchBalanceSnapshot('deepseek')).rejects.toThrow(/DEEPSEEK_API_KEY/);
    delete process.env.OPENROUTER_API_KEY;
    await expect(fetchBalanceSnapshot('openrouter')).rejects.toThrow(/OPENROUTER_API_KEY/);
  });

  it('returns an unavailable snapshot for providers without a balance endpoint', async () => {
    process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
      {
        id: 'local',
        baseUrl: 'http://127.0.0.1:9999/v1',
        defaultModel: 'm',
        models: ['m'],
      },
    ]);
    const snapshot = await fetchBalanceSnapshot('local');
    expect(snapshot).toEqual({
      isAvailable: false,
      currency: 'USD',
      total: 0,
      details: {},
    });
  });
});
