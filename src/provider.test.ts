import { afterEach, describe, expect, it } from "vitest";
import {
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getDefaultModel,
  getModelPricing,
  getProvider,
} from "./provider.js";
import { DEFAULT_OPENROUTER_MODEL } from "./openrouter.js";
import { resetOpenRouterModelsCache } from "./openrouter.js";

describe("getProvider", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("defaults to deepseek", () => {
    delete process.env.ZEN_AGENT_LLM_PROVIDER;
    expect(getProvider()).toBe("deepseek");
  });

  it("selects openrouter", () => {
    process.env.ZEN_AGENT_LLM_PROVIDER = "openrouter";
    expect(getProvider()).toBe("openrouter");
  });

  it("throws on an unknown value", () => {
    process.env.ZEN_AGENT_LLM_PROVIDER = "nope";
    expect(() => getProvider()).toThrow(/ZEN_AGENT_LLM_PROVIDER/);
  });
});

describe("getDefaultModel", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns the DeepSeek fallback", () => {
    expect(getDefaultModel("deepseek")).toBe("deepseek-v4-flash");
  });

  it("returns ZEN_AGENT_OPENROUTER_MODEL when set, else the curated default", () => {
    delete process.env.ZEN_AGENT_OPENROUTER_MODEL;
    expect(getDefaultModel("openrouter")).toBe(DEFAULT_OPENROUTER_MODEL);
    process.env.ZEN_AGENT_OPENROUTER_MODEL = "openai/gpt-5";
    expect(getDefaultModel("openrouter")).toBe("openai/gpt-5");
  });
});

describe("getModelPricing", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
  });

  it("returns CNY DeepSeek pricing (off-peak flash)", async () => {
    // 2026-08-19 04:00 UTC = 12:00 Beijing = off-peak.
    const offPeak = new Date(Date.UTC(2026, 7, 19, 4));
    const pricing = await getModelPricing("deepseek", "deepseek-v4-flash", offPeak);
    expect(pricing).toEqual({
      currency: "CNY",
      cacheHitPerM: 0.05,
      cacheMissPerM: 1.5,
      outputPerM: 4.5,
    });
  });

  it("returns USD OpenRouter pricing from the fallback table without network", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    const pricing = await getModelPricing("openrouter", "anthropic/claude-sonnet-4");
    // OpenRouter bills cached reads at the regular input rate.
    expect(pricing).toEqual({
      currency: "USD",
      cacheHitPerM: 3,
      cacheMissPerM: 3,
      outputPerM: 15,
    });
  });
});

describe("getContextWindowTokens", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
  });

  it("uses DEEPSEEK_CONTEXT_WINDOW for deepseek", async () => {
    process.env.DEEPSEEK_CONTEXT_WINDOW = "123456";
    expect(await getContextWindowTokens("deepseek", "deepseek-v4-flash")).toBe(123456);
  });

  it("uses the model's context length for openrouter", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    expect(await getContextWindowTokens("openrouter", "google/gemini-2.5-pro")).toBe(1_000_000);
  });
});

describe("fetchBalanceSnapshot", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("throws with a provider hint when the provider's key is missing", async () => {
    delete process.env.DEEPSEEK_API_KEY;
    await expect(fetchBalanceSnapshot("deepseek")).rejects.toThrow(/DEEPSEEK_API_KEY/);
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    await expect(fetchBalanceSnapshot("openrouter")).rejects.toThrow(
      /ZEN_AGENT_OPENROUTER_API_KEY/,
    );
  });
});
