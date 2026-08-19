import { describe, expect, it } from "vitest";
import {
  costYuan,
  extractUsage,
  getModelPricing,
  isPeakTime,
} from "./deepseek.js";

const timing = { llmMs: 5000, thinkingMs: 2000, answeringMs: 3000 };

describe("extractUsage", () => {
  it("reads DeepSeek-style cache tokens from raw usage", () => {
    const usage = extractUsage(
      {
        inputTokens: 10000,
        outputTokens: 500,
        totalTokens: 10500,
        inputTokenDetails: { noCacheTokens: undefined, cacheReadTokens: undefined, cacheWriteTokens: undefined },
        outputTokenDetails: { textTokens: undefined, reasoningTokens: 200 },
        raw: {
          prompt_cache_hit_tokens: 8000,
          prompt_cache_miss_tokens: 2000,
          completion_tokens_details: { reasoning_tokens: 200 },
        },
      },
      timing,
    );

    expect(usage).not.toBeNull();
    expect(usage?.inputTokens).toBe(10000);
    expect(usage?.outputTokens).toBe(500);
    expect(usage?.cacheReadTokens).toBe(8000);
    expect(usage?.cacheMissTokens).toBe(2000);
    expect(usage?.reasoningTokens).toBe(200);
    expect(usage?.llmMs).toBe(5000);
    expect(usage?.thinkingMs).toBe(2000);
    expect(usage?.answeringMs).toBe(3000);
  });

  it("falls back to inputTokenDetails.cacheReadTokens and input-minus-cache for miss", () => {
    const usage = extractUsage(
      {
        inputTokens: 10000,
        outputTokens: 100,
        totalTokens: 10100,
        inputTokenDetails: { noCacheTokens: 6000, cacheReadTokens: 4000, cacheWriteTokens: 0 },
        outputTokenDetails: { textTokens: 100, reasoningTokens: 0 },
        raw: {},
      },
      timing,
    );

    expect(usage?.cacheReadTokens).toBe(4000);
    expect(usage?.cacheMissTokens).toBe(6000);
  });

  it("returns null when no token counts are reported", () => {
    const usage = extractUsage(
      {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        inputTokenDetails: { noCacheTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 },
        outputTokenDetails: { textTokens: 0, reasoningTokens: 0 },
        raw: {},
      },
      timing,
    );
    expect(usage).toBeNull();
  });
});

describe("isPeakTime", () => {
  // Beijing time = UTC+8, no DST.
  const atBeijing = (h: number, m = 0) =>
    new Date(Date.UTC(2026, 7, 19, h - 8, m));

  it("treats 09:00-11:59 Beijing as peak", () => {
    expect(isPeakTime(atBeijing(9, 0))).toBe(true);
    expect(isPeakTime(atBeijing(10, 30))).toBe(true);
    expect(isPeakTime(atBeijing(11, 59))).toBe(true);
  });

  it("treats 12:00-13:59 Beijing as off-peak", () => {
    expect(isPeakTime(atBeijing(12, 0))).toBe(false);
    expect(isPeakTime(atBeijing(13, 59))).toBe(false);
  });

  it("treats 14:00-17:59 Beijing as peak", () => {
    expect(isPeakTime(atBeijing(14, 0))).toBe(true);
    expect(isPeakTime(atBeijing(17, 59))).toBe(true);
  });

  it("treats 18:00-08:59 Beijing as off-peak", () => {
    expect(isPeakTime(atBeijing(18, 0))).toBe(false);
    expect(isPeakTime(atBeijing(0, 0))).toBe(false);
    expect(isPeakTime(atBeijing(8, 59))).toBe(false);
  });
});

describe("costYuan (official DeepSeek V4 pricing, CNY per 1M tokens)", () => {
  // Off-peak: flash 0.05 hit / 1.5 miss / 4.5 out
  // Peak:     flash 0.10 hit / 3.0 miss / 9.0 out
  const usage = {
    inputTokens: 1_000_000,
    outputTokens: 1_000_000,
    totalTokens: 2_000_000,
    cacheReadTokens: 800_000,
    cacheMissTokens: 200_000,
    reasoningTokens: 0,
    llmMs: 0,
    thinkingMs: 0,
    answeringMs: 0,
  };

  it("flash off-peak: 0.8M hit*0.05 + 0.2M miss*1.5 + 1M out*4.5 = 4.84", () => {
    const offPeak = new Date(Date.UTC(2026, 7, 19, 4)); // 12:00 Beijing
    const pricing = getModelPricing("deepseek-v4-flash", offPeak);
    expect(pricing).toEqual({
      cacheHitCnyPerM: 0.05,
      cacheMissCnyPerM: 1.5,
      outputCnyPerM: 4.5,
    });
    expect(costYuan(usage, pricing)).toBeCloseTo(4.84, 5);
  });

  it("flash peak: 0.8M hit*0.10 + 0.2M miss*3.0 + 1M out*9.0 = 9.68", () => {
    const peak = new Date(Date.UTC(2026, 7, 19, 2)); // 10:00 Beijing
    const pricing = getModelPricing("deepseek-v4-flash", peak);
    expect(pricing).toEqual({
      cacheHitCnyPerM: 0.1,
      cacheMissCnyPerM: 3.0,
      outputCnyPerM: 9.0,
    });
    expect(costYuan(usage, pricing)).toBeCloseTo(9.68, 5);
  });

  it("pro off-peak: 1M hit*0.15 + 1M out*13.5 = 13.65", () => {
    const offPeak = new Date(Date.UTC(2026, 7, 19, 4));
    const usagePro = { ...usage, cacheReadTokens: 1_000_000, cacheMissTokens: 0 };
    const pricing = getModelPricing("deepseek-v4-pro", offPeak);
    expect(pricing).toEqual({
      cacheHitCnyPerM: 0.15,
      cacheMissCnyPerM: 4.5,
      outputCnyPerM: 13.5,
    });
    expect(costYuan(usagePro, pricing)).toBeCloseTo(13.65, 5);
  });

  it("pro peak: 1M hit*0.30 + 1M out*27.0 = 27.30", () => {
    const peak = new Date(Date.UTC(2026, 7, 19, 2));
    const usagePro = { ...usage, cacheReadTokens: 1_000_000, cacheMissTokens: 0 };
    const pricing = getModelPricing("deepseek-v4-pro", peak);
    expect(pricing).toEqual({
      cacheHitCnyPerM: 0.3,
      cacheMissCnyPerM: 9.0,
      outputCnyPerM: 27.0,
    });
    expect(costYuan(usagePro, pricing)).toBeCloseTo(27.3, 5);
  });
});
