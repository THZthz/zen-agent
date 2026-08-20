import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  costYuan,
  getModelPricing,
  isPeakTime,
  parseDeepSeekUsage,
  runLlmStep,
} from "./deepseek.js";

const timing = { llmMs: 5000, thinkingMs: 2000, answeringMs: 3000 };

describe("parseDeepSeekUsage", () => {
  it("reads DeepSeek-style cache tokens and reasoning tokens from raw usage", () => {
    const usage = parseDeepSeekUsage(
      {
        prompt_tokens: 10000,
        completion_tokens: 500,
        total_tokens: 10500,
        prompt_cache_hit_tokens: 8000,
        prompt_cache_miss_tokens: 2000,
        completion_tokens_details: { reasoning_tokens: 200 },
      },
      timing,
    );

    expect(usage).not.toBeNull();
    expect(usage?.inputTokens).toBe(10000);
    expect(usage?.outputTokens).toBe(500);
    expect(usage?.totalTokens).toBe(10500);
    expect(usage?.cacheReadTokens).toBe(8000);
    expect(usage?.cacheMissTokens).toBe(2000);
    expect(usage?.reasoningTokens).toBe(200);
    expect(usage?.llmMs).toBe(5000);
    expect(usage?.thinkingMs).toBe(2000);
    expect(usage?.answeringMs).toBe(3000);
  });

  it("falls back to input-minus-cache for miss and sums totals when absent", () => {
    const usage = parseDeepSeekUsage(
      { prompt_tokens: 10000, completion_tokens: 100, prompt_cache_hit_tokens: 4000 },
      timing,
    );
    expect(usage?.cacheReadTokens).toBe(4000);
    expect(usage?.cacheMissTokens).toBe(6000);
    expect(usage?.totalTokens).toBe(10100);
  });

  it("returns null when no token counts are reported", () => {
    expect(parseDeepSeekUsage(undefined, timing)).toBeNull();
    expect(parseDeepSeekUsage({}, timing)).toBeNull();
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

describe("runLlmStep (live SSE, no AI SDK)", () => {
  const originalEnv = { ...process.env };
  let server: import("node:http").Server | undefined;

  const makeChunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
    `data: ${JSON.stringify({
      id: "1",
      object: "chat.completion.chunk",
      created: 1,
      model: "deepseek",
      choices: [{ index: 0, delta, finish_reason: null }],
      ...extra,
    })}\n\n`;

  function startServer(handler: (res: import("node:http").ServerResponse) => void): Promise<number> {
    return new Promise((resolve) => {
      const srv = require("node:http").createServer((_req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
        res.writeHead(200, { "content-type": "text/event-stream" });
        handler(res);
      });
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });
  }

  beforeEach(() => {
    process.env.DEEPSEEK_API_KEY = "test";
    process.env.DEEPSEEK_MODEL = "deepseek-v4-flash";
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
    vi.restoreAllMocks();
  });

  it("streams reasoning_content LIVE (before the answer starts) and reads cache tokens", async () => {
    const port = await startServer((res) => {
      res.write(makeChunk({ role: "assistant", content: "" }));
      setTimeout(() => {
        res.write(makeChunk({ reasoning_content: "Let me think. " }));
      }, 150);
      setTimeout(() => {
        res.write(makeChunk({ reasoning_content: "Second thought." }));
      }, 300);
      setTimeout(() => {
        res.write(makeChunk({ content: "The answer." }));
      }, 450);
      setTimeout(() => {
        res.write(
          makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
        );
        res.write(
          makeChunk(
            {},
            {
              choices: [],
              usage: {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
                prompt_cache_hit_tokens: 90,
                prompt_cache_miss_tokens: 10,
                completion_tokens_details: { reasoning_tokens: 8 },
              },
            },
          ),
        );
        res.write("data: [DONE]\n\n");
        res.end();
      }, 600);
    });

    process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${port}`;
    const t0 = Date.now();
    const reasoningArrivals: number[] = [];
    const textArrivals: number[] = [];
    const result = await runLlmStep({
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      model: "deepseek-v4-flash",
      thinkingEffort: "max",
      onReasoningDelta: async () => { reasoningArrivals.push(Date.now() - t0); },
      onTextDelta: async () => { textArrivals.push(Date.now() - t0); },
    });

    expect(result.text).toBe("The answer.");
    expect(result.reasoning).toBe("Let me think. Second thought.");
    expect(result.finishReason).toBe("stop");
    // Reasoning arrived BEFORE the answer started (450ms) — the AI SDK
    // buffered everything to ~600ms; our direct client must not.
    expect(reasoningArrivals.length).toBe(2);
    expect(reasoningArrivals[0]).toBeLessThan(450);
    expect(reasoningArrivals[1]).toBeGreaterThan(150);
    expect(textArrivals[0]).toBeGreaterThanOrEqual(450);
    // Raw cache tokens survive parsing.
    expect(result.usage?.cacheReadTokens).toBe(90);
    expect(result.usage?.cacheMissTokens).toBe(10);
    expect(result.usage?.reasoningTokens).toBe(8);
  });

  it("accumulates streaming tool call fragments", async () => {
    const port = await startServer((res) => {
      res.write(
        makeChunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "bash", arguments: '{"command":' },
            },
          ],
        }),
      );
      res.write(makeChunk({ tool_calls: [{ index: 0, function: { arguments: '"echo hi"}' } }] }));
      res.write(
        makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] }),
      );
      res.write("data: [DONE]\n\n");
      res.end();
    });

    process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${port}`;
    const result = await runLlmStep({
      messages: [{ role: "user", content: "run" }],
      model: "deepseek-v4-flash",
      thinkingEffort: "off",
    });

    expect(result.finishReason).toBe("tool-calls");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]).toEqual({
      id: "call_1",
      name: "bash",
      input: { command: "echo hi" },
    });
    expect(result.text).toBe("");
  });

  it("converts stored tool-result history to the wire format (assistant+tool messages)", async () => {
    let requestBody: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer((req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
        let body = "";
        req.on("data", (d: Buffer) => (body += d.toString()));
        req.on("end", () => {
          requestBody = JSON.parse(body);
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.write(makeChunk({ content: "ok" }));
          res.write(
            makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
          );
          res.write("data: [DONE]\n\n");
          res.end();
        });
      });
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${port}`;
    const result = await runLlmStep({
      messages: [
        { role: "user", content: "check" },
        {
          role: "assistant",
          content: [
            { type: "reasoning", text: "think hard" },
            { type: "text", text: "running" },
            { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "ls" } },
          ],
        },
        {
          role: "tool",
          content: [
            { type: "tool-result", toolCallId: "c1", toolName: "bash", output: { type: "text", value: "file.txt" } },
          ],
        },
      ],
      model: "deepseek-v4-flash",
      thinkingEffort: "off",
      system: "custom system",
    });

    expect(result.text).toBe("ok");
    const messages = requestBody?.messages as Array<Record<string, unknown>>;
    expect(messages).toHaveLength(4);
    expect(messages[0]).toEqual({
      role: "system",
      content: "custom system",
    });
    expect(messages[2]).toMatchObject({
      role: "assistant",
      content: "running",
      reasoning_content: "think hard",
      tool_calls: [
        {
          id: "c1",
          type: "function",
          function: { name: "bash", arguments: '{"command":"ls"}' },
        },
      ],
    });
    expect(messages[3]).toEqual({
      role: "tool",
      tool_call_id: "c1",
      content: "file.txt",
    });
    expect(requestBody?.reasoning_effort).toBeUndefined();
  });

  it("sends reasoning_effort when thinking is enabled", async () => {
    let requestBody: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer((req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
        let body = "";
        req.on("data", (d: Buffer) => (body += d.toString()));
        req.on("end", () => {
          requestBody = JSON.parse(body);
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.write(makeChunk({ content: "ok" }));
          res.write(
            makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
          );
          res.write("data: [DONE]\n\n");
          res.end();
        });
      });
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${port}`;
    await runLlmStep({
      messages: [{ role: "user", content: "hi" }],
      model: "deepseek-v4-flash",
      thinkingEffort: "high",
    });

    expect(requestBody?.reasoning_effort).toBe("high");
    expect(requestBody?.stream).toBe(true);
    expect((requestBody?.tools as unknown[]).length).toBe(1);
  });

  it("handles CRLF-delimited SSE events (some servers use \r\n)", async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (_req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          res.writeHead(200, { "content-type": "text/event-stream" });
          const chunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
            `data: ${JSON.stringify({
              id: "1",
              object: "chat.completion.chunk",
              choices: [{ index: 0, delta, finish_reason: null }],
              ...extra,
            })}\r\n\r\n`;
          res.write(chunk({ role: "assistant", content: "" }));
          res.write(chunk({ reasoning_content: "think" }));
          res.write(chunk({ content: "done" }));
          res.write(chunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }));
          res.write("data: [DONE]\r\n\r\n");
          res.end();
        },
      );
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.DEEPSEEK_BASE_URL = `http://127.0.0.1:${port}`;
    const reasoning: string[] = [];
    const result = await runLlmStep({
      messages: [{ role: "user", content: "hi" }],
      model: "deepseek-v4-flash",
      thinkingEffort: "off",
      onReasoningDelta: async (d) => {
        reasoning.push(d);
      },
    });

    expect(reasoning).toEqual(["think"]);
    expect(result.reasoning).toBe("think");
    expect(result.text).toBe("done");
    expect(result.finishReason).toBe("stop");
  });
});
