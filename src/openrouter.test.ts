import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_OPENROUTER_MODEL,
  fetchOpenRouterBalance,
  getOpenRouterModelInfo,
  parseOpenRouterUsage,
  resetOpenRouterModelsCache,
  runOpenRouterStep,
} from "./openrouter.js";

const timing = { llmMs: 5000, thinkingMs: 2000, answeringMs: 3000 };

describe("parseOpenRouterUsage", () => {
  it("reads generic OpenAI usage and passthrough cache/reasoning fields", () => {
    const usage = parseOpenRouterUsage(
      {
        prompt_tokens: 10000,
        completion_tokens: 500,
        total_tokens: 10500,
        prompt_cache_hit_tokens: 8000,
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
  });

  it("reads Anthropic-style prompt_tokens_details.cached_tokens", () => {
    const usage = parseOpenRouterUsage(
      {
        prompt_tokens: 1000,
        completion_tokens: 100,
        prompt_tokens_details: { cached_tokens: 700 },
      },
      timing,
    );
    expect(usage?.cacheReadTokens).toBe(700);
    expect(usage?.cacheMissTokens).toBe(300);
  });

  it("returns null when no token counts are reported", () => {
    expect(parseOpenRouterUsage(undefined, timing)).toBeNull();
    expect(parseOpenRouterUsage({}, timing)).toBeNull();
  });
});

describe("runOpenRouterStep (live SSE)", () => {
  const originalEnv = { ...process.env };
  let server: import("node:http").Server | undefined;

  const makeChunk = (delta: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
    `data: ${JSON.stringify({
      id: "1",
      object: "chat.completion.chunk",
      created: 1,
      model: "or-model",
      choices: [{ index: 0, delta, finish_reason: null }],
      ...extra,
    })}\n\n`;

  beforeEach(() => {
    process.env.ZEN_AGENT_OPENROUTER_API_KEY = "test";
    delete process.env.ZEN_AGENT_OPENROUTER_MODEL;
    delete process.env.ZEN_AGENT_OPENROUTER_SITE_URL;
    delete process.env.ZEN_AGENT_OPENROUTER_APP_NAME;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
  });

  it("streams reasoning (delta.reasoning) LIVE and parses the final usage chunk", async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (_req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.write(makeChunk({ role: "assistant", content: "" }));
          setTimeout(() => {
            res.write(makeChunk({ reasoning: "Let me think. " }));
          }, 150);
          setTimeout(() => {
            res.write(makeChunk({ reasoning: "Second thought." }));
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
                    completion_tokens_details: { reasoning_tokens: 8 },
                  },
                },
              ),
            );
            res.write("data: [DONE]\n\n");
            res.end();
          }, 600);
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const t0 = Date.now();
    const reasoningArrivals: number[] = [];
    const result = await runOpenRouterStep({
      messages: [{ role: "user", content: "hi" }],
      system: "sys",
      thinkingEffort: "high",
      onReasoningDelta: async () => {
        reasoningArrivals.push(Date.now() - t0);
      },
    });

    expect(result.text).toBe("The answer.");
    expect(result.reasoning).toBe("Let me think. Second thought.");
    expect(result.finishReason).toBe("stop");
    // Reasoning arrived BEFORE the answer started (450ms) — live streaming,
    // not a buffered burst.
    expect(reasoningArrivals.length).toBe(2);
    expect(reasoningArrivals[0]).toBeLessThan(450);
    // Usage came from the final chunk (no cache fields reported → 0).
    expect(result.usage?.inputTokens).toBe(100);
    expect(result.usage?.outputTokens).toBe(20);
    expect(result.usage?.cacheReadTokens).toBe(0);
    expect(result.usage?.reasoningTokens).toBe(8);
  });

  it("accepts delta.reasoning_content passthrough (DeepSeek routes)", async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (_req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.write(makeChunk({ reasoning_content: "thinking via passthrough" }));
          res.write(makeChunk({ content: "done" }));
          res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }));
          res.write("data: [DONE]\n\n");
          res.end();
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const result = await runOpenRouterStep({
      messages: [{ role: "user", content: "hi" }],
      thinkingEffort: "off",
    });
    expect(result.reasoning).toBe("thinking via passthrough");
    expect(result.text).toBe("done");
  });

  it("requests include_usage, maps max→high, and omits reasoning_effort for off", async () => {
    let bodies: Array<Record<string, unknown>> = [];
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          const chunks: Buffer[] = [];
          req.on("data", (c: Buffer) => chunks.push(c));
          req.on("end", () => {
            bodies.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
            res.writeHead(200, { "content-type": "text/event-stream" });
            res.write(makeChunk({ content: "ok" }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }));
            res.write("data: [DONE]\n\n");
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;

    await runOpenRouterStep({ messages: [{ role: "user", content: "hi" }], thinkingEffort: "max" });
    await runOpenRouterStep({ messages: [{ role: "user", content: "hi" }], thinkingEffort: "off" });

    expect(bodies).toHaveLength(2);
    expect(bodies[0]?.reasoning_effort).toBe("high");
    expect(bodies[0]?.stream_options).toEqual({ include_usage: true });
    expect(bodies[0]?.stream).toBe(true);
    expect((bodies[0]?.tools as unknown[]).length).toBe(1);
    expect(bodies[0]?.model).toBe(DEFAULT_OPENROUTER_MODEL);
    expect(bodies[1]?.reasoning_effort).toBeUndefined();
  });

  it("sends HTTP-Referer and X-Title when configured, and honors ZEN_AGENT_OPENROUTER_MODEL", async () => {
    let headers: import("node:http").IncomingHttpHeaders | undefined;
    let body: Record<string, unknown> | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          headers = req.headers;
          const chunks: Buffer[] = [];
          req.on("data", (c: Buffer) => chunks.push(c));
          req.on("end", () => {
            body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
            res.writeHead(200, { "content-type": "text/event-stream" });
            res.write(makeChunk({ content: "ok" }));
            res.write(makeChunk({}, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }));
            res.write("data: [DONE]\n\n");
            res.end();
          });
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    process.env.ZEN_AGENT_OPENROUTER_SITE_URL = "https://zed.dev";
    process.env.ZEN_AGENT_OPENROUTER_APP_NAME = "Zen Agent";
    process.env.ZEN_AGENT_OPENROUTER_MODEL = "openai/gpt-5";

    await runOpenRouterStep({ messages: [{ role: "user", content: "hi" }], thinkingEffort: "off" });

    expect(headers?.["http-referer"]).toBe("https://zed.dev");
    expect(headers?.["x-title"]).toBe("Zen Agent");
    expect(body?.model).toBe("openai/gpt-5");
  });

  it("requires ZEN_AGENT_OPENROUTER_API_KEY", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    await expect(
      runOpenRouterStep({ messages: [{ role: "user", content: "hi" }] }),
    ).rejects.toThrow(/ZEN_AGENT_OPENROUTER_API_KEY/);
  });
});

describe("getOpenRouterModelInfo", () => {
  const originalEnv = { ...process.env };
  let server: import("node:http").Server | undefined;

  afterEach(() => {
    process.env = { ...originalEnv };
    resetOpenRouterModelsCache();
    server?.close();
    server = undefined;
  });

  it("falls back to the static table without an API key", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    const info = await getOpenRouterModelInfo("openrouter/free");
    expect(info).toEqual({ inputPerM: 0, outputPerM: 0, contextLength: 128_000 });
  });

  it("falls back to generic defaults for unknown models", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    const info = await getOpenRouterModelInfo("some/unknown-model");
    expect(info.contextLength).toBe(200_000);
    expect(info.inputPerM).toBeGreaterThan(0);
    expect(info.outputPerM).toBeGreaterThan(0);
  });

  it("prefers the live /models catalog when available", async () => {
    const port = await new Promise<number>((resolve) => {
      const srv = require("node:http").createServer(
        (_req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify({
              data: [
                {
                  id: "vendor/model",
                  context_length: 123456,
                  pricing: { prompt: "0.5", completion: "1.5" },
                },
              ],
            }),
          );
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });

    process.env.ZEN_AGENT_OPENROUTER_API_KEY = "test";
    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const info = await getOpenRouterModelInfo("vendor/model");
    expect(info).toEqual({ inputPerM: 0.5, outputPerM: 1.5, contextLength: 123456 });
  });
});

describe("fetchOpenRouterBalance", () => {
  const originalEnv = { ...process.env };
  let server: import("node:http").Server | undefined;

  function startServer(
    handler: (
      req: import("node:http").IncomingMessage,
      res: import("node:http").ServerResponse,
    ) => void,
  ): Promise<number> {
    return new Promise((resolve) => {
      const srv = require("node:http").createServer(handler);
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import("node:net").AddressInfo;
        resolve(addr.port);
      });
    });
  }

  beforeEach(() => {
    process.env.ZEN_AGENT_OPENROUTER_API_KEY = "test";
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.close();
    server = undefined;
  });

  it("reads key usage and limit as USD credits", async () => {
    let authorization: string | undefined;
    const port = await startServer((req, res) => {
      authorization = req.headers.authorization;
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          data: { label: "test", usage: 4.2, limit: 100, is_free_tier: false },
        }),
      );
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    const balance = await fetchOpenRouterBalance();

    expect(authorization).toBe("Bearer test");
    expect(balance).toEqual({
      isAvailable: true,
      currency: "USD",
      remainingUsd: 95.8,
      usageUsd: 4.2,
      limitUsd: 100,
      isFreeTier: false,
    });
  });

  it("throws on a non-OK response", async () => {
    const port = await startServer((_req, res) => {
      res.writeHead(401);
      res.end("{\"error\":{\"message\":\"invalid key\"}}");
    });

    process.env.ZEN_AGENT_OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
    await expect(fetchOpenRouterBalance()).rejects.toThrow(/401/);
  });

  it("throws without ZEN_AGENT_OPENROUTER_API_KEY", async () => {
    delete process.env.ZEN_AGENT_OPENROUTER_API_KEY;
    await expect(fetchOpenRouterBalance()).rejects.toThrow(/ZEN_AGENT_OPENROUTER_API_KEY/);
  });
});
