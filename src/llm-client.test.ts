import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runChatCompletions } from "./llm-client.js";

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

describe("runChatCompletions", () => {
  beforeEach(() => {
    process.env.ZEN_AGENT_CHAT_TIMEOUT_MS = "100";
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    server?.closeAllConnections?.();
    server?.close();
    server = undefined;
  });

  it("times out a hung request with a clear message", async () => {
    const port = await startServer(() => {
      /* never respond — the stream stalls on reader.read() */
    });
    await expect(
      runChatCompletions({
        baseUrl: `http://127.0.0.1:${port}`,
        apiKey: "test",
        label: "Test",
        model: "test-model",
        messages: [],
        system: "system",
        thinkingEffort: "off",
        reasoningMessageField: "reasoning_content",
        reasoningDeltaFields: ["reasoning_content"],
        parseUsage: () => null,
      }),
    ).rejects.toThrow("timed out after 100ms");
  });
});
