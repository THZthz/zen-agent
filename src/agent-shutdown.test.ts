import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as acp from "@agentclientprotocol/sdk";

vi.mock("./provider.js", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from "./agent.js";
import { runLlmStep, type LlmStepOptions } from "./provider.js";

const mockedRunLlmStep = vi.mocked(runLlmStep);

describe("ZenAgent.dispose", () => {
  it("aborts the running turn and waits for it to unwind", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-dispose-"));
    try {
      const agent = new ZenAgent();
      await agent.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
      } as acp.InitializeRequest);
      const cx = {
        notify: vi.fn(async () => {}),
        request: vi.fn(() => Promise.reject(new Error("unexpected"))),
      } as unknown as acp.AgentContext;
      const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);

      // A turn whose LLM step only ends when aborted (like a real stream).
      let sawAbort = false;
      mockedRunLlmStep.mockImplementationOnce(async (_provider, options: LlmStepOptions) => {
        const signal = options.signal!;
        if (!signal.aborted) {
          await new Promise<void>((resolve) => {
            signal.addEventListener("abort", () => resolve(), { once: true });
          });
        }
        sawAbort = true;
        throw Object.assign(new Error("aborted"), { name: "AbortError" });
      });

      const promptPromise = agent.prompt(
        { sessionId: created.sessionId, prompt: [{ type: "text", text: "long running" }] },
        cx,
      );
      await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalledTimes(1));

      // Dispose resolves promptly (well under any pathological timeout) and
      // the abort reached the in-flight step.
      await agent.dispose(1_000);
      expect(sawAbort).toBe(true);
      // The interrupted turn resolves as cancelled (prompt() maps an abort
      // to a cancelled response), and the session is idle afterwards.
      await expect(promptPromise).resolves.toEqual({ stopReason: "cancelled" });

      const active = (
        agent as unknown as {
          sessions: Map<string, { turnPromise: Promise<unknown> | null; abortController: AbortController | null }>;
        }
      ).sessions.get(created.sessionId)!;
      expect(active.abortController).toBeNull();
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
