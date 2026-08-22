import { beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdtempSync, realpathSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import * as acp from "@agentclientprotocol/sdk";

vi.mock("./deepseek.js", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from "./agent.js";
import { runLlmStep, type LlmStepResult } from "./deepseek.js";

const mockedRunLlmStep = vi.mocked(runLlmStep);

function makeAgentContext() {
  const notifications: Array<{ sessionId: string; update: acp.SessionUpdate }> = [];
  const request = vi.fn((method: string, _params?: unknown) => {
    switch (method) {
      case acp.methods.client.terminal.create:
        return Promise.resolve({ terminalId: "t1" });
      case acp.methods.client.terminal.waitForExit:
        return Promise.resolve({ exitCode: 0, signal: null });
      case acp.methods.client.terminal.output:
        return Promise.resolve({ output: "done", truncated: false });
      case acp.methods.client.terminal.release:
        return Promise.resolve({});
      case acp.methods.client.terminal.kill:
        return Promise.resolve({});
      default:
        return Promise.reject(new Error(`unexpected client request: ${method}`));
    }
  });
  const notify = vi.fn(async (method: string, params: { sessionId: string; update: acp.SessionUpdate }) => {
    if (method === acp.methods.client.session.update) {
      notifications.push(params);
    }
  });
  return {
    cx: { request, notify } as unknown as acp.AgentContext,
    notifications,
    request,
  };
}

async function setupAgent(cwd: string) {
  const agent = new ZenAgent();
  await agent.initialize({
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: { terminal: true },
  } as acp.InitializeRequest);
  const { cx, notifications, request } = makeAgentContext();
  const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
  return { agent, cx, notifications, request, sessionId: created.sessionId };
}

function bashStep(): LlmStepResult {
  return {
    text: "",
    reasoning: "",
    toolCalls: [{ id: "c1", name: "bash", input: { command: "echo hi" } }],
    finishReason: "tool-calls",
    usage: null,
  };
}

/** Queues one bash step followed by a final answer so the turn terminates. */
function queueBashThenAnswer(): void {
  mockedRunLlmStep
    .mockResolvedValueOnce(bashStep())
    .mockResolvedValueOnce({
      text: "done",
      reasoning: "",
      toolCalls: [],
      finishReason: "end_turn",
      usage: null,
    });
}

function agentMessages(notifications: Array<{ update: acp.SessionUpdate }>): string[] {
  return notifications
    .filter((n) => n.update.sessionUpdate === "agent_message_chunk")
    .map((n) => {
      const c = n.update as { content?: { type?: string; text?: string } };
      return c.content?.text ?? "";
    });
}

/** The `-lc` script passed to terminal.create, or undefined. */
function createdScript(
  request: ReturnType<typeof makeAgentContext>["request"],
): string | undefined {
  const call = request.mock.calls.find(
    (c) => c[0] === acp.methods.client.terminal.create,
  );
  const params = call?.[1] as { args?: string[] } | undefined;
  return params?.args?.[1];
}

describe("/sandbox slash command", () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  it("reports OFF by default and does not wrap bash calls", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);
      const status = await agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "/sandbox" }] },
        cx,
      );
      expect(status.stopReason).toBe("end_turn");
      expect(agentMessages(notifications).join("\n")).toContain(
        "Bash tool sandbox: OFF (off)",
      );

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "run it" }] }, cx);
      expect(createdScript(request)).not.toContain("bwrap");
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("turns the sandbox on and wraps subsequent bash calls in bwrap", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/sandbox on" }] }, cx);
      expect(agentMessages(notifications).join("\n")).toContain(
        "Bash tool calls are now sandboxed",
      );
      const active = (agent as unknown as {
        sessions: Map<string, { session: { config: { sandbox: boolean } } }>;
      }).sessions.get(sessionId)!;
      expect(active.session.config.sandbox).toBe(true);

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "run it" }] }, cx);
      const script = createdScript(request)!;
      expect(script).toContain("bwrap");
      expect(script).toMatch(/bwrap .*--ro-bind \/mnt \/mnt/);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("turns the sandbox off and stops wrapping bash calls", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/sandbox on" }] }, cx);
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/sandbox off" }] }, cx);
      expect(agentMessages(notifications).join("\n")).toContain(
        "Bash tool sandbox disabled",
      );

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "run it" }] }, cx);
      expect(createdScript(request)).not.toContain("bwrap");
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("rejects unknown arguments without changing state", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
      const response = await agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "/sandbox maybe" }] },
        cx,
      );
      expect(response.stopReason).toBe("end_turn");
      expect(agentMessages(notifications).join("\n")).toContain("Usage: /sandbox on | off");
      const active = (agent as unknown as {
        sessions: Map<string, { session: { config: { sandbox: boolean } } }>;
      }).sessions.get(sessionId)!;
      expect(active.session.config.sandbox).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("reports ON and refuses to disable when ZEN_AGENT_SANDBOX=1 is set", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    process.env.ZEN_AGENT_SANDBOX = "1";
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      const status = await agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "/sandbox" }] },
        cx,
      );
      expect(agentMessages(notifications).join("\n")).toContain(
        "Bash tool sandbox: ON (enforced by ZEN_AGENT_SANDBOX=1)",
      );

      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/sandbox off" }] }, cx);
      expect(agentMessages(notifications).join("\n")).toContain("Cannot disable");

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "run it" }] }, cx);
      expect(createdScript(request)).toContain("bwrap");
      void status;
    } finally {
      delete process.env.ZEN_AGENT_SANDBOX;
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});


describe("bash tool sandbox blocks rm/grep/find", () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  it("shadows every distinct rm/grep/find binary with the refusing shim", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-sandbox-"));
    try {
      const { agent, cx, request, sessionId } = await setupAgent(cwd);
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/sandbox on" }] }, cx);

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "run it" }] }, cx);
      const script = createdScript(request)!;
      expect(script).toContain("bwrap");

      const shim = fileURLToPath(
        new URL("../bin/zen-agent-sandbox-block.sh", import.meta.url),
      );
      // Mirror the deduplication in tool-execution.ts: /bin is a symlink to
      // /usr/bin on most distros, so only distinct real binaries are bound.
      const seen = new Set<string>();
      const expectedBinds: string[] = [];
      for (const cmd of ["rm", "grep", "find"]) {
        for (const dir of ["/usr/bin", "/bin"]) {
          const dest = join(dir, cmd);
          if (!existsSync(dest)) continue;
          const resolved = realpathSync(dest);
          if (seen.has(resolved)) continue;
          seen.add(resolved);
          expectedBinds.push(`--ro-bind ${shim} ${dest}`);
        }
      }
      expect(expectedBinds.length).toBeGreaterThan(0);
      for (const bind of expectedBinds) {
        expect(script).toContain(bind);
      }
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("refuses rm, grep and find and suggests their substitutes", () => {
    const dir = mkdtempSync(join(tmpdir(), "zen-agent-block-"));
    try {
      const shim = fileURLToPath(
        new URL("../bin/zen-agent-sandbox-block.sh", import.meta.url),
      );
      const substitutes: Record<string, string> = {
        rm: "trash",
        grep: "rg",
        find: "fdfind",
      };
      for (const [cmd, substitute] of Object.entries(substitutes)) {
        const link = join(dir, cmd);
        symlinkSync(shim, link);
        let error: unknown;
        try {
          execFileSync(link, ["--some-flag"], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
          });
        } catch (e) {
          error = e;
        }
        expect(error).toBeDefined();
        const e = error as { status?: number; stderr?: string };
        expect(e.status).toBe(1);
        expect(e.stderr).toContain(`'${cmd}' is blocked`);
        expect(e.stderr).toContain(`use '${substitute}' instead`);
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
