import { beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as acp from "@agentclientprotocol/sdk";

vi.mock("./deepseek.js", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from "./agent.js";
import { runLlmStep, type LlmStepResult } from "./deepseek.js";

const mockedRunLlmStep = vi.mocked(runLlmStep);

interface NotifyCall {
  sessionId: string;
  update: acp.SessionUpdate;
}

function makeAgentContext(options: {
  waitForExit?: Promise<{ exitCode: number | null; signal: number | null }>;
}) {
  const notifications: NotifyCall[] = [];
  const request = vi.fn((method: string) => {
    switch (method) {
      case acp.methods.client.terminal.create:
        return Promise.resolve({ terminalId: "t1" });
      case acp.methods.client.terminal.waitForExit:
        return options.waitForExit ?? Promise.resolve({ exitCode: 0, signal: null });
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
  const notify = vi.fn(async (method: string, params: NotifyCall) => {
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

function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function bashStep(overrides: Partial<LlmStepResult> = {}): LlmStepResult {
  return {
    text: "",
    reasoning: "",
    toolCalls: [{ id: "c1", name: "bash", input: { command: "echo hi" } }],
    finishReason: "tool-calls",
    usage: null,
    ...overrides,
  };
}

/**
 * A runLlmStep mock that behaves like the real streaming step: it awaits a
 * deferred result, and when the result has text it streams it through the
 * onTextDelta callback (which runTurn feeds into StreamThrottle, producing
 * agent_message_chunk notifications).
 */
function streamingStepMock(resultPromise: Promise<LlmStepResult>) {
  mockedRunLlmStep.mockImplementationOnce(async (options: {
    onTextDelta?: (delta: string) => void | Promise<void>;
  }) => {
    const result = await resultPromise;
    if (result.text.length > 0) {
      await options.onTextDelta?.(result.text);
    }
    return result;
  });
}

async function setupAgent(cwd: string) {
  const agent = new ZenAgent();
  // Advertise terminal capability so executeLlmToolCall uses the real
  // terminal path instead of failing fast (clientCapabilities is only set
  // by initialize()).
  await agent.initialize({
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: { terminal: true },
  } as acp.InitializeRequest);
  const { cx, notifications, request } = makeAgentContext({});
  const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
  return { agent, cx, notifications, request, sessionId: created.sessionId };
}

function statusOf(notifications: NotifyCall[], toolCallId: string): string | undefined {
  const updates = notifications.filter(
    (n) =>
      n.update.sessionUpdate === "tool_call_update" &&
      n.update.toolCallId === toolCallId,
  );
  const last = updates.at(-1)?.update as
    | { sessionUpdate: "tool_call_update"; status?: string }
    | undefined;
  return last?.status;
}

describe("graceful cancel in runTurn", () => {
  beforeEach(() => {
    // Clear call history and once-queues from previous tests so
    // toHaveBeenCalled() reflects only the current test.
    mockedRunLlmStep.mockReset();
  });

  it("lets the in-flight bash tool finish, persists results, then returns cancelled", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-test-"));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);
      const exit = deferred<{ exitCode: number | null; signal: number | null }>();
      const ctx = makeAgentContext({ waitForExit: exit.promise });

      // First LLM step proposes a bash tool call. The mock must be installed
      // before prompt() runs, since runTurn calls runLlmStep immediately.
      mockedRunLlmStep.mockResolvedValueOnce(bashStep());
      const promptPromise = agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "run it" }] },
        ctx.cx,
      );

      await vi.waitFor(() => {
        expect(ctx.request).toHaveBeenCalledWith(
          acp.methods.client.terminal.create,
          expect.anything(),
        );
      });

      // User force-sends a message while the tool is running.
      agent.cancel({ sessionId });

      // The tool is NOT killed; it finishes normally.
      exit.resolve({ exitCode: 0, signal: null });
      const response = await promptPromise;

      expect(response.stopReason).toBe("cancelled");
      expect(ctx.request).not.toHaveBeenCalledWith(acp.methods.client.terminal.kill, expect.anything());
      expect(statusOf(ctx.notifications, "c1")).toBe("completed");
      // The turn emitted a live tool card for the finished command.
      expect(ctx.notifications.some((n) => n.update.sessionUpdate === "tool_call")).toBe(true);

      const session = (agent as unknown as { sessions: Map<string, { session: import("./storage.js").StoredSession }> })
        .sessions.get(sessionId)!.session;
      const assistant = session.llmMessages.find((m) => m.role === "assistant")!;
      expect(assistant.content).toEqual([
        { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "echo hi" } },
      ]);
      const tool = session.llmMessages.find((m) => m.role === "tool")!;
      expect(tool.content).toHaveLength(1);
      // The completed tool result is persisted so the follow-up turn has context.
      expect(session.llmMessages.at(-1)?.role).toBe("tool");
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("discards tool calls proposed by an LLM step that finished after cancel", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-test-"));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);
      const step = deferred<LlmStepResult>();
      mockedRunLlmStep.mockReturnValueOnce(step.promise);

      const ctx = makeAgentContext({});
      const promptPromise = agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "think then act" }] },
        ctx.cx,
      );

      // Cancel while the LLM step is in flight ("thinking").
      await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalled());
      agent.cancel({ sessionId });
      step.resolve(bashStep());

      const response = await promptPromise;

      expect(response.stopReason).toBe("cancelled");
      expect(ctx.request).not.toHaveBeenCalledWith(acp.methods.client.terminal.create, expect.anything());
      const session = (agent as unknown as { sessions: Map<string, { session: import("./storage.js").StoredSession }> })
        .sessions.get(sessionId)!.session;
      expect(session.llmMessages.some((m) => m.role === "tool")).toBe(false);
      expect(notifications.filter((n) => n.update.sessionUpdate === "tool_call")).toHaveLength(0);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it("keeps a completed final answer and returns cancelled", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-test-"));
    try {
      const { agent, cx, sessionId } = await setupAgent(cwd);
      const step = deferred<LlmStepResult>();
      streamingStepMock(step.promise);

      const ctx = makeAgentContext({});
      const promptPromise = agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "answer me" }] },
        ctx.cx,
      );

      await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalled());
      agent.cancel({ sessionId });
      step.resolve({ text: "the answer", reasoning: "", toolCalls: [], finishReason: "stop", usage: null });

      const response = await promptPromise;

      expect(response.stopReason).toBe("cancelled");
      const session = (agent as unknown as { sessions: Map<string, { session: import("./storage.js").StoredSession }> })
        .sessions.get(sessionId)!.session;
      // The completed answer is kept in the conversation history.
      expect(session.llmMessages.at(-1)?.role).toBe("assistant");
      // ...and was streamed to Zed as a message chunk before the turn stopped.
      expect(ctx.notifications.some((n) => n.update.sessionUpdate === "agent_message_chunk")).toBe(true);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});

describe("per-turn stats line", () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  it("is shown to the user and persisted in events, but never sent to the LLM", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-test-"));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
      mockedRunLlmStep.mockResolvedValueOnce({
        text: "the answer",
        reasoning: "thinking...",
        toolCalls: [],
        finishReason: "stop",
        usage: {
          inputTokens: 1000,
          outputTokens: 50,
          totalTokens: 1050,
          cacheReadTokens: 900,
          cacheMissTokens: 100,
          reasoningTokens: 10,
          llmMs: 500,
          thinkingMs: 300,
          answeringMs: 200,
        },
      });

      await agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "hi" }] },
        cx,
      );


      const statsRegex = /Turn \d+ · \d+ step/;
      const isStatsBubble = (u: acp.SessionUpdate) =>
        u.sessionUpdate === "agent_message_chunk" &&
        "content" in u &&
        (u as { content: { type: string; text?: string } }).content?.type === "text" &&
        statsRegex.test(
          String((u as { content: { text?: string } }).content.text ?? ""),
        );

      // Shown to the user live.
      expect(
        notifications.filter((n) => isStatsBubble(n.update)),
      ).not.toHaveLength(0);

      const session = (
        agent as unknown as {
          sessions: Map<string, { session: import("./storage.js").StoredSession }>;
        }
      ).sessions.get(sessionId)!.session;

      // Persisted in events so the bubble survives reload.
      expect(session.events.some((e) => isStatsBubble(e))).toBe(true);

      // NEVER sent to the LLM: stats text must not appear in llmMessages.
      const llmText = JSON.stringify(session.llmMessages);
      expect(llmText).not.toMatch(statsRegex);
      expect(llmText).not.toContain("(session ¥");
      // The assistant message carries only the real answer.
      expect(
        session.llmMessages.some(
          (m) => m.role === "assistant" && JSON.stringify(m.content).includes("the answer"),
        ),
      ).toBe(true);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});

describe("debug log stats", () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  it("writes per-step and per-turn stats with cache hit ratio to log.jsonl", async () => {
    const { readFileSync } = await import("node:fs");
    const cwd = mkdtempSync(join(tmpdir(), "zen-agent-test-"));
    try {
      const { agent, cx, sessionId } = await setupAgent(cwd);
      mockedRunLlmStep.mockResolvedValueOnce({
        text: "the answer",
        reasoning: "thinking...",
        toolCalls: [],
        finishReason: "stop",
        usage: {
          inputTokens: 1000,
          outputTokens: 50,
          totalTokens: 1050,
          cacheReadTokens: 900,
          cacheMissTokens: 100,
          reasoningTokens: 10,
          llmMs: 500,
          thinkingMs: 300,
          answeringMs: 200,
        },
      });

      await agent.prompt(
        { sessionId, prompt: [{ type: "text", text: "hi" }] },
        cx,
      );

      const startupLogKey = (
        agent as unknown as { startupLogKey: string }
      ).startupLogKey;
      // Local-time startup timestamp + uuid: 2026-08-21-23-06-04_<uuid>.
      expect(startupLogKey).toMatch(
        /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_[0-9a-f-]{36}$/,
      );
      const logPath = join(cwd, ".sessions", "client", startupLogKey, "log.jsonl");

      let entries: Array<Record<string, unknown>> = [];
      // logRuntime is fire-and-forget; wait until the turn entry lands.
      await vi.waitFor(() => {
        const raw = readFileSync(logPath, "utf8");
        entries = raw
          .trim()
          .split("\n")
          .filter(Boolean)
          .map((line) => JSON.parse(line) as Record<string, unknown>);
        expect(entries.some((e) => e.message === "turn stats")).toBe(true);
      });

      const step = entries.find((e) => e.message === "llm step stats");
      expect(step).toBeDefined();
      expect(step).toMatchObject({
        sessionId,
        step: 1,
        model: "deepseek-v4-flash",
        inputTokens: 1000,
        outputTokens: 50,
        cacheReadTokens: 900,
        cacheMissTokens: 100,
        cacheHitPercent: "90.00%",
        reasoningTokens: 10,
        finishReason: "stop",
        toolCalls: 0,
      });
      expect(typeof step!.costYuan).toBe("number");
      expect(step!.llmMs).toBe(500);
      expect(step!.thinkingMs).toBe(300);
      expect(step!.answeringMs).toBe(200);

      const turn = entries.find((e) => e.message === "turn stats");
      expect(turn).toBeDefined();
      expect(turn).toMatchObject({
        sessionId,
        turn: 1,
        model: "deepseek-v4-flash",
        stopReason: "end_turn",
        steps: 1,
        inputTokens: 1000,
        outputTokens: 50,
        cacheReadTokens: 900,
        cacheMissTokens: 100,
        cacheHitPercent: "90.00%",
        reasoningTokens: 10,
      });
      expect(turn!.llmMs).toBe(500);
      expect(turn!.toolMs).toBe(0);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
