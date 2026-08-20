import { describe, expect, it } from "vitest";
import type { SessionUpdate } from "@agentclientprotocol/sdk";
import { ZenAgent } from "./agent.js";
import { prepareReplayEvents } from "./replay.js";
import { emptySessionUsage, type StoredSession } from "./storage.js";

type ReplayEvent = {
  sessionUpdate: string;
  toolCallId?: string;
  status?: string;
  content?: unknown;
  [k: string]: unknown;
};

function prepare(events: ReplayEvent[]): SessionUpdate[] {
  return prepareReplayEvents(events as unknown as SessionUpdate[]);
}

function makeSession(sessionId = "s1"): StoredSession {
  return {
    sessionId,
    cwd: "/tmp",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    title: null,
    events: [],
    llmMessages: [],
    config: { model: "deepseek-v4-flash", thinkingEffort: "off", systemPrompt: "" },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

// Cast through unknown: ZenAgent's `sessions` map is private, so a plain
// intersection with a public duplicate would collapse to `never`.
type TestAgent = {
  sessions: Map<
    string,
    {
      session: StoredSession;
      abortController: AbortController | null;
      gracefulCancel: boolean;
      cancelTimer: NodeJS.Timeout | null;
    }
  >;
  abortActiveSession(sessionId: string): void;
  cancel(params: { sessionId: string }): void;
};

describe("prepareReplayEvents", () => {
  it("keeps final tool call pairs, strips terminal content, drops in-progress and orphan updates", () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: "tool_call",
        toolCallId: "c1",
        title: "$ ls",
        kind: "execute",
        status: "pending",
        rawInput: { command: "ls" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c1",
        status: "in_progress",
        content: [{ type: "terminal", terminalId: "t1" }],
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c1",
        status: "completed",
        content: [
          { type: "terminal", terminalId: "t1" },
          { type: "content", content: { type: "text", text: "out" } },
        ],
        rawOutput: { output: "out" },
      },
      // Interrupted call: initial event but no final update -> dropped.
      {
        sessionUpdate: "tool_call",
        toolCallId: "c2",
        title: "$ sleep",
        kind: "execute",
        status: "pending",
        rawInput: { command: "sleep" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c2",
        status: "in_progress",
        content: [{ type: "terminal", terminalId: "t2" }],
      },
      // Orphan update without a tool_call event -> dropped.
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c3",
        status: "completed",
        content: [{ type: "content", content: { type: "text", text: "orphan" } }],
      },
      { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "hi" } },
    ];

    expect(prepare(events)).toEqual([
      {
        sessionUpdate: "tool_call",
        toolCallId: "c1",
        title: "$ ls",
        kind: "execute",
        status: "pending",
        rawInput: { command: "ls" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c1",
        status: "completed",
        content: [{ type: "content", content: { type: "text", text: "out" } }],
        rawOutput: { output: "out" },
      },
      { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "hi" } },
    ]);
  });

  it("keeps failed tool calls with text results", () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: "tool_call",
        toolCallId: "f1",
        title: "$ nope",
        kind: "execute",
        status: "pending",
        rawInput: { command: "nope" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "f1",
        status: "failed",
        content: [
          { type: "terminal", terminalId: "t9" },
          { type: "content", content: { type: "text", text: "boom" } },
        ],
        rawOutput: { output: "boom", exitCode: 127 },
      },
    ];
    expect(prepare(events)).toEqual([
      {
        sessionUpdate: "tool_call",
        toolCallId: "f1",
        title: "$ nope",
        kind: "execute",
        status: "pending",
        rawInput: { command: "nope" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "f1",
        status: "failed",
        content: [{ type: "content", content: { type: "text", text: "boom" } }],
        rawOutput: { output: "boom", exitCode: 127 },
      },
    ]);
  });

  it("keeps tool calls that completed during a graceful cancel (both events present)", () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: "tool_call",
        toolCallId: "g1",
        title: "$ npm test",
        kind: "execute",
        status: "pending",
        rawInput: { command: "npm test" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "g1",
        status: "completed",
        content: [{ type: "content", content: { type: "text", text: "passed" } }],
        rawOutput: { output: "passed" },
      },
      { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "Turn stats" } },
    ];
    const out = prepare(events);
    expect(out).toEqual(events);
    expect(out.filter((e) => e.sessionUpdate === "tool_call")).toHaveLength(1);
  });
});

describe("replay preparation on a session with terminal content", () => {
  it("produces a replay stream with no terminal content and no orphan/in-progress updates", () => {
    // Representative event stream as persisted by the agent: tool calls with
    // terminal cards, an in-progress update, an orphan update and a failed
    // call with a text result.
    const events: ReplayEvent[] = [
      { sessionUpdate: "user_message_chunk", content: { type: "text", text: "hello" } },
      {
        sessionUpdate: "tool_call",
        toolCallId: "c1",
        title: "$ ls",
        kind: "execute",
        status: "pending",
        rawInput: { command: "ls" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c1",
        status: "in_progress",
        content: [{ type: "terminal", terminalId: "t1" }],
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c1",
        status: "completed",
        content: [
          { type: "terminal", terminalId: "t1" },
          { type: "content", content: { type: "text", text: "a.txt" } },
        ],
        rawOutput: { output: "a.txt" },
      },
      {
        sessionUpdate: "tool_call",
        toolCallId: "c2",
        title: "$ git status",
        kind: "execute",
        status: "pending",
        rawInput: { command: "git status" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c2",
        status: "in_progress",
        content: [{ type: "terminal", terminalId: "t2" }],
      },
      // Orphan update: no matching tool_call, must be dropped.
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "orphan",
        status: "completed",
        content: [{ type: "content", content: { type: "text", text: "x" } }],
      },
      {
        sessionUpdate: "tool_call",
        toolCallId: "c3",
        title: "$ nope",
        kind: "execute",
        status: "pending",
        rawInput: { command: "nope" },
      },
      {
        sessionUpdate: "tool_call_update",
        toolCallId: "c3",
        status: "failed",
        content: [{ type: "content", content: { type: "text", text: "boom" } }],
        rawOutput: { output: "boom", exitCode: 127 },
      },
      { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "done" } },
    ];
    const out = prepare(events);

    const toolCalls = out.filter((e) => e.sessionUpdate === "tool_call");
    const updates = out.filter((e) => e.sessionUpdate === "tool_call_update");
    const finalUpdates = updates.filter(
      (e) => e.status === "completed" || e.status === "failed",
    );
    const inProgress = updates.filter((e) => e.status === "in_progress");

    expect(inProgress).toHaveLength(0);
    expect(toolCalls).toHaveLength(finalUpdates.length);
    // c1 (completed) and c3 (failed) survive; c2 has no final update and
    // the orphan update is dropped.
    expect(toolCalls).toHaveLength(2);
    for (const u of updates) {
      expect(u.content ?? []).not.toContainEqual(
        expect.objectContaining({ type: "terminal" }),
      );
    }
    // All kept updates must have a matching kept tool_call.
    const ids = new Set(toolCalls.map((e) => e.toolCallId));
    for (const u of updates) {
      expect(ids.has(u.toolCallId!)).toBe(true);
    }
  });
});

describe("graceful cancel", () => {
  it("sets gracefulCancel instead of aborting when a turn is running", () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    const controller = new AbortController();
    agent.sessions.set("s1", {
      session: makeSession(),
      abortController: controller,
      gracefulCancel: false,
      cancelTimer: null,
    });

    agent.cancel({ sessionId: "s1" });

    expect(controller.signal.aborted).toBe(false);
    const entry = agent.sessions.get("s1")!;
    expect(entry.gracefulCancel).toBe(true);
    expect(entry.cancelTimer).toBeNull();
  });

  it("does nothing when no turn is running", () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    agent.sessions.set("s1", {
      session: makeSession(),
      abortController: null,
      gracefulCancel: false,
      cancelTimer: null,
    });

    agent.cancel({ sessionId: "s1" });

    const entry = agent.sessions.get("s1")!;
    expect(entry.gracefulCancel).toBe(false);
  });

  it("does nothing for an unknown session", () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    expect(() => agent.cancel({ sessionId: "missing" })).not.toThrow();
  });

  it("hard abort (session close) clears graceful state and aborts", () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    const controller = new AbortController();
    agent.sessions.set("s1", {
      session: makeSession(),
      abortController: controller,
      gracefulCancel: true,
      cancelTimer: null,
    });

    agent.abortActiveSession("s1");

    expect(controller.signal.aborted).toBe(true);
    const entry = agent.sessions.get("s1")!;
    expect(entry.gracefulCancel).toBe(false);
  });

  it("schedules a hard-abort timer when the timeout env is set", async () => {
    const { vi } = await import("vitest");
    vi.resetModules();
    process.env.ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS = "10";
    try {
      const { ZenAgent: ReloadedZenAgent } = await import("./agent.js");
      const agent = new ReloadedZenAgent() as unknown as TestAgent;
      const controller = new AbortController();
      agent.sessions.set("s1", {
        session: makeSession(),
        abortController: controller,
        gracefulCancel: false,
        cancelTimer: null,
      });

      agent.cancel({ sessionId: "s1" });

      const entry = agent.sessions.get("s1")!;
      expect(entry.gracefulCancel).toBe(true);
      expect(entry.cancelTimer).not.toBeNull();

      await new Promise((resolve) => setTimeout(resolve, 40));
      expect(controller.signal.aborted).toBe(true);
      expect(agent.sessions.get("s1")!.cancelTimer).toBeNull();
    } finally {
      delete process.env.ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS;
    }
  });
});

describe("loadSession environment backfill", () => {
  it("prepends a frozen environment message and appends a continuation notice for empty sessions", async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const { vi } = await import("vitest");

    const dir = mkdtempSync(join(tmpdir(), "zen-agent-load-empty-"));
    try {
      const agent = new ZenAgent() as unknown as TestAgent & {
        loadSession(
          params: { cwd: string; sessionId: string },
          cx: {
            notify: (method: string, params: { sessionId: string; update: unknown }) => Promise<void>;
          },
        ): Promise<unknown>;
      };

      const session = makeSession("sess_empty");
      session.cwd = dir;
      mkdirSync(join(dir, ".sessions", "sess_empty"), { recursive: true });
      writeFileSync(
        join(dir, ".sessions", "sess_empty", "state.json"),
        JSON.stringify(session),
        "utf8",
      );

      const notify = vi.fn(async () => {});
      const cx = { notify } as unknown as Parameters<typeof agent.loadSession>[1];

      await agent.loadSession({ cwd: dir, sessionId: "sess_empty" }, cx);

      const loaded = agent.sessions.get("sess_empty")!.session;
      expect(loaded.llmMessages).toHaveLength(2);
      expect(loaded.llmMessages[0]).toMatchObject({ role: "user", name: "environment" });
      expect(String(loaded.llmMessages[0]!.content)).toContain("Working directory:");
      expect(loaded.llmMessages[1]).toMatchObject({ role: "user", name: "environment" });
      expect(String(loaded.llmMessages[1]!.content)).toContain(
        "Session continued/resumed.",
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
