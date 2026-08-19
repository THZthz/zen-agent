import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { ZenAgent } from "./agent.js";

type ReplayEvent = {
  sessionUpdate: string;
  toolCallId?: string;
  status?: string;
  content?: unknown;
  [k: string]: unknown;
};

function prepare(events: ReplayEvent[]): ReplayEvent[] {
  const agent = new ZenAgent() as unknown as {
    prepareReplayEvents(events: ReplayEvent[]): ReplayEvent[];
  };
  return agent.prepareReplayEvents(events);
}

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
});

describe("replay preparation on real session data", () => {
  it("produces a replay stream with no terminal content and no orphan/in-progress updates", () => {
    const raw = readFileSync(
      ".sessions/sessions/sess_3eacc6f080853c52110c1791.json",
      "utf8",
    );
    const session = JSON.parse(raw) as { events: ReplayEvent[] };
    const out = prepare(session.events);

    const toolCalls = out.filter((e) => e.sessionUpdate === "tool_call");
    const updates = out.filter((e) => e.sessionUpdate === "tool_call_update");
    const finalUpdates = updates.filter(
      (e) => e.status === "completed" || e.status === "failed",
    );
    const inProgress = updates.filter((e) => e.status === "in_progress");

    expect(inProgress).toHaveLength(0);
    expect(toolCalls).toHaveLength(finalUpdates.length);
    for (const u of updates) {
      expect(u.content ?? []).not.toContainEqual(
        expect.objectContaining({ type: "terminal" }),
      );
    }
    expect(toolCalls.length).toBeGreaterThan(0);
    // All kept updates must have a matching kept tool_call.
    const ids = new Set(toolCalls.map((e) => e.toolCallId));
    for (const u of updates) {
      expect(ids.has(u.toolCallId!)).toBe(true);
    }
  });
});
