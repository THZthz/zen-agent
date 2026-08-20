import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { ModelMessage } from "ai";
import {
  formatRecoveredSummary,
  needsSessionStatsRecovery,
  rebuildSessionStats,
} from "./session-stats.js";
import {
  createStoredSession,
  emptySessionUsage,
  readStoredSession,
  sessionLlmLogPath,
} from "./storage.js";

const SESSION_ID = "sess_test123";
const SYSTEM = "You are a test agent.";
const MODEL = "deepseek-v4-flash" as const;

function user(text: string): ModelMessage {
  return { role: "user", content: text };
}
function assistant(text: string, toolCall = false): ModelMessage {
  return {
    role: "assistant",
    content: toolCall
      ? [
          {
            type: "tool-call",
            toolCallId: "call_1",
            toolName: "bash",
            input: { command: "ls" },
          },
        ]
      : [{ type: "text", text }],
  };
}
function toolResult(text: string): ModelMessage {
  return {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "call_1",
        toolName: "bash",
        output: { type: "text", value: text },
      },
    ],
  };
}

function legacyMessages(): ModelMessage[] {
  return [
    user("turn one"),
    assistant("thinking...", true),
    toolResult("file1\nfile2"),
    assistant("done with turn one"),
    user("turn two"),
    assistant("final answer"),
  ];
}

function writeLog(lines: Array<Record<string, unknown>>): string {
  const dir = mkdtempSync(join(tmpdir(), "zen-agent-stats-"));
  mkdirSync(join(dir, ".sessions", "llm"), { recursive: true });
  writeFileSync(
    join(dir, ".sessions", "llm", `${SESSION_ID}.jsonl`),
    lines.map((l) => JSON.stringify(l)).join("\n") + "\n",
    "utf8",
  );
  return dir;
}

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function environmentMessage(content: string): ModelMessage & { name: string } {
  return { role: "user", content, name: "environment" };
}

describe("needsSessionStatsRecovery", () => {
  it("is false when usage is populated", () => {
    const usage = emptySessionUsage();
    usage.turns = 2;
    expect(needsSessionStatsRecovery({ usage, llmMessages: legacyMessages() })).toBe(false);
  });

  it("is false when there is no history", () => {
    expect(needsSessionStatsRecovery({ usage: emptySessionUsage(), llmMessages: [] })).toBe(false);
  });

  it("is true for legacy sessions (empty usage + history)", () => {
    expect(
      needsSessionStatsRecovery({ usage: emptySessionUsage(), llmMessages: legacyMessages() }),
    ).toBe(true);
  });

  it("is false when only environment messages exist (no real conversation)", () => {
    expect(
      needsSessionStatsRecovery({
        usage: emptySessionUsage(),
        llmMessages: [environmentMessage("Working directory: /tmp")],
      }),
    ).toBe(false);
  });
});

describe("rebuildSessionStats from legacy llm log (no usage)", () => {
  it("recovers turns, steps, tokens, times and marks the result estimated", async () => {
    const t0 = "2026-08-19T10:00:00.000Z";
    const t1 = "2026-08-19T10:00:02.000Z"; // 2s llm step 1 (tool call)
    const t2 = "2026-08-19T10:00:05.000Z"; // 3s tool call
    const t3 = "2026-08-19T10:00:06.500Z"; // 1.5s llm step 2 (answer)
    const t4 = "2026-08-19T10:00:10.000Z"; // 3.5s gap before turn 2
    const t5 = "2026-08-19T10:00:11.000Z"; // 1s llm step 3

    const req1 = {
      type: "llm_request",
      timestamp: t0,
      model: MODEL,
      system: SYSTEM,
      messages: [user("turn one")],
    };
    const resp1 = {
      type: "llm_response",
      timestamp: t1,
      text: "",
      toolCalls: [{ id: "call_1", name: "bash", input: { command: "ls" } }],
      finishReason: "tool-calls",
    };
    const req2 = {
      type: "llm_request",
      timestamp: t2,
      model: MODEL,
      system: SYSTEM,
      messages: [user("turn one"), assistant("", true), toolResult("file1\nfile2")],
    };
    const resp2 = {
      type: "llm_response",
      timestamp: t3,
      text: "done with turn one",
      toolCalls: [],
      finishReason: "stop",
    };
    const req3 = {
      type: "llm_request",
      timestamp: t4,
      model: MODEL,
      system: SYSTEM,
      messages: [user("turn one"), assistant("", true), toolResult("file1\nfile2"), assistant("done with turn one"), user("turn two")],
    };
    const resp3 = {
      type: "llm_response",
      timestamp: t5,
      text: "final answer",
      toolCalls: [],
      finishReason: "stop",
    };

    const dir = writeLog([req1, resp1, req2, resp2, req3, resp3]);
    dirs.push(dir);

    const rebuilt = await rebuildSessionStats(
      dir,
      SESSION_ID,
      legacyMessages(),
      SYSTEM,
      MODEL,
    );
    expect(rebuilt).not.toBeNull();
    expect(rebuilt!.estimated).toBe(true);
    expect(rebuilt!.usage.turns).toBe(2);
    expect(rebuilt!.usage.steps).toBe(3);
    expect(rebuilt!.turnStats[0]!.steps).toBe(2);
    expect(rebuilt!.turnStats[1]!.steps).toBe(1);
    // llmMs: 2s + 1.5s = 3500; 1s = 1000
    expect(rebuilt!.turnStats[0]!.llmMs).toBe(3500);
    expect(rebuilt!.turnStats[1]!.llmMs).toBe(1000);
    // toolMs: gap between resp1 (t1) and req2 (t2) = 3000
    expect(rebuilt!.turnStats[0]!.toolMs).toBe(3000);
    expect(rebuilt!.usage.llmMs).toBe(4500);
    expect(rebuilt!.usage.toolMs).toBe(3000);
    expect(rebuilt!.usage.inputTokens).toBeGreaterThan(0);
    expect(rebuilt!.usage.outputTokens).toBeGreaterThan(0);
    expect(rebuilt!.usage.cacheReadTokens).toBe(0);
    expect(rebuilt!.usage.cacheMissTokens).toBe(0);
    expect(rebuilt!.lastContextUsed).toBeGreaterThan(0);
    expect(rebuilt!.usage.costYuan).toBeGreaterThan(0);

    const summary = formatRecoveredSummary(rebuilt!);
    expect(summary).toContain("Recovered session stats (estimated)");
    expect(summary).toContain("2 turns");
    expect(summary).toContain("3 steps");
    expect(summary).toContain("cache n/a");
  });
});

describe("rebuildSessionStats from exact llm log (with usage)", () => {
  it("uses exact usage and is not marked estimated", async () => {
    const dir = writeLog([
      {
        type: "llm_request",
        timestamp: "2026-08-19T10:00:00.000Z",
        model: MODEL,
        system: SYSTEM,
        messages: [user("hello")],
      },
      {
        type: "llm_response",
        timestamp: "2026-08-19T10:00:01.000Z",
        text: "hi",
        toolCalls: [],
        finishReason: "stop",
        usage: {
          inputTokens: 1234,
          outputTokens: 56,
          totalTokens: 1290,
          cacheReadTokens: 100,
          cacheMissTokens: 1134,
          reasoningTokens: 10,
          llmMs: 950,
          thinkingMs: 900,
          answeringMs: 50,
        },
      },
    ]);
    dirs.push(dir);

    const rebuilt = await rebuildSessionStats(
      dir,
      SESSION_ID,
      [user("hello")],
      SYSTEM,
      MODEL,
    );
    expect(rebuilt).not.toBeNull();
    expect(rebuilt!.estimated).toBe(false);
    expect(rebuilt!.usage.turns).toBe(1);
    expect(rebuilt!.usage.steps).toBe(1);
    expect(rebuilt!.usage.inputTokens).toBe(1234);
    expect(rebuilt!.usage.outputTokens).toBe(56);
    expect(rebuilt!.usage.cacheReadTokens).toBe(100);
    expect(rebuilt!.usage.cacheMissTokens).toBe(1134);
    expect(rebuilt!.usage.reasoningTokens).toBe(10);
    expect(rebuilt!.usage.llmMs).toBe(950);
    expect(rebuilt!.usage.thinkingMs).toBe(900);
    expect(rebuilt!.usage.answeringMs).toBe(50);
    expect(rebuilt!.lastContextUsed).toBe(1234);

    const summary = formatRecoveredSummary(rebuilt!);
    expect(summary).not.toContain("estimated");
    expect(summary).toContain("cache 8%");
  });
});

describe("rebuildSessionStats without llm log", () => {
  it("falls back to messages-only estimation (no times)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "zen-agent-stats-"));
    dirs.push(dir);

    const rebuilt = await rebuildSessionStats(
      dir,
      SESSION_ID,
      legacyMessages(),
      SYSTEM,
      MODEL,
    );
    expect(rebuilt).not.toBeNull();
    expect(rebuilt!.estimated).toBe(true);
    expect(rebuilt!.usage.turns).toBe(2);
    expect(rebuilt!.usage.steps).toBe(3);
    expect(rebuilt!.usage.inputTokens).toBeGreaterThan(0);
    expect(rebuilt!.usage.outputTokens).toBeGreaterThan(0);
    expect(rebuilt!.usage.llmMs).toBe(0);
    expect(rebuilt!.usage.toolMs).toBe(0);
    expect(rebuilt!.usage.costYuan).toBeGreaterThan(0);
  });

  it("returns null when there is no message history", async () => {
    const dir = mkdtempSync(join(tmpdir(), "zen-agent-stats-"));
    dirs.push(dir);
    const rebuilt = await rebuildSessionStats(dir, SESSION_ID, [], SYSTEM, MODEL);
    expect(rebuilt).toBeNull();
  });
});

describe("storage round-trip", () => {
  it("creates sessions with empty turnStats and backfills legacy files", async () => {
    const dir = mkdtempSync(join(tmpdir(), "zen-agent-stats-"));
    dirs.push(dir);

    const created = await createStoredSession(dir);
    expect(created.turnStats).toEqual([]);
    expect(created.usage.turns).toBe(0);

    // Simulate a legacy session file (no usage, no turnStats).
    const legacy = JSON.parse(JSON.stringify(created));
    delete legacy.usage;
    delete legacy.turnStats;
    writeFileSync(
      join(dir, ".sessions", "sessions", `${created.sessionId}.json`),
      JSON.stringify(legacy),
      "utf8",
    );

    const loaded = await readStoredSession(dir, created.sessionId);
    expect(loaded.turnStats).toEqual([]);
    expect(loaded.usage.turns).toBe(0);
  });
});

describe("rebuildSessionStats turn counting with environment messages", () => {
  it("ignores environment messages when counting user turns", async () => {
    const env = environmentMessage("Working directory: /tmp\nGit branch: main");
    const cont = environmentMessage("Session continued/resumed.\nWorking directory: /tmp");
    const dir = writeLog([
      {
        type: "llm_request",
        timestamp: "2026-08-19T10:00:00.000Z",
        system: SYSTEM,
        messages: [env, user("turn one"), assistant("thinking...", true), toolResult("out")],
      },
      {
        type: "llm_response",
        timestamp: "2026-08-19T10:00:01.000Z",
        text: "answer one",
        toolCalls: [],
        finishReason: "stop",
      },
      {
        type: "llm_request",
        timestamp: "2026-08-19T10:00:02.000Z",
        system: SYSTEM,
        messages: [env, user("turn one"), assistant("thinking...", true), toolResult("out"), cont, user("turn two")],
      },
      {
        type: "llm_response",
        timestamp: "2026-08-19T10:00:03.000Z",
        text: "answer two",
        toolCalls: [],
        finishReason: "stop",
      },
    ]);
    dirs.push(dir);

    const rebuilt = await rebuildSessionStats(dir, SESSION_ID, legacyMessages(), SYSTEM, MODEL);
    expect(rebuilt).not.toBeNull();
    expect(rebuilt!.usage.turns).toBe(2);
  });
});
