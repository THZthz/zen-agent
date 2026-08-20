import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import {
  buildEnvironmentMessage,
  buildSessionContinuedMessage,
  getUserMessageName,
  isEnvironmentMessage,
} from "./system-prompt.js";
import { emptySessionUsage, type StoredSession } from "./storage.js";

let dir: string;
let configDir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "zen-agent-git-"));
  // Isolate git from the machine's global/system config so tests are
  // deterministic regardless of the host's git user.name. The config file
  // lives OUTSIDE the repo so it never shows up in `git status`.
  configDir = mkdtempSync(join(tmpdir(), "zen-agent-gitcfg-"));
  writeFileSync(join(configDir, "gitconfig"), "");
  process.env.GIT_CONFIG_GLOBAL = join(configDir, "gitconfig");
  process.env.GIT_CONFIG_SYSTEM = "/dev/null";
});

afterEach(() => {
  delete process.env.GIT_CONFIG_GLOBAL;
  delete process.env.GIT_CONFIG_SYSTEM;
  rmSync(dir, { recursive: true, force: true });
  rmSync(configDir, { recursive: true, force: true });
});

function makeSession(cwd: string): StoredSession {
  return {
    sessionId: "s1",
    cwd,
    createdAt: "2026-08-20T02:00:00.000Z",
    updatedAt: "2026-08-20T02:00:00.000Z",
    title: null,
    events: [],
    llmMessages: [],
    config: { model: "deepseek-v4-flash", thinkingEffort: "off", systemPrompt: "" },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

function git(args: string[]): void {
  execFileSync("git", args, { cwd: dir, stdio: "ignore" });
}

describe("getUserMessageName", () => {
  it("returns the git user name, sanitized to the OpenAI name charset", async () => {
    git(["init", "-b", "main"]);
    git(["config", "user.name", "John Doe"]);
    git(["config", "user.email", "john@example.com"]);
    expect(await getUserMessageName(dir)).toBe("John_Doe");
  });

  it("falls back to 'User' when git has no user.name", async () => {
    git(["init", "-b", "main"]);
    // No user.name configured anywhere.
    expect(await getUserMessageName(dir)).toBe("User");
  });

  it("falls back to 'User' outside a git repository", async () => {
    expect(await getUserMessageName(dir)).toBe("User");
  });
});

describe("buildEnvironmentMessage", () => {
  it("includes working directory and date even without git", async () => {
    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toBe(
      `Working directory: ${dir}\nCurrent date/time: 2026-08-20T02:00:00.000Z`,
    );
  });

  it("adds branch, commit and clean status inside a git repository", async () => {
    git(["init", "-b", "feature/test"]);
    git(["config", "user.name", "Tester"]);
    git(["config", "user.email", "tester@example.com"]);
    writeFileSync(join(dir, "a.txt"), "hello\n");
    git(["add", "a.txt"]);
    git(["commit", "-m", "init"]);

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain(`Working directory: ${dir}`);
    expect(text).toContain("Git branch: feature/test");
    expect(text).toMatch(/Git commit: [0-9a-f]{7}/);
    expect(text).toContain("Git status: clean");
  });

  it("reports the number of changed files when the tree is dirty", async () => {
    git(["init", "-b", "main"]);
    git(["config", "user.name", "Tester"]);
    git(["config", "user.email", "tester@example.com"]);
    writeFileSync(join(dir, "a.txt"), "hello\n");
    git(["add", "a.txt"]);
    git(["commit", "-m", "init"]);
    writeFileSync(join(dir, "a.txt"), "hello, world\n");
    writeFileSync(join(dir, "b.txt"), "new\n");

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain("Git status: 2 changed files");
  });
});

describe("buildSessionContinuedMessage", () => {
  it("marks the continuation and includes fresh git state", async () => {
    git(["init", "-b", "main"]);
    git(["config", "user.name", "Tester"]);
    git(["config", "user.email", "tester@example.com"]);
    writeFileSync(join(dir, "a.txt"), "hello\n");
    git(["add", "a.txt"]);
    git(["commit", "-m", "init"]);
    writeFileSync(join(dir, "a.txt"), "changed\n");

    const text = await buildSessionContinuedMessage(makeSession(dir));
    expect(text).toContain("Session continued/resumed.");
    expect(text).toContain(`Working directory: ${dir}`);
    expect(text).toContain("Git branch: main");
    expect(text).toContain("Git status: 1 changed file");
  });
});

describe("isEnvironmentMessage", () => {
  it("detects only auto-generated environment user messages", () => {
    expect(
      isEnvironmentMessage({ role: "user", content: "x", name: "Environment" }),
    ).toBe(true);
    expect(
      isEnvironmentMessage({ role: "user", content: "x", name: "Amias" }),
    ).toBe(false);
    expect(isEnvironmentMessage({ role: "user", content: "x" })).toBe(false);
    expect(isEnvironmentMessage({ role: "assistant", content: [] })).toBe(false);
  });
});
