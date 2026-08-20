import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { StoredSession } from "./storage.js";

const execFileAsync = promisify(execFile);

const GIT_TIMEOUT_MS = 5_000;

export const SYSTEM_PROMPT = `You are an experiened software engineer.

You have exactly one tool: bash.
You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.
There is no approval gate: every command you run is executed immediately.
Prefer small, targeted bash commands. Avoid large output from using bash tool.
When modifying files, use shell tools such as cat, sed, awk, or tee. Prefer using trash (npm install --global trash-cli), rg, fdfind (fd), jq, uv if they exist.

> Always use utf-8, no emojis unless needed by your task.`;

/**
 * The `name` attached to the auto-generated environment message. It is
 * deliberately different from the human's name (the git user name) so the
 * model can tell the two apart.
 */
export const ENVIRONMENT_MESSAGE_NAME = "environment";

/**
 * The system prompt contains only the agent's instructions; environment
 * context (working directory, time, git state) is sent separately as a
 * user-role message named `environment` (see buildEnvironmentMessage).
 */
export function buildSystemPrompt(session: StoredSession): string {
  return session.config.systemPrompt || SYSTEM_PROMPT;
}

/**
 * Environment context sent to the model as a user message named
 * `environment` (not as part of the system prompt): working directory,
 * session creation time, and simple git state (branch, commit, dirty files).
 * The git part is best-effort: when `cwd` is not a git repository (or git is
 * unavailable), the git lines are omitted.
 */
export async function buildEnvironmentMessage(
  session: StoredSession,
): Promise<string> {
  const lines = [
    `Working directory: ${session.cwd}`,
    `Current date/time: ${session.createdAt}`,
  ];
  const git = await readSimpleGitInfo(session.cwd);
  if (git) {
    lines.push(...git);
  }
  return lines.join("\n");
}

/**
 * The `name` shown for the human's own messages: `git config user.name`
 * (project config first, falling back to global), defaulting to "User" when
 * unavailable. Sanitized to the OpenAI `name` charset `^[a-zA-Z0-9_-]{0,64}$`.
 */
export async function getUserMessageName(cwd: string): Promise<string> {
  try {
    const { stdout } = await execFileAsync(
      "git",
      ["config", "--get", "user.name"],
      { cwd, timeout: GIT_TIMEOUT_MS, encoding: "utf8" },
    );
    const sanitized = sanitizeMessageName(stdout.trim());
    if (sanitized.length > 0) {
      return sanitized;
    }
  } catch {
    // git missing, not configured, or not a repository — fall through.
  }
  return "User";
}

/** Returns git lines, or null when cwd is not a git repository. */
async function readSimpleGitInfo(cwd: string): Promise<string[] | null> {
  try {
    const branch = await runGit(["rev-parse", "--abbrev-ref", "HEAD"], cwd);
    const commit = await runGit(["rev-parse", "--short", "HEAD"], cwd);
    const status = await runGit(["status", "--porcelain"], cwd);
    const changed = status.length > 0 ? status.split("\n").length : 0;
    const state =
      changed === 0
        ? "clean"
        : `${changed} changed file${changed === 1 ? "" : "s"}`;
    return [
      `Git branch: ${branch}`,
      `Git commit: ${commit}`,
      `Git status: ${state}`,
    ];
  } catch {
    return null;
  }
}

async function runGit(args: string[], cwd: string): Promise<string> {
  const { stdout } = await execFileAsync("git", args, {
    cwd,
    timeout: GIT_TIMEOUT_MS,
    encoding: "utf8",
  });
  return stdout.trim();
}

/** Keep only characters OpenAI accepts in message `name` fields. */
function sanitizeMessageName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}
