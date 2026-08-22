import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { sessionDirectory, type LlmMessage, type StoredSession } from "./storage.js";
import { buildSkillsSection, listSkills } from "./skills.js";

const execFileAsync = promisify(execFile);

const GIT_TIMEOUT_MS = 5_000;

export const SYSTEM_PROMPT = `You are an experiened software engineer.

You have exactly one tool: bash. You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation. There is no approval gate: every command you run is executed immediately. Prefer small, targeted bash commands. Avoid large output from using bash tool.

When modifying files, use shell tools such as cat, sed, awk, or tee. ALWAYS use trash (npm install --global trash-cli) instead of rm, rg instead of grep, fdfind (fd) instead of find if they exist. Prefer using uv to manage python.

> Always use utf-8, no emojis unless needed by your task.`;

/**
 * The `name` attached to the auto-generated environment message. It is
 * deliberately different from the human's name (the git user name) so the
 * model can tell the two apart.
 */
export const ENVIRONMENT_MESSAGE_NAME = "Environment";

/** True for the auto-generated environment/continuation user messages. */
export function isEnvironmentMessage(message: LlmMessage): boolean {
  return (
    message.role === "user" &&
    "name" in message &&
    message.name === ENVIRONMENT_MESSAGE_NAME
  );
}

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
 * session creation time, and simple git state (branch, dirty files), plus
 * git workflow guidance (Conventional Commits, focused incremental commits,
 * submodule handling). The git part is best-effort: when `cwd` is not a git
 * repository (or git is unavailable), the git lines are omitted.
 */
export async function buildEnvironmentMessage(
  session: StoredSession,
): Promise<string> {
  const lines = [
    "<environment>",
    "<session-state>fresh-started</session-state>",
    `<session-transcript>${sessionDirectory(session.cwd)}</session-transcript>`,
    `<working-directory>${session.cwd}</working-directory>`,
    `<current-time>${session.createdAt}</current-time>`,
  ];
  const git = await readSimpleGitInfo(session.cwd);
  if (git) {
    lines.push(...git);
    lines.push(
      "",
      "<git-remainder>",
      "> Follow Conventional Commits; keep the commit message body concise.",
      "> Split your changes into multiple commits if needed; each commit should be focused on a single purpose; commit as you work.",
    );
    const submodules = await readSubmodulePaths(session.cwd);
    if (submodules.length > 0) {
      lines.push(
        `> This project contains git submodules: ${submodules.join(", ")}.`,
      );
    }
    lines.push("</git-remainder>");
  }
  // Skills are opt-in and frozen at session creation: with
  // ZEN_AGENT_SHOW_SKILLS_CATALOG=1 the catalog (and everything else in this
  // message) stays byte-identical for DeepSeek's prefix cache, and skills
  // installed after session creation are picked up by the next session. By
  // default Zen Agent stays minimal and passes no skill information at all.
  if (process.env.ZEN_AGENT_SHOW_SKILLS_CATALOG === "1") {
    const skills = await listSkills(session.cwd);
    if (skills.length > 0) {
      lines.push("", buildSkillsSection(skills));
    }
  }
  lines.push("</environment>");
  return lines.join("\n");
}

/**
 * Environment notification appended when a session is loaded/resumed after a
 * restart. Appended at the END of the conversation (after all history), so
 * it never touches the cached prefix: the frozen environment message plus
 * the persisted history stay byte-identical, keeping DeepSeek's context
 * cache hit ratio intact across restarts. The model sees a fresh snapshot of
 * the environment (now, git state) and knows the session was continued.
 */
export async function buildSessionContinuedMessage(
  session: StoredSession,
): Promise<string> {
  const lines = [
    "<environment>",
    "<session-state>resumed</session-state>",
    `<current-time>${new Date().toISOString()}</current-time>`,
  ];
  const git = await readSimpleGitInfo(session.cwd);
  if (git) {
    lines.push(...git);
  }
  lines.push("</environment>");
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
    const status = await runGit(["status", "--porcelain"], cwd);
    const changed = status.length > 0 ? status.split("\n").length : 0;
    const state =
      changed === 0
        ? "clean"
        : `${changed} changed file${changed === 1 ? "" : "s"}`;
    return [
      `<git-branch>${branch}</git-branch>`,
      `<git-status>${state}</git-status>`,
    ];
  } catch {
    return null;
  }
}

/**
 * Best-effort list of submodule paths via `git submodule status` (one line
 * per submodule, path is the second column). Returns [] when the repo has
 * no submodules or git is unavailable.
 */
async function readSubmodulePaths(cwd: string): Promise<string[]> {
  try {
    const { stdout } = await execFileAsync(
      "git",
      ["submodule", "status"],
      { cwd, timeout: GIT_TIMEOUT_MS, encoding: "utf8" },
    );
    return stdout
      .split("\n")
      .map((line) => line.trim().split(/\s+/)[1])
      .filter((path): path is string => Boolean(path));
  } catch {
    return [];
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
