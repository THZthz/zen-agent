import type { StoredSession } from "./storage.js";

export const SYSTEM_PROMPT = `You are an experiened software engineer.

You have exactly one tool: bash.
You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.
There is no approval gate: every command you run is executed immediately.
Prefer small, targeted bash commands. Avoid large output from using bash tool.
When modifying files, use shell tools such as cat, sed, awk, or tee. Prefer using trash (npm install --global trash-cli), rg, fdfind (fd), jq, uv if they exist.

> Always use utf-8, no emojis unless needed by your task.`;

export function buildSystemPrompt(session: StoredSession): string {
  const environmentInfo = [
    `Working directory: ${session.cwd}`,
    `Current date/time: ${session.createdAt}`,
  ].join("\n");
  const base = session.config.systemPrompt || SYSTEM_PROMPT;
  return `${base}\n---\n${environmentInfo}`;
}
