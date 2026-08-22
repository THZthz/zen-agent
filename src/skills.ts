import { homedir } from "node:os";
import { join } from "node:path";
import { readFile, readdir, stat } from "node:fs/promises";

/**
 * Agent Skills (the open format behind skills.sh) support.
 *
 * Skills are folders containing a `SKILL.md` (YAML frontmatter + Markdown
 * instructions) and optional `scripts/`, `references/` and `assets/`. Zed,
 * Zen Agent's ACP host, loads them from `~/.agents/skills/` (global) and
 * `<project>/.agents/skills/` (project-local) — the exact locations
 * `npx skills add <owner/repo> -a zed` writes to. Zed only exposes skills
 * to its built-in agent, not to ACP agents, so Zen Agent discovers and
 * lists them itself (see buildSkillsSection in system-prompt.ts).
 */

export type SkillScope = "project" | "global";

export interface SkillInfo {
  /** Skill name from the SKILL.md frontmatter (falls back to folder name). */
  name: string;
  /** What the skill does and when to use it (may be empty). */
  description: string;
  /** Absolute path to the skill folder (contains SKILL.md). */
  path: string;
  scope: SkillScope;
  /**
   * `disable-model-invocation: true` from the SKILL.md frontmatter: the
   * skill is meant to be invoked by the user (slash command) only, never
   * autonomously by the model.
   */
  disableModelInvocation: boolean;
}

const SKILL_FILE = "SKILL.md";

/** Global skills directory: `~/.agents/skills/` (matches Zed and skills.sh). */
export function globalSkillsDir(): string {
  return join(homedir(), ".agents", "skills");
}

/** Project-local skills directory: `<cwd>/.agents/skills/`. */
export function projectSkillsDir(cwd: string): string {
  return join(cwd, ".agents", "skills");
}

export interface SkillFrontmatter {
  name?: string;
  description?: string;
  disableModelInvocation?: boolean;
}

/**
 * Parse the YAML frontmatter block of a SKILL.md (a leading `---` delimited
 * block). Deliberately small and lenient: only the `name` and `description`
 * fields are read, and only as single-line scalars. Anything unrecognized
 * or malformed is ignored.
 */
export function parseSkillFrontmatter(content: string): SkillFrontmatter {
  const result: SkillFrontmatter = {};
  const lines = content.split(/\r?\n/);
  if (lines[0]?.trim() !== "---") {
    return result;
  }
  const end = lines.slice(1).findIndex((line) => line.trim() === "---");
  if (end === -1) {
    return result;
  }
  for (const raw of lines.slice(1, 1 + end)) {
    const line = raw.trim();
    if (line.length === 0 || line.startsWith("#")) {
      continue;
    }
    const colon = line.indexOf(":");
    if (colon === -1) {
      continue;
    }
    const key = line.slice(0, colon).trim();
    let value = line.slice(colon + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key === "name") {
      result.name = value || undefined;
    } else if (key === "description") {
      result.description = value || undefined;
    } else if (key === "disable-model-invocation") {
      result.disableModelInvocation =
        value === "true" || value === "yes" || value === "1";
    }
  }
  return result;
}

/** Reads one skill folder. Returns null when SKILL.md is missing/unreadable. */
export async function readSkill(
  dir: string,
  scope: SkillScope,
): Promise<SkillInfo | null> {
  let content: string;
  try {
    content = await readFile(join(dir, SKILL_FILE), "utf8");
  } catch {
    return null;
  }
  const meta = parseSkillFrontmatter(content);
  const name = (meta.name ?? dir.split("/").pop() ?? "").trim();
  if (name.length === 0) {
    return null;
  }
  return {
    name,
    description: meta.description ?? "",
    path: dir,
    scope,
    disableModelInvocation: meta.disableModelInvocation ?? false,
  };
}

export interface ListSkillsOptions {
  /** Override for the global skills dir (tests). */
  globalDir?: string;
  /** Override for the project skills dir (tests). */
  projectDir?: string;
}

/**
 * Discover all skills available to a session: project-local skills take
 * precedence over same-named global skills. `stat` (not `readdir`
 * isDirectory) is used so symlinked skill folders — how `npx skills add`
 * installs by default — are followed. Deterministic (sorted by name) so the
 * rendered catalog stays byte-identical for the cached LLM prefix.
 */
export async function listSkills(
  cwd: string,
  options: ListSkillsOptions = {},
): Promise<SkillInfo[]> {
  const byName = new Map<string, SkillInfo>();
  const scopes = [
    { scope: "global" as const, dir: options.globalDir ?? globalSkillsDir() },
    { scope: "project" as const, dir: options.projectDir ?? projectSkillsDir(cwd) },
  ];
  for (const { scope, dir } of scopes) {
    let entries: string[];
    try {
      entries = await readdir(dir);
    } catch {
      continue; // No skills directory — nothing to do.
    }
    for (const entryName of entries) {
      const skillPath = join(dir, entryName);
      try {
        const info = await stat(skillPath);
        if (!info.isDirectory()) {
          continue;
        }
      } catch {
        continue;
      }
      const skill = await readSkill(skillPath, scope);
      if (skill) {
        // Project scope wins over global for the same skill name.
        byName.set(skill.name, skill);
      }
    }
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Markdown catalog injected into the frozen Environment message so the
 * model knows which skills exist and how to load them through its existing
 * bash tool (Zed renders the `cat` as a normal terminal card). Skills are
 * invoked by hand only: the agent never loads one on its own.
 */
export function buildSkillsSection(skills: SkillInfo[]): string {
  const lines = [
    "## Skills",
    "",
    "Skills are reusable capability packages installed from the skills.sh registry (`.agents/skills/`).",
    "Each skill is a folder with `SKILL.md` (instructions) and optional `scripts/`, `references/`, and `assets/`.",
    "",
    "Skills are invoked by hand only: load a skill's `SKILL.md` with bash (`cat <path>/SKILL.md`) ONLY when the user explicitly asks for that skill by name. Never load a skill autonomously just because it matches the task. Follow the skill's instructions, including any files it references.",
    "",
    "Available skills:",
    "<skills>"
  ];
  for (const skill of skills) {
    const description = skill.description ? ` — ${skill.description}` : "";
    lines.push(
      `- ${skill.name} [${skill.scope}]${description} — load with: cat ${join(skill.path, SKILL_FILE)}`,
    );
  }
  lines.push("</skills>");
  return lines.join("\n");
}

/**
 * Read a skill's full `SKILL.md` body (frontmatter + instructions) as it
 * should be handed to the model when the skill is invoked.
 */
export async function readSkillMarkdown(skill: SkillInfo): Promise<string> {
  return readFile(join(skill.path, SKILL_FILE), "utf8");
}

/**
 * Build the user message that starts a model turn when a skill is invoked
 * through its `/skill-name` slash command: the user's argument plus the
 * skill's full SKILL.md so the model follows the skill's instructions
 * (reading referenced files/scripts with its bash tool as needed).
 */
export async function buildSkillInvocationPrompt(
  skill: SkillInfo,
  argument: string,
): Promise<string> {
  const content = (await readSkillMarkdown(skill)).trim();
  const lines = [
    "<skill-invoked>",
    `<skill-name>${skill.name}</skill-name>`,
  ];
  if (argument.length > 0) {
    lines.push(`<skill-argument>\n${argument}\n</skill-argument>`);
  }
  lines.push(
    "<skill-instruction>",
    `<skill-instruction-path>${join(skill.path, SKILL_FILE)}</skill-instruction-path>`,
    "<skill-instruction-content>",
    content,
    "</skill-instruction-content>",
    "</skill-instruction>",
  );
  lines.push("</skill-invoked>");
  return lines.join("\n");
}
