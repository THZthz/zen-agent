import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  buildSkillInvocationPrompt,
  buildSkillsSection,
  listSkills,
  parseSkillFrontmatter,
  projectSkillsDir,
  readSkill,
  readSkillMarkdown,
} from '../skills.js';
import { buildEnvironmentMessage } from '../system-prompt.js';
import { emptySessionUsage, type StoredSession } from '../storage.js';

let homeDir: string;
let projectDir: string;

beforeEach(() => {
  homeDir = mkdtempSync(join(tmpdir(), 'zen-agent-skills-home-'));
  projectDir = mkdtempSync(join(tmpdir(), 'zen-agent-skills-project-'));
  process.env.HOME = homeDir;
});

afterEach(() => {
  delete process.env.HOME;
  delete process.env.ZEN_AGENT_SHOW_SKILLS_CATALOG;
  rmSync(homeDir, { recursive: true, force: true });
  rmSync(projectDir, { recursive: true, force: true });
});

function writeSkill(root: string, name: string, content: string): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'SKILL.md'), content, 'utf8');
  return dir;
}

function makeSession(cwd: string): StoredSession {
  return {
    sessionId: 's1',
    cwd,
    createdAt: '2026-08-20T02:00:00.000Z',
    updatedAt: '2026-08-20T02:00:00.000Z',
    title: null,
    events: [],
    llmMessages: [],
    config: {
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      thinkingEffort: 'off',
      systemPrompt: '',
      sandbox: false,
      roBindEnabled: false,
      toolsEnabled: true,
    },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

describe('parseSkillFrontmatter', () => {
  it('parses name and description', () => {
    expect(
      parseSkillFrontmatter(
        '---\nname: frontend-design\ndescription: Build polished frontends.\n---\n## Instructions\n...',
      ),
    ).toEqual({
      name: 'frontend-design',
      description: 'Build polished frontends.',
    });
  });

  it('unquotes quoted scalars and parses disable-model-invocation', () => {
    expect(
      parseSkillFrontmatter(
        '---\nname: "my-skill"\ndescription: "A skill"\ndisable-model-invocation: true\n---\nbody',
      ),
    ).toEqual({
      name: 'my-skill',
      description: 'A skill',
      disableModelInvocation: true,
    });
    expect(
      parseSkillFrontmatter('---\nname: x\ndisable-model-invocation: false\n---\n')
        .disableModelInvocation,
    ).toBe(false);
    expect(
      parseSkillFrontmatter('---\nname: x\ndisable-model-invocation: 1\n---\n')
        .disableModelInvocation,
    ).toBe(true);
  });

  it('returns an empty object without a leading frontmatter block', () => {
    expect(parseSkillFrontmatter('# just markdown')).toEqual({});
    expect(parseSkillFrontmatter('---\nname: x\n(unterminated')).toEqual({});
  });

  it('ignores unrecognized keys', () => {
    expect(parseSkillFrontmatter('---\nname: x\nlicense: MIT\nfoo: bar\n---\n')).toEqual({
      name: 'x',
    });
  });
});

describe('readSkill', () => {
  it('returns skill info from a folder with SKILL.md', async () => {
    const dir = writeSkill(
      projectDir,
      'frontend-design',
      '---\nname: frontend-design\ndescription: Build polished frontends.\n---\n## Instructions',
    );
    expect(await readSkill(dir, 'project')).toEqual({
      name: 'frontend-design',
      description: 'Build polished frontends.',
      path: dir,
      scope: 'project',
      disableModelInvocation: false,
    });
  });

  it('reads disable-model-invocation from the frontmatter', async () => {
    const dir = writeSkill(
      projectDir,
      'grill-me',
      '---\nname: grill-me\ndescription: A relentless interview.\ndisable-model-invocation: true\n---\nbody',
    );
    const skill = await readSkill(dir, 'project');
    expect(skill?.disableModelInvocation).toBe(true);
  });

  it('falls back to the folder name when frontmatter has no name', async () => {
    const dir = writeSkill(projectDir, 'unnamed-skill', '# just markdown\n');
    const skill = await readSkill(dir, 'project');
    expect(skill?.name).toBe('unnamed-skill');
    expect(skill?.description).toBe('');
  });

  it('returns null when SKILL.md is missing', async () => {
    const dir = join(projectDir, 'no-skill');
    mkdirSync(dir, { recursive: true });
    expect(await readSkill(dir, 'project')).toBeNull();
  });
});

describe('listSkills', () => {
  it('discovers global and project skills and sorts by name', async () => {
    writeSkill(
      join(homeDir, '.agents', 'skills'),
      'zzz-global',
      '---\nname: zzz-global\ndescription: Global skill\n---\n',
    );
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'aaa-project',
      '---\nname: aaa-project\ndescription: Project skill\n---\n',
    );
    const skills = await listSkills(projectDir);
    expect(skills.map((s) => s.name)).toEqual(['aaa-project', 'zzz-global']);
    expect(skills.map((s) => s.scope)).toEqual(['project', 'global']);
  });

  it('lets the project skill shadow a same-named global skill', async () => {
    writeSkill(
      join(homeDir, '.agents', 'skills'),
      'shared',
      '---\nname: shared\ndescription: Global version\n---\n',
    );
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'shared',
      '---\nname: shared\ndescription: Project version\n---\n',
    );
    const skills = await listSkills(projectDir);
    expect(skills).toHaveLength(1);
    expect(skills[0]?.description).toBe('Project version');
    expect(skills[0]?.scope).toBe('project');
  });

  it('follows symlinked skill folders (npx skills add default install)', async () => {
    const real = writeSkill(
      projectDir,
      'real-skill',
      '---\nname: symlinked-skill\ndescription: via symlink\n---\n',
    );
    mkdirSync(join(projectDir, '.agents', 'skills'), { recursive: true });
    symlinkSync(real, join(projectDir, '.agents', 'skills', 'symlinked-skill'));
    const skills = await listSkills(projectDir);
    expect(skills.map((s) => s.name)).toEqual(['symlinked-skill']);
  });

  it('ignores missing directories, non-directories, and skills without SKILL.md', async () => {
    mkdirSync(join(projectDir, '.agents', 'skills'), { recursive: true });
    writeFileSync(join(projectDir, '.agents', 'skills', 'not-a-skill'), 'hello', 'utf8');
    mkdirSync(join(projectDir, '.agents', 'skills', 'empty-dir'), { recursive: true });
    expect(await listSkills(join(projectDir, 'nonexistent'))).toEqual([]);
    expect(await listSkills(projectDir)).toEqual([]);
  });

  it('supports explicit directory overrides', async () => {
    writeSkill(join(homeDir, '.agents', 'skills'), 'g', '---\nname: g\ndescription: g\n---\n');
    writeSkill(join(projectDir, '.agents', 'skills'), 'p', '---\nname: p\ndescription: p\n---\n');
    const skills = await listSkills(projectDir, {
      globalDir: join(homeDir, '.agents', 'skills'),
      projectDir: join(projectDir, '.agents', 'skills'),
    });
    expect(skills.map((s) => s.name)).toEqual(['g', 'p']);
  });
});

describe('buildSkillsSection', () => {
  it('renders the catalog with load commands and manual-only instructions', async () => {
    const auto = writeSkill(
      join(homeDir, '.agents', 'skills'),
      'auto-skill',
      '---\nname: auto-skill\ndescription: Auto skill\n---\n',
    );
    const manual = writeSkill(
      join(projectDir, '.agents', 'skills'),
      'manual-skill',
      '---\nname: manual-skill\ndescription: Manual skill\n---\n',
    );
    const skills = await listSkills(projectDir);
    const section = buildSkillsSection(skills);
    expect(section).toContain('## Skills');
    expect(section).toContain('invoked by hand only');
    expect(section).toContain(
      `- auto-skill [global] — Auto skill — load with: cat ${join(auto, 'SKILL.md')}`,
    );
    expect(section).toContain(
      `- manual-skill [project] — Manual skill — load with: cat ${join(manual, 'SKILL.md')}`,
    );
  });
});

describe('buildEnvironmentMessage integration', () => {
  it('includes the skills catalog only with ZEN_AGENT_SHOW_SKILLS_CATALOG=1', async () => {
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'frontend-design',
      '---\nname: frontend-design\ndescription: Build polished frontends.\n---\n',
    );
    process.env.ZEN_AGENT_SHOW_SKILLS_CATALOG = '1';
    const message = await buildEnvironmentMessage(makeSession(projectDir));
    expect(message).toContain('## Skills');
    expect(message).toContain('frontend-design');
    expect(message).toContain(
      'cat ' + join(projectDir, '.agents', 'skills', 'frontend-design', 'SKILL.md'),
    );
  });

  it('passes no skill information by default, even when skills are installed', async () => {
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'frontend-design',
      '---\nname: frontend-design\ndescription: Build polished frontends.\n---\n',
    );
    const message = await buildEnvironmentMessage(makeSession(projectDir));
    expect(message).not.toContain('## Skills');
    expect(message).not.toContain('frontend-design');
  });

  it('omits the skills section when no skills are installed', async () => {
    process.env.ZEN_AGENT_SHOW_SKILLS_CATALOG = '1';
    const message = await buildEnvironmentMessage(makeSession(projectDir));
    expect(message).not.toContain('## Skills');
  });

  it("uses the project skills dir matching Zed's layout", () => {
    expect(projectSkillsDir('/a/b')).toBe('/a/b/.agents/skills');
    expect(existsSync(projectDir)).toBe(true);
  });
});

describe('readSkillMarkdown', () => {
  it('returns the full SKILL.md content', async () => {
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'grill-me',
      '---\nname: grill-me\ndescription: Grill me.\n---\nInterview the user relentlessly.',
    );
    const skills = await listSkills(projectDir);
    expect(await readSkillMarkdown(skills[0]!)).toBe(
      '---\nname: grill-me\ndescription: Grill me.\n---\nInterview the user relentlessly.',
    );
  });
});

describe('buildSkillInvocationPrompt', () => {
  it("injects the skill instructions and the user's argument", async () => {
    writeSkill(
      join(projectDir, '.agents', 'skills'),
      'grill-me',
      '---\nname: grill-me\ndescription: A relentless interview.\ndisable-model-invocation: true\n---\nInterview the user relentlessly.',
    );
    const skills = await listSkills(projectDir);
    const prompt = await buildSkillInvocationPrompt(skills[0]!, 'my plan');
    expect(prompt).toContain('<skill-invoked>');
    expect(prompt).toContain('<skill-name>grill-me</skill-name>');
    expect(prompt).toContain('<skill-argument>\nmy plan\n</skill-argument>');
    expect(prompt).toContain('<skill-path>');
    expect(prompt).toContain('Interview the user relentlessly.');
    expect(prompt).toContain('</skill-invoked>');
  });

  it('omits the argument line when the argument is empty', async () => {
    writeSkill(join(projectDir, '.agents', 'skills'), 'grill-me', '---\nname: grill-me\n---\nbody');
    const skills = await listSkills(projectDir);
    const prompt = await buildSkillInvocationPrompt(skills[0]!, '');
    expect(prompt).not.toContain('<skill-argument>');
    expect(prompt).toContain('body');
  });
});
