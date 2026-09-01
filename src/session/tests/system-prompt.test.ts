import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  buildEnvironmentMessage,
  buildSessionContinuedMessage,
  buildSystemPrompt,
  getUserMessageName,
  isEnvironmentMessage,
} from '../system-prompt.js';
import { emptySessionUsage, type StoredSession } from '../storage.js';

let dir: string;
let configDir: string;
let homeDir: string;
let subDir: string | undefined;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'zen-agent-git-'));
  // Isolate git from the machine's global/system config so tests are
  // deterministic regardless of the host's git user.name. The config file
  // lives OUTSIDE the repo so it never shows up in `git status`.
  configDir = mkdtempSync(join(tmpdir(), 'zen-agent-gitcfg-'));
  writeFileSync(join(configDir, 'gitconfig'), '');
  process.env.GIT_CONFIG_GLOBAL = join(configDir, 'gitconfig');
  process.env.GIT_CONFIG_SYSTEM = '/dev/null';
  // Isolate HOME too: buildEnvironmentMessage discovers skills under
  // ~/.agents/skills, so a real HOME would leak the developer's skills
  // into these assertions.
  homeDir = mkdtempSync(join(tmpdir(), 'zen-agent-home-'));
  process.env.HOME = homeDir;
});

afterEach(() => {
  delete process.env.GIT_CONFIG_GLOBAL;
  delete process.env.GIT_CONFIG_SYSTEM;
  delete process.env.HOME;
  rmSync(dir, { recursive: true, force: true });
  rmSync(configDir, { recursive: true, force: true });
  rmSync(homeDir, { recursive: true, force: true });
  if (subDir) {
    rmSync(subDir, { recursive: true, force: true });
    subDir = undefined;
  }
});

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
      toolsEnabled: true,
    },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

function git(args: string[]): void {
  execFileSync('git', args, { cwd: dir, stdio: 'ignore' });
}

describe('getUserMessageName', () => {
  it('returns the git user name, sanitized to the OpenAI name charset', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'John Doe']);
    git(['config', 'user.email', 'john@example.com']);
    expect(await getUserMessageName(dir)).toBe('John_Doe');
  });

  it("falls back to 'User' when git has no user.name", async () => {
    git(['init', '-b', 'main']);
    // No user.name configured anywhere.
    expect(await getUserMessageName(dir)).toBe('User');
  });

  it("falls back to 'User' outside a git repository", async () => {
    expect(await getUserMessageName(dir)).toBe('User');
  });
});

describe('buildEnvironmentMessage', () => {
  it('includes working directory and date even without git', async () => {
    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain('<environment>');
    expect(text).toContain(`<working-directory>${dir}</working-directory>`);
    expect(text).toContain('<current-time>2026-08-20T02:00:00.000Z</current-time>');
    expect(text).toContain('</environment>');
  });

  it('adds branch and clean status inside a git repository', async () => {
    git(['init', '-b', 'feature/test']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain(`<working-directory>${dir}</working-directory>`);
    expect(text).toContain('<git-branch>feature/test</git-branch>');
    expect(text).toContain('<git-status>clean</git-status>');
    expect(text).not.toContain('submodule');
  });

  it('reports the number of changed files when the tree is dirty', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);
    writeFileSync(join(dir, 'a.txt'), 'hello, world\n');
    writeFileSync(join(dir, 'b.txt'), 'new\n');

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain('<git-branch>main</git-branch>');
    expect(text).toContain('<git-status>2 changed files</git-status>');
  });

  it('reminds the agent to follow Conventional Commits with a concise body', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain('> Follow Conventional Commits; keep the commit message body concise.');
  });

  it('reminds the agent to split work into focused commits while working', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain(
      '> Split your changes into multiple commits if needed; each commit should be focused on a single purpose; commit as you work.',
    );
  });

  it('notifies the agent about submodules and not to bump the pointer', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);

    // A real submodule: source repo lives outside `dir` so it is not
    // picked up as an embedded repository by `git add`.
    subDir = mkdtempSync(join(tmpdir(), 'zen-agent-sub-'));
    execFileSync('git', ['init', '-b', 'main'], {
      cwd: subDir,
      stdio: 'ignore',
    });
    execFileSync('git', ['config', 'user.name', 'Tester'], {
      cwd: subDir,
      stdio: 'ignore',
    });
    execFileSync('git', ['config', 'user.email', 'tester@example.com'], {
      cwd: subDir,
      stdio: 'ignore',
    });
    writeFileSync(join(subDir, 'f.txt'), 'hello\n');
    execFileSync('git', ['add', 'f.txt'], { cwd: subDir, stdio: 'ignore' });
    execFileSync('git', ['commit', '-m', 'init'], {
      cwd: subDir,
      stdio: 'ignore',
    });

    git(['-c', 'protocol.file.allow=always', 'submodule', 'add', subDir, 'submod']);
    git(['commit', '-am', 'add submodule']);

    const text = await buildEnvironmentMessage(makeSession(dir));
    expect(text).toContain('> This project contains git submodules: submod.');
  });
});

describe('buildSessionContinuedMessage', () => {
  it('marks the continuation and includes fresh git state', async () => {
    git(['init', '-b', 'main']);
    git(['config', 'user.name', 'Tester']);
    git(['config', 'user.email', 'tester@example.com']);
    writeFileSync(join(dir, 'a.txt'), 'hello\n');
    git(['add', 'a.txt']);
    git(['commit', '-m', 'init']);
    writeFileSync(join(dir, 'a.txt'), 'changed\n');

    const text = await buildSessionContinuedMessage(makeSession(dir));
    expect(text).toContain('<environment>');
    expect(text).toContain('<session-state>resumed</session-state>');
    expect(text).toContain('<current-time>');
    expect(text).toContain('<git-branch>main</git-branch>');
    expect(text).toContain('<git-status>1 changed file</git-status>');
    // Workflow guidance lives in the initial environment message, not here,
    // so the frozen cached prefix stays byte-identical across restarts.
    expect(text).not.toContain('Follow Conventional Commits');
  });
});

describe('isEnvironmentMessage', () => {
  it('detects only auto-generated environment user messages', () => {
    expect(isEnvironmentMessage({ role: 'user', content: 'x', name: 'Environment' })).toBe(true);
    expect(isEnvironmentMessage({ role: 'user', content: 'x', name: 'Amias' })).toBe(false);
    expect(isEnvironmentMessage({ role: 'user', content: 'x' })).toBe(false);
    expect(isEnvironmentMessage({ role: 'assistant', content: [] })).toBe(false);
  });
});

describe('buildSystemPrompt tools gating', () => {
  it('includes the tool guidance when tools are enabled', () => {
    const prompt = buildSystemPrompt(makeSession(dir));
    expect(prompt).toContain(
      'You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.',
    );
    // The prompt is modality-independent: read_media is described by its tool
    // schema alone, so the lazy modality lookup can never churn the cached
    // prefix via the system prompt.
    expect(prompt).not.toContain('read_media');
    expect(prompt).not.toContain('<read-media-tool>');
  });

  it('omits every tool reference when /tools off', () => {
    const session = makeSession(dir);
    session.config.toolsEnabled = false;
    const prompt = buildSystemPrompt(session);
    expect(prompt).not.toContain('You can use bash to inspect files');
    expect(prompt).not.toContain('use shell tools such as `cat`, `sed`, `awk`, or `tee`');
    expect(prompt).not.toContain('<toolbox>');
    expect(prompt).not.toMatch(/bash/i);
    expect(prompt).toContain('You are an experienced and prudent software engineer');
    expect(prompt).toContain('Always use utf-8');
    expect(prompt).toContain('<system-prompt>');
    expect(prompt).toContain('</system-prompt>');
  });

  it('keeps the enabled prompt byte-identical with and without the no-tools split', () => {
    // The /tools-off prompt must not change the tools-on prompt: the system
    // prompt is part of the provider's cache prefix.
    const session = makeSession(dir);
    session.config.toolsEnabled = true;
    const prompt = buildSystemPrompt(session);
    expect(prompt).toBe(
      [
        '<system-prompt>',
        '<persona>',
        'You are an experienced and prudent software engineer.',
        '</persona>',
        '',
        '<principles>',
        '<workflow>',
        "**You approach every code change with caution.** Code is not an asset—it is a liability. Your goal is to achieve clarity, modularity, and maintainability. Upon receiving a task, you first gather sufficient information, then clarify the user's requirements to make sure you understand exactly what needs to be done. Next, you carefully review the code and concisely explain your proposed plan to the user. You proceed with editing only after user's confirmation.",
        '</workflow>',
        '<think-before-coding>',
        "**Don't assume. Don't hide confusion. Surface tradeoffs.** State your assumptions explicitly; if uncertain, ask. If multiple interpretations exist, present them - don't pick silently. If something is unclear, stop; name what's confusing and ask.",
        '</think-before-coding>',
        '<simplicity-first>',
        '**Minimum code that solves the problem. Nothing speculative.** No features beyond what was asked. No abstractions for single-use code. No "flexibility" or "configurability" that wasn\'t requested. No error handling for impossible scenarios. If you write 200 lines and it could be 50, rewrite it. Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.',
        '</simplicity-first>',
        '</principles>',
        '',
        '<toolbox>',
        '<bash-tool>',
        'You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation. Prefer small, targeted bash commands. Avoid large output from using bash tool. When modifying files, use shell tools such as `cat`, `sed`, `awk`, or `tee`. ALWAYS use `trash` instead of `rm`, `rg` instead of `grep`, `fdfind` (`fd`) instead of `find` if they exist. Prefer using `uv` to manage python.',
        '</bash-tool>',
        '</toolbox>',
        '',
        '<reminder>',
        '> Always use utf-8, no emojis unless needed by your task.',
        '</reminder>',
        '</system-prompt>',
      ].join('\n'),
    );
  });

  it('leaves a custom system prompt untouched regardless of tools', () => {
    const session = makeSession(dir);
    session.config.systemPrompt = 'custom instructions';
    session.config.toolsEnabled = false;
    expect(buildSystemPrompt(session)).toBe('custom instructions');
  });
});
