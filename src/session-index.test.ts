import { execFile } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  findSessionCwd,
  forgetSession,
  indexDirectory,
  readIndex,
  rememberSession,
} from './session-index.js';
import { emptySessionUsage, type StoredSession } from './storage.js';

const execFileAsync = promisify(execFile);

function makeSession(sessionId: string, cwd: string): StoredSession {
  const now = new Date().toISOString();
  return {
    sessionId,
    cwd,
    createdAt: now,
    updatedAt: now,
    title: sessionId,
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

describe('session index', () => {
  let root: string;
  let dataHome: string;
  let previousDataHome: string | undefined;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'zen-session-index-'));
    dataHome = join(root, 'xdg');
    previousDataHome = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = dataHome;
  });

  afterEach(() => {
    if (previousDataHome === undefined) delete process.env.XDG_DATA_HOME;
    else process.env.XDG_DATA_HOME = previousDataHome;
    rmSync(root, { recursive: true, force: true });
  });

  it('keeps concurrent in-process remember and forget mutations independent', async () => {
    const initial = Array.from({ length: 20 }, (_, index) =>
      makeSession(`sess_local_${index}`, root),
    );
    await Promise.all(initial.map((session) => rememberSession(session)));

    await Promise.all([
      forgetSession('sess_local_0'),
      rememberSession(makeSession('sess_local_new', root)),
      rememberSession({ ...initial[1]!, title: 'updated' }),
    ]);

    const index = await readIndex();
    expect(Object.keys(index)).toHaveLength(20);
    expect(index.sess_local_0).toBeUndefined();
    expect(index.sess_local_new?.cwd).toBe(root);
    expect(index.sess_local_1?.title).toBe('updated');
  });

  it('uses tombstones so legacy entries stay deleted and can later be restored', async () => {
    mkdirSync(indexDirectory(), { recursive: true });
    const filePath = join(indexDirectory(), 'index.json');
    writeFileSync(
      filePath,
      JSON.stringify({
        sess_legacy: {
          cwd: root,
          updatedAt: new Date().toISOString(),
          title: 'legacy',
        },
      }),
      'utf8',
    );

    expect(await findSessionCwd('sess_legacy')).toBe(root);
    await forgetSession('sess_legacy');
    expect(await findSessionCwd('sess_legacy')).toBeUndefined();
    expect((await readIndex()).sess_legacy).toBeUndefined();

    await rememberSession(makeSession('sess_legacy', root));
    expect(await findSessionCwd('sess_legacy')).toBe(root);
    expect((await readIndex()).sess_legacy?.cwd).toBe(root);
  });

  it('does not lose entries written concurrently by separate processes', async () => {
    const ids = Array.from({ length: 8 }, (_, index) => `sess_process_${index}`);
    const moduleUrl = new URL('./session-index.ts', import.meta.url).href;
    const childScript = `
      const { rememberSession } = await import(process.env.ZEN_TEST_INDEX_MODULE);
      const now = new Date().toISOString();
      await rememberSession({
        sessionId: process.env.ZEN_TEST_SESSION_ID,
        cwd: process.env.ZEN_TEST_CWD,
        updatedAt: now,
        title: process.env.ZEN_TEST_SESSION_ID,
      });
    `;

    await Promise.all(
      ids.map((sessionId) =>
        execFileAsync(
          process.execPath,
          ['--import', 'tsx', '--input-type=module', '--eval', childScript],
          {
            cwd: process.cwd(),
            env: {
              ...process.env,
              XDG_DATA_HOME: dataHome,
              ZEN_TEST_INDEX_MODULE: moduleUrl,
              ZEN_TEST_SESSION_ID: sessionId,
              ZEN_TEST_CWD: root,
            },
          },
        ),
      ),
    );

    const index = await readIndex();
    expect(Object.keys(index).sort()).toEqual([...ids].sort());
    await Promise.all(
      ids.map(async (sessionId) => {
        expect(await findSessionCwd(sessionId)).toBe(root);
      }),
    );
  }, 20_000);

  it('preserves corrupt legacy data without blocking independent entry writes', async () => {
    expect(Object.keys(await readIndex())).toEqual([]);

    mkdirSync(indexDirectory(), { recursive: true });
    const filePath = join(indexDirectory(), 'index.json');
    writeFileSync(filePath, '{not-json', 'utf8');

    await expect(readIndex()).rejects.toThrow(/Session index .* is corrupted/);
    await rememberSession(makeSession('sess_safe', root));
    expect(await findSessionCwd('sess_safe')).toBe(root);
    expect(readFileSync(filePath, 'utf8')).toBe('{not-json');

    rmSync(filePath, { force: true });
    mkdirSync(filePath);
    await expect(readIndex()).rejects.toThrow(/Failed to read session index/);
  });
});
