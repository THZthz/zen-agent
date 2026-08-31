import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  createStoredSession,
  deleteStoredSession,
  listStoredSessions,
  writeSession,
} from '../storage.js';

describe('deleteStoredSession', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-delete-'));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  it('rejects traversal ids without deleting the project or another session', async () => {
    const indexDirBefore = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = join(cwd, 'xdg');
    try {
      const safe = await createStoredSession(cwd);
      created.push(safe.cwd);
      const marker = join(cwd, 'project-marker.txt');
      writeFileSync(marker, 'keep');

      for (const sessionId of ['..', '../outside', 'nested/session', 'nested\\session']) {
        await expect(deleteStoredSession(cwd, sessionId)).rejects.toThrow(/Invalid session ID/);
      }

      expect(existsSync(marker)).toBe(true);
      expect(existsSync(join(cwd, '.sessions', safe.sessionId, 'state.json'))).toBe(true);
      expect((await listStoredSessions(cwd)).map((entry) => entry.sessionId)).toContain(
        safe.sessionId,
      );
    } finally {
      if (indexDirBefore === undefined) delete process.env.XDG_DATA_HOME;
      else process.env.XDG_DATA_HOME = indexDirBefore;
    }
  });

  it('removes the whole session tree, not just state.json', async () => {
    const indexDirBefore = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = join(cwd, 'xdg');
    try {
      const session = await createStoredSession(cwd);
      created.push(session.cwd);

      // Artifacts a real session accumulates.
      const terminals = join(cwd, '.sessions', session.sessionId, 'terminals');
      mkdirSync(terminals, { recursive: true });
      writeFileSync(join(terminals, 'output-1-c1.log'), 'out');
      writeFileSync(join(cwd, '.sessions', session.sessionId, 'llm.jsonl'), '{}\n');

      expect(existsSync(join(cwd, '.sessions', session.sessionId))).toBe(true);

      await deleteStoredSession(cwd, session.sessionId);

      expect(existsSync(join(cwd, '.sessions', session.sessionId))).toBe(false);
      const index = JSON.parse(
        await readFileOrNull(join(process.env.XDG_DATA_HOME!, 'zen-agent', 'index.json')),
      );
      expect(index[session.sessionId]).toBeUndefined();
    } finally {
      if (indexDirBefore === undefined) delete process.env.XDG_DATA_HOME;
      else process.env.XDG_DATA_HOME = indexDirBefore;
    }
  });
});

async function readFileOrNull(path: string): Promise<string> {
  const { readFile } = await import('node:fs/promises');
  try {
    return await readFile(path, 'utf8');
  } catch {
    return '{}';
  }
}

describe('listStoredSessions', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-list-'));
    process.env.XDG_DATA_HOME = join(cwd, 'xdg');
  });

  afterEach(() => {
    delete process.env.XDG_DATA_HOME;
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  it('lists indexed sessions from the index plus unindexed on-disk ones', async () => {
    const a = await createStoredSession(cwd);
    created.push(a.cwd);
    a.title = 'Indexed session';
    await writeSession(a);

    // An unindexed session: state.json on disk, no index entry.
    mkdirSync(join(cwd, '.sessions', 'sess_manual'), { recursive: true });
    writeFileSync(
      join(cwd, '.sessions', 'sess_manual', 'state.json'),
      JSON.stringify({
        sessionId: 'sess_manual',
        cwd,
        createdAt: '2026-01-02T00:00:00.000Z',
        updatedAt: '2026-01-02T00:00:00.000Z',
        title: 'Manual session',
        events: [],
        llmMessages: [],
        config: {},
        usage: {},
        turnStats: [],
      }),
    );

    const listed = await listStoredSessions(cwd);
    const byId = new Map(listed.map((entry) => [entry.sessionId, entry]));
    expect(byId.get(a.sessionId)?.title).toBe('Indexed session');
    expect(byId.get('sess_manual')?.title).toBe('Manual session');
    expect(listed.map((e) => e.sessionId)).toEqual(
      [...listed]
        .sort((x, y) => Date.parse(y.updatedAt ?? '') - Date.parse(x.updatedAt ?? ''))
        .map((e) => e.sessionId),
    );
  });
});
