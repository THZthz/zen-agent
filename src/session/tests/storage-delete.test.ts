import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  createStoredSession,
  deleteStoredSession,
  findSessionCwd,
  listStoredSessions,
  writeSession,
} from '../storage.js';
import { insertLlmLogEntry, insertTerminalCall, openDb } from '../db.js';

function sessionRowCount(sessionId: string): number {
  return (
    openDb().prepare('SELECT COUNT(*) AS n FROM sessions WHERE session_id = ?').get(sessionId) as {
      n: number;
    }
  ).n;
}

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

  it('rejects traversal ids without touching stored sessions', async () => {
    const safe = await createStoredSession(cwd);
    created.push(safe.cwd);

    for (const sessionId of ['..', '../outside', 'nested/session', 'nested\\session']) {
      await expect(deleteStoredSession(sessionId)).rejects.toThrow(/Invalid session ID/);
    }

    expect(sessionRowCount(safe.sessionId)).toBe(1);
    expect((await listStoredSessions(cwd)).map((entry) => entry.sessionId)).toContain(
      safe.sessionId,
    );
  });

  it('removes the session row and its transcripts and terminal records', async () => {
    const session = await createStoredSession(cwd);
    created.push(session.cwd);
    insertLlmLogEntry(session.sessionId, { type: 'llm_request' });
    insertTerminalCall({
      id: 'msg_test1',
      sessionId: session.sessionId,
      command: 'ls',
      output: 'out',
    });
    expect(
      openDb()
        .prepare('SELECT COUNT(*) AS n FROM terminal_calls WHERE session_id = ?')
        .get(session.sessionId),
    ).toMatchObject({ n: 1 });
    expect(
      openDb()
        .prepare('SELECT COUNT(*) AS n FROM llm_log WHERE session_id = ?')
        .get(session.sessionId),
    ).toMatchObject({ n: 1 });

    expect(sessionRowCount(session.sessionId)).toBe(1);
    await deleteStoredSession(session.sessionId);

    expect(sessionRowCount(session.sessionId)).toBe(0);
    expect(
      openDb()
        .prepare('SELECT COUNT(*) AS n FROM terminal_calls WHERE session_id = ?')
        .get(session.sessionId),
    ).toMatchObject({ n: 0 });
    expect(
      openDb()
        .prepare('SELECT COUNT(*) AS n FROM llm_log WHERE session_id = ?')
        .get(session.sessionId),
    ).toMatchObject({ n: 0 });
    expect(await findSessionCwd(session.sessionId)).toBeUndefined();
  });
});

describe('listStoredSessions', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-list-'));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  it('lists this project only, newest first', async () => {
    const a = await createStoredSession(cwd);
    created.push(a.cwd);
    a.title = 'Older session';
    a.updatedAt = '2026-01-01T00:00:00.000Z';
    await writeSession(a);

    const other = mkdtempSync(join(tmpdir(), 'zen-storage-list-other-'));
    created.push(other);
    const b = await createStoredSession(other);
    b.title = 'Other project';
    await writeSession(b);

    const listed = await listStoredSessions(cwd);
    expect(listed.map((entry) => entry.sessionId)).toEqual([a.sessionId]);
    expect(listed[0]).toMatchObject({
      sessionId: a.sessionId,
      cwd,
      title: 'Older session',
      updatedAt: '2026-01-01T00:00:00.000Z',
    });

    const everywhere = (await listStoredSessions()).filter((entry) =>
      [cwd, other].includes(entry.cwd),
    );
    expect(new Set(everywhere.map((entry) => entry.sessionId))).toEqual(
      new Set([a.sessionId, b.sessionId]),
    );
    // Newest first regardless of insertion order.
    expect(everywhere[0]?.sessionId).toBe(b.sessionId);
  });
});
