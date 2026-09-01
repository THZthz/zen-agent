import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  dbFilePath,
  insertLlmLogEntry,
  insertRuntimeLogEntry,
  insertTerminalCall,
  openDb,
} from '../db.js';
import {
  createStoredSession,
  findSessionCwd,
  readStoredSession,
  writeSession,
} from '../storage.js';

describe('db', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-db-'));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  it('resolves the db file from ZEN_AGENT_DB_FILE or the package root', () => {
    const previous = process.env.ZEN_AGENT_DB_FILE;
    try {
      delete process.env.ZEN_AGENT_DB_FILE;
      // Default: next to the package (dist/../zen-agent.db in a build).
      expect(dbFilePath()).toMatch(/zen-agent\.db$/);
      process.env.ZEN_AGENT_DB_FILE = 'relative.db';
      expect(dbFilePath()).toBe(join(process.cwd(), 'relative.db'));
      process.env.ZEN_AGENT_DB_FILE = '/absolute/zen.db';
      expect(dbFilePath()).toBe('/absolute/zen.db');
    } finally {
      if (previous === undefined) delete process.env.ZEN_AGENT_DB_FILE;
      else process.env.ZEN_AGENT_DB_FILE = previous;
    }
  });

  it('stores session rows in WAL mode with the schema applied', () => {
    const db = openDb();
    expect(db.prepare('PRAGMA journal_mode').get()).toMatchObject({ journal_mode: 'wal' });
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(tables.map((t) => t.name)).toEqual(
      expect.arrayContaining(['sessions', 'llm_log', 'runtime_log', 'terminal_calls']),
    );
  });

  it('upserts session rows and reads them back', async () => {
    const session = await createStoredSession(cwd);
    created.push(session.cwd);
    session.title = 'Renamed';
    session.usage.turns = 4;
    session.llmMessages.push({ role: 'user', content: 'hi' });
    await writeSession(session);

    const reloaded = (await readStoredSession(cwd, session.sessionId)).session;
    expect(reloaded.title).toBe('Renamed');
    expect(reloaded.usage.turns).toBe(4);
    expect(reloaded.llmMessages).toEqual([{ role: 'user', content: 'hi' }]);
    // One row only: the insert replaced, not duplicated.
    expect(
      openDb()
        .prepare('SELECT COUNT(*) AS n FROM sessions WHERE session_id = ?')
        .get(session.sessionId),
    ).toMatchObject({ n: 1 });
    expect(await findSessionCwd(session.sessionId)).toBe(cwd);
  });

  it('appends llm, runtime and terminal records', async () => {
    const session = await createStoredSession(cwd);
    created.push(session.cwd);

    insertLlmLogEntry(session.sessionId, { type: 'llm_request', step: 1 });
    insertLlmLogEntry(session.sessionId, { type: 'llm_response', step: 1 });
    insertRuntimeLogEntry('2026-09-01-00-00-00_x', cwd, { level: 'info', message: 'hello' });
    expect(
      insertTerminalCall({
        id: 'msg_t1',
        sessionId: session.sessionId,
        command: 'echo hi',
        output: 'hi',
      }),
    ).toBe(true);

    const llmRows = openDb()
      .prepare('SELECT entry FROM llm_log WHERE session_id = ? ORDER BY seq')
      .all(session.sessionId) as Array<{ entry: string }>;
    expect(llmRows.map((row) => JSON.parse(row.entry))).toEqual([
      { type: 'llm_request', step: 1 },
      { type: 'llm_response', step: 1 },
    ]);

    const runtimeRows = openDb()
      .prepare('SELECT entry, cwd, startup_key FROM runtime_log WHERE startup_key = ?')
      .all('2026-09-01-00-00-00_x') as Array<{ entry: string; cwd: string; startup_key: string }>;
    expect(runtimeRows).toHaveLength(1);
    expect(JSON.parse(runtimeRows[0]!.entry)).toMatchObject({ message: 'hello' });
    expect(runtimeRows[0]!.cwd).toBe(cwd);
    expect(runtimeRows[0]!.startup_key).toBe('2026-09-01-00-00-00_x');

    const terminal = openDb()
      .prepare('SELECT * FROM terminal_calls WHERE id = ?')
      .get('msg_t1') as Record<string, unknown>;
    expect(terminal).toMatchObject({
      id: 'msg_t1',
      session_id: session.sessionId,
      command: 'echo hi',
      output: 'hi',
    });
  });

  it('never throws on duplicate terminal ids or corrupt logging payloads', () => {
    expect(
      insertTerminalCall({ id: 'msg_dup', sessionId: 'sess_x', command: 'a', output: 'b' }),
    ).toBe(true);
    expect(
      insertTerminalCall({ id: 'msg_dup', sessionId: 'sess_x', command: 'a', output: 'b' }),
    ).toBe(false);
    // Circular structures are swallowed like the former appendFile failures.
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => insertLlmLogEntry('sess_x', circular)).not.toThrow();
    expect(() => insertRuntimeLogEntry('k', cwd, circular)).not.toThrow();
  });
});
