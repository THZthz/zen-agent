import { mkdirSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';

/**
 * Single-file SQLite persistence for everything Zen Agent stores: session
 * state, the LLM transcript, the per-process runtime log and terminal
 * input/output records. One database file serves every project (rows carry
 * the session's cwd), so concurrent agent processes share it through WAL
 * mode plus a busy timeout.
 *
 * node:sqlite is synchronous; the helpers here are intentionally sync and
 * the async storage API on top of it is preserved for callers.
 */

/** Default location: next to the package (dist/../zen-agent.db). */
export const DEFAULT_DB_FILE_NAME = 'zen-agent.db';

/** Resolved database file path: ZEN_AGENT_DB_FILE or the package-root default. */
export function dbFilePath(): string {
  const configured = process.env.ZEN_AGENT_DB_FILE?.trim();
  if (configured) {
    return isAbsolute(configured) ? configured : resolve(process.cwd(), configured);
  }
  return fileURLToPath(new URL(`../../${DEFAULT_DB_FILE_NAME}`, import.meta.url));
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS sessions (
  session_id        TEXT PRIMARY KEY,
  cwd               TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL,
  title             TEXT,
  config            TEXT NOT NULL,
  usage             TEXT NOT NULL,
  events            TEXT NOT NULL,
  llm_messages      TEXT NOT NULL,
  turn_stats        TEXT NOT NULL,
  cache_diagnostics TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_cwd ON sessions (cwd, updated_at);

CREATE TABLE IF NOT EXISTS llm_log (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  entry      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_log_by_session ON llm_log (session_id, seq);

CREATE TABLE IF NOT EXISTS runtime_log (
  seq         INTEGER PRIMARY KEY AUTOINCREMENT,
  startup_key TEXT NOT NULL,
  cwd         TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  entry       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runtime_log_by_startup ON runtime_log (startup_key, seq);

CREATE TABLE IF NOT EXISTS terminal_calls (
  id         TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  command    TEXT NOT NULL,
  output     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS terminal_calls_by_session ON terminal_calls (session_id, created_at);
`;

let cached: DatabaseSync | undefined;

/** Open (once per process) the shared database and make sure the schema exists. */
export function openDb(): DatabaseSync {
  if (!cached) {
    const path = dbFilePath();
    mkdirSync(dirname(path), { recursive: true });
    const db = new DatabaseSync(path);
    db.exec('PRAGMA journal_mode = WAL');
    db.exec('PRAGMA busy_timeout = 5000');
    db.exec(SCHEMA);
    cached = db;
  }
  return cached;
}

/** Append one LLM request/response transcript entry. Never throws. */
export function insertLlmLogEntry(sessionId: string, entry: Record<string, unknown>): void {
  try {
    openDb()
      .prepare('INSERT INTO llm_log (session_id, created_at, entry) VALUES (?, ?, ?)')
      .run(sessionId, new Date().toISOString(), JSON.stringify(entry));
  } catch {
    // Logging must never break the agent.
  }
}

/** Append one per-process runtime log entry. Never throws. */
export function insertRuntimeLogEntry(
  startupKey: string,
  cwd: string,
  entry: Record<string, unknown>,
): void {
  try {
    openDb()
      .prepare('INSERT INTO runtime_log (startup_key, cwd, created_at, entry) VALUES (?, ?, ?, ?)')
      .run(startupKey, cwd, new Date().toISOString(), JSON.stringify(entry));
  } catch {
    // Logging must never break the agent.
  }
}

/** Persist one bash tool call (command + full output). Never throws; returns
 * whether the row landed, so the caller can surface a store failure. */
export function insertTerminalCall(call: {
  id: string;
  sessionId: string;
  command: string;
  output: string;
}): boolean {
  try {
    openDb()
      .prepare(
        'INSERT INTO terminal_calls (id, session_id, created_at, command, output) VALUES (?, ?, ?, ?, ?)',
      )
      .run(call.id, call.sessionId, new Date().toISOString(), call.command, call.output);
    return true;
  } catch {
    return false;
  }
}
