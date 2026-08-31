import { randomBytes } from 'node:crypto';
import { mkdir, readFile, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import type { SessionInfo } from '@agentclientprotocol/sdk';
import {
  readStoredSession,
  sessionDirectory,
  validateSessionId,
  type StoredSession,
} from './storage.js';

interface SessionIndexEntry {
  cwd: string;
  updatedAt: string;
  title?: string | null;
}

interface SessionIndex {
  [sessionId: string]: SessionIndexEntry;
}

type StoredIndexEntry = SessionIndexEntry | { deleted: true };

async function ensureDirectory(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
}

/** Crash-safe replace used by session state, model metadata and index entries. */
export async function writeFileAtomic(filePath: string, contents: string): Promise<void> {
  await ensureDirectory(dirname(filePath));
  const tmp = `${filePath}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    await writeFile(tmp, contents, 'utf8');
    await rename(tmp, filePath);
  } catch (error) {
    await unlink(tmp).catch(() => {});
    throw error;
  }
}

export function indexDirectory(): string {
  const dataHome = process.env.XDG_DATA_HOME ?? join(homedir(), '.local', 'share');
  return join(dataHome, 'zen-agent');
}

function legacyIndexPath(): string {
  return join(indexDirectory(), 'index.json');
}

function entriesDirectory(): string {
  return join(indexDirectory(), 'sessions');
}

function entryPath(sessionId: string): string {
  validateSessionId(sessionId);
  return join(entriesDirectory(), `${sessionId}.json`);
}

function isErrnoException(error: unknown, code: string): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    (error as NodeJS.ErrnoException).code === code
  );
}

function parseIndexEntry(raw: unknown, context: string): StoredIndexEntry {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`Session index ${context} is corrupted: expected an object`);
  }
  const value = raw as Record<string, unknown>;
  if (value.deleted === true) {
    return { deleted: true };
  }
  if (
    typeof value.cwd !== 'string' ||
    typeof value.updatedAt !== 'string' ||
    (value.title !== undefined && value.title !== null && typeof value.title !== 'string')
  ) {
    throw new Error(`Session index ${context} is corrupted: invalid entry`);
  }
  return {
    cwd: value.cwd,
    updatedAt: value.updatedAt,
    title: value.title as string | null | undefined,
  };
}

function parseJson(raw: string, context: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch (error) {
    throw new Error(
      `Session index ${context} is corrupted: not valid JSON (${error instanceof Error ? error.message : String(error)})`,
      { cause: error },
    );
  }
}

async function readLegacyIndex(): Promise<SessionIndex> {
  const filePath = legacyIndexPath();
  let raw: string;
  try {
    raw = await readFile(filePath, 'utf8');
  } catch (error) {
    if (isErrnoException(error, 'ENOENT')) {
      return Object.create(null) as SessionIndex;
    }
    throw new Error(
      `Failed to read session index ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }

  const parsed = parseJson(raw, filePath);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`Session index ${filePath} is corrupted: expected an object`);
  }

  const index: SessionIndex = Object.create(null) as SessionIndex;
  for (const [sessionId, rawEntry] of Object.entries(parsed)) {
    validateSessionId(sessionId);
    const entry = parseIndexEntry(rawEntry, `${filePath} entry ${sessionId}`);
    if ('deleted' in entry) {
      throw new Error(`Session index ${filePath} is corrupted: invalid legacy entry ${sessionId}`);
    }
    index[sessionId] = entry;
  }
  return index;
}

async function readStoredIndexEntry(filePath: string): Promise<StoredIndexEntry> {
  let raw: string;
  try {
    raw = await readFile(filePath, 'utf8');
  } catch (error) {
    throw new Error(
      `Failed to read session index entry ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
  return parseIndexEntry(parseJson(raw, filePath), filePath);
}

/**
 * Combine the legacy shared index with atomic per-session entries. Per-session
 * files win, so concurrent agents never rewrite unrelated sessions; tombstones
 * keep deletions from being resurrected by a legacy index entry.
 */
export async function readIndex(): Promise<SessionIndex> {
  const index = await readLegacyIndex();
  let files: string[];
  try {
    files = await readdir(entriesDirectory());
  } catch (error) {
    if (isErrnoException(error, 'ENOENT')) return index;
    throw new Error(
      `Failed to read session index directory ${entriesDirectory()}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }

  for (const file of files) {
    if (!file.endsWith('.json')) continue;
    const sessionId = basename(file, '.json');
    validateSessionId(sessionId);
    const entry = await readStoredIndexEntry(join(entriesDirectory(), file));
    if ('deleted' in entry) {
      delete index[sessionId];
    } else {
      index[sessionId] = entry;
    }
  }
  return index;
}

export async function rememberSession(session: StoredSession): Promise<void> {
  validateSessionId(session.sessionId);
  const entry: SessionIndexEntry = {
    cwd: session.cwd,
    updatedAt: session.updatedAt,
    title: session.title,
  };
  await writeFileAtomic(entryPath(session.sessionId), `${JSON.stringify(entry, null, 2)}\n`);
}

export async function forgetSession(sessionId: string): Promise<void> {
  await writeFileAtomic(entryPath(sessionId), `${JSON.stringify({ deleted: true }, null, 2)}\n`);
}

export async function findSessionCwd(sessionId: string): Promise<string | undefined> {
  const filePath = entryPath(sessionId);
  try {
    const entry = await readStoredIndexEntry(filePath);
    return 'deleted' in entry ? undefined : entry.cwd;
  } catch (error) {
    if (!(error instanceof Error) || !isErrnoException(error.cause, 'ENOENT')) {
      throw error;
    }
  }
  return (await readLegacyIndex())[sessionId]?.cwd;
}

export async function listStoredSessions(cwd?: string): Promise<SessionInfo[]> {
  const index = await readIndex();

  if (cwd) {
    const sessions: SessionInfo[] = [];
    const indexed = new Set<string>();
    for (const [sessionId, entry] of Object.entries(index)) {
      if (entry.cwd !== cwd) continue;
      indexed.add(sessionId);
      sessions.push({
        sessionId,
        cwd,
        title: entry.title ?? null,
        updatedAt: entry.updatedAt,
      });
    }

    let entries: string[];
    try {
      entries = await readdir(sessionDirectory(cwd));
    } catch {
      entries = [];
    }
    for (const entry of entries) {
      if (indexed.has(entry)) continue;
      try {
        validateSessionId(entry);
        const parsed = await readStoredSession(cwd, entry);
        sessions.push({
          sessionId: parsed.sessionId,
          cwd: parsed.cwd,
          title: parsed.title,
          updatedAt: parsed.updatedAt,
        });
      } catch {
        // Not a valid per-session directory or the state file is malformed.
      }
    }

    sessions.sort(byUpdatedAtDesc);
    return sessions;
  }

  const sessions: SessionInfo[] = [];
  for (const [sessionId, entry] of Object.entries(index)) {
    try {
      const parsed = await readStoredSession(entry.cwd, sessionId);
      sessions.push({
        sessionId: parsed.sessionId,
        cwd: parsed.cwd,
        title: parsed.title,
        updatedAt: parsed.updatedAt,
      });
    } catch {
      // Ignore sessions whose files are missing or malformed.
    }
  }
  sessions.sort(byUpdatedAtDesc);
  return sessions;
}

function byUpdatedAtDesc(a: SessionInfo, b: SessionInfo): number {
  const at = a.updatedAt ? Date.parse(a.updatedAt) : 0;
  const bt = b.updatedAt ? Date.parse(b.updatedAt) : 0;
  return bt - at;
}
