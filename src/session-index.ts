import { randomBytes } from 'node:crypto';
import { mkdir, readFile, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import type { SessionInfo } from '@agentclientprotocol/sdk';
import { readStoredSession, sessionDirectory, type StoredSession } from './storage.js';

/**
 * Crash-safe file write: write to a unique temp file in the destination
 * directory, then atomically rename over the target. A crash mid-write can
 * at worst leave a stray .tmp file — never a truncated/corrupt target. This
 * matters for state.json, which holds the full session history including
 * base64 media and is rewritten multiple times per turn.
 */
interface SessionIndex {
  [sessionId: string]: {
    cwd: string;
    updatedAt: string;
    /** Session title, mirrored here so listings do not have to parse the (potentially multi-MB) state.json. */
    title?: string | null;
  };
}

async function ensureDirectory(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
}

export async function writeFileAtomic(filePath: string, contents: string): Promise<void> {
  await ensureDirectory(dirname(filePath));
  const tmp = `${filePath}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    await writeFile(tmp, contents, 'utf8');
    await rename(tmp, filePath);
  } catch (error) {
    // Best-effort cleanup so failed writes do not litter the directory.
    await unlink(tmp).catch(() => {});
    throw error;
  }
}
export function indexDirectory(): string {
  const dataHome = process.env.XDG_DATA_HOME ?? join(homedir(), '.local', 'share');
  return join(dataHome, 'zen-agent');
}

function indexPath(): string {
  return join(indexDirectory(), 'index.json');
}

export async function readIndex(): Promise<SessionIndex> {
  try {
    const raw = await readFile(indexPath(), 'utf8');
    return JSON.parse(raw) as SessionIndex;
  } catch {
    return {};
  }
}

async function writeIndex(index: SessionIndex): Promise<void> {
  await writeFileAtomic(indexPath(), `${JSON.stringify(index, null, 2)}\n`);
}

export async function rememberSession(session: StoredSession): Promise<void> {
  const index = await readIndex();
  index[session.sessionId] = {
    cwd: session.cwd,
    updatedAt: session.updatedAt,
    title: session.title,
  };
  await writeIndex(index);
}

export async function forgetSession(sessionId: string): Promise<void> {
  const index = await readIndex();
  if (sessionId in index) {
    delete index[sessionId];
    await writeIndex(index);
  }
}
export async function findSessionCwd(sessionId: string): Promise<string | undefined> {
  const index = await readIndex();
  return index[sessionId]?.cwd;
}

export async function listStoredSessions(cwd?: string): Promise<SessionInfo[]> {
  const index = await readIndex();

  if (cwd) {
    const sessions: SessionInfo[] = [];
    const indexed = new Set<string>();

    // Fast path: indexed sessions resolve from the small index file without
    // parsing every state.json (which can carry base64 media payloads).
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

    // Fallback: pick up on-disk sessions the index does not know about
    // (hand-copied .sessions directories, wiped indexes).
    let entries: string[];
    try {
      entries = await readdir(sessionDirectory(cwd));
    } catch {
      entries = [];
    }
    for (const entry of entries) {
      if (indexed.has(entry)) continue;
      let raw: string;
      try {
        raw = await readFile(join(sessionDirectory(cwd), entry, 'state.json'), 'utf8');
      } catch {
        // Not a per-session directory (client/, llm/, logs/, ...).
        continue;
      }
      try {
        const parsed = JSON.parse(raw) as StoredSession;
        sessions.push({
          sessionId: parsed.sessionId,
          cwd: parsed.cwd,
          title: parsed.title,
          updatedAt: parsed.updatedAt,
        });
      } catch {
        // Ignore malformed session files.
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
