import { randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ModelMessage } from "ai";
import type { SessionInfo, SessionUpdate } from "@agentclientprotocol/sdk";

export interface StoredSession {
  sessionId: string;
  cwd: string;
  createdAt: string;
  updatedAt: string;
  title: string | null;
  events: SessionUpdate[];
  llmMessages: ModelMessage[];
}

interface SessionIndex {
  [sessionId: string]: {
    cwd: string;
    updatedAt: string;
  };
}

export function sessionDirectory(cwd: string): string {
  return join(cwd, "sessions");
}

export function sessionPath(cwd: string, sessionId: string): string {
  return join(sessionDirectory(cwd), `${sessionId}.json`);
}

function generateSessionId(): string {
  return `sess_${randomBytes(12).toString("hex")}`;
}

async function ensureDirectory(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
}

function indexDirectory(): string {
  const dataHome = process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share");
  return join(dataHome, "zen-agent");
}

function indexPath(): string {
  return join(indexDirectory(), "index.json");
}

async function readIndex(): Promise<SessionIndex> {
  try {
    const raw = await readFile(indexPath(), "utf8");
    return JSON.parse(raw) as SessionIndex;
  } catch {
    return {};
  }
}

async function writeIndex(index: SessionIndex): Promise<void> {
  await ensureDirectory(indexDirectory());
  await writeFile(indexPath(), `${JSON.stringify(index, null, 2)}\n`, "utf8");
}

async function rememberSession(session: StoredSession): Promise<void> {
  const index = await readIndex();
  index[session.sessionId] = {
    cwd: session.cwd,
    updatedAt: session.updatedAt,
  };
  await writeIndex(index);
}

async function forgetSession(sessionId: string): Promise<void> {
  const index = await readIndex();
  if (sessionId in index) {
    delete index[sessionId];
    await writeIndex(index);
  }
}

export async function createStoredSession(cwd: string): Promise<StoredSession> {
  await ensureDirectory(sessionDirectory(cwd));
  const now = new Date().toISOString();
  const session: StoredSession = {
    sessionId: generateSessionId(),
    cwd,
    createdAt: now,
    updatedAt: now,
    title: null,
    events: [],
    llmMessages: [],
  };
  await writeSession(session);
  await rememberSession(session);
  return session;
}

export async function writeSession(session: StoredSession): Promise<void> {
  await ensureDirectory(sessionDirectory(session.cwd));
  await writeFile(
    sessionPath(session.cwd, session.sessionId),
    `${JSON.stringify(session, null, 2)}\n`,
    "utf8",
  );
  await rememberSession(session);
}

export async function readStoredSession(
  cwd: string,
  sessionId: string,
): Promise<StoredSession> {
  const raw = await readFile(sessionPath(cwd, sessionId), "utf8");
  const parsed = JSON.parse(raw) as StoredSession;
  if (parsed.sessionId !== sessionId) {
    throw new Error(`Session file ${sessionId} has an invalid sessionId`);
  }
  if (parsed.cwd !== cwd) {
    throw new Error(`Session ${sessionId} belongs to ${parsed.cwd}, not ${cwd}`);
  }
  return parsed;
}

export async function findSessionCwd(sessionId: string): Promise<string | undefined> {
  const index = await readIndex();
  return index[sessionId]?.cwd;
}

export async function listStoredSessions(cwd?: string): Promise<SessionInfo[]> {
  if (cwd) {
    let files: string[];
    try {
      files = await readdir(sessionDirectory(cwd));
    } catch {
      return [];
    }

    const sessions: SessionInfo[] = [];
    for (const file of files) {
      if (!file.endsWith(".json")) continue;
      try {
        const raw = await readFile(join(sessionDirectory(cwd), file), "utf8");
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

  const index = await readIndex();
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

export async function deleteStoredSession(
  cwd: string,
  sessionId: string,
): Promise<void> {
  await rm(sessionPath(cwd, sessionId), { force: true });
  await forgetSession(sessionId);
}
