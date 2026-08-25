import { randomBytes } from "node:crypto";
import { mkdir, readFile, readdir, rename, rm, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { SessionInfo, SessionUpdate } from "@agentclientprotocol/sdk";
import type { CacheDiagnosticEntry } from "./cache-diagnostics.js";
import type { TurnStats } from "./turn-stats.js";

/**
 * One part of a multi-part user message. Media parts carry base64 payloads
 * attached by the client (Zed paste / drag & drop / @-mention all arrive as
 * ACP image blocks) and are sent to multimodal models as OpenAI-compatible
 * `image_url` (data URI) / `input_audio` content parts.
 */
export type UserContentPart =
  | { type: "text"; text: string }
  | {
      type: "image";
      /** MIME type, e.g. "image/png". */
      mimeType: string;
      /** Base64-encoded payload. */
      data: string;
      /**
       * Source URI when known (`file://...` for @-mentioned files). Display
       * and logging only - never sent to the LLM.
       */
      uri?: string;
    }
  | {
      type: "audio";
      /** MIME type, e.g. "audio/wav". */
      mimeType: string;
      /** Base64-encoded payload. */
      data: string;
    };

/**
 * A user message that may carry a `name` (the OpenAI wire `name` field,
 * e.g. the git user name for the human's prompts, or "environment" for the
 * auto-generated environment message) and either plain text or multi-part
 * content (text + attached media).
 */
export interface NamedUserMessage {
  role: "user";
  content: string | UserContentPart[];
  name?: string;
}

export interface AssistantMessage {
  role: "assistant";
  content: Array<
    | { type: "text"; text: string }
    | { type: "reasoning"; text: string }
    | {
        type: "tool-call";
        toolCallId: string;
        toolName: string;
        input: unknown;
      }
  >;
}

export interface ToolMessage {
  role: "tool";
  content: Array<{
    type: "tool-result";
    toolCallId: string;
    toolName?: string;
    output: unknown;
  }>;
}

/**
 * Every message we can persist in a session and send to the LLM.
 *
 * These are our own structural types (the AI SDK was dropped - see SPEC section 6.1); they intentionally match the shapes historically persisted in
 * state.json so old sessions keep loading.
 */
export type LlmMessage = NamedUserMessage | AssistantMessage | ToolMessage;

export type ProviderId = "deepseek" | "openrouter";

/**
 * Model identifier for the active provider, e.g. "deepseek-v4-flash" or
 * "anthropic/claude-sonnet-4" (OpenRouter uses its own model slugs).
 */
export type ModelId = string;
export type ThinkingEffort = "off" | "high" | "max";

export const DEFAULT_DEEPSEEK_MODEL: ModelId = "deepseek-v4-flash";
export const DEFAULT_OPENROUTER_MODEL: ModelId = "openrouter/free";
export const DEFAULT_PROVIDER: ProviderId = "deepseek";
export const DEFAULT_THINKING_EFFORT: ThinkingEffort = "off";

export interface SessionConfig {
  /**
   * LLM provider backing this session (persisted so cost and currency stay
   * consistent across restarts). Chosen per session via the `provider`
   * config option; sessions created before providers existed default to
   * "deepseek" on load.
   */
  provider: ProviderId;
  model: ModelId;
  thinkingEffort: ThinkingEffort;
  systemPrompt: string;
  /**
   * Wrap every bash tool call in its own bubblewrap invocation. Set at
   * runtime with the `/sandbox` slash command (per session); the
   * `ZEN_AGENT_SANDBOX=1` environment variable is the global equivalent and
   * applies even when this flag is off (see tool-execution.ts).
   */
  sandbox: boolean;
}

/**
 * Cumulative usage/cost statistics for a session. Used to drive the ACP
 * `usage_update` notification (context window + cost in CNY) and the
 * per-turn stats line shown to the user.
 */
export interface SessionUsage {
  /** Number of LLM-backed prompt turns completed. */
  turns: number;
  /** Total LLM/tool steps across all turns. */
  steps: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheMissTokens: number;
  reasoningTokens: number;
  /** Cumulative cost in CNY. */
  costYuan: number;
  /** Cumulative LLM request wall time in ms. */
  llmMs: number;
  /** Cumulative thinking (reasoning) time in ms. */
  thinkingMs: number;
  /** Cumulative answer-streaming time in ms. */
  answeringMs: number;
  /** Cumulative bash tool execution time in ms. */
  toolMs: number;
  /**
   * True when this usage rollup is an estimate rather than exact API usage.
   * Not currently produced; kept for forward compatibility of the persisted
   * shape.
   */
  estimated?: boolean;
}

export function emptySessionUsage(): SessionUsage {
  return {
    turns: 0,
    steps: 0,
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheMissTokens: 0,
    reasoningTokens: 0,
    costYuan: 0,
    llmMs: 0,
    thinkingMs: 0,
    answeringMs: 0,
    toolMs: 0,
  };
}

/**
 * Persistent session state.
 *
 * Zed loads a session via `session/load` and replays our stored `events`
 * (see ZenAgent.prepareReplayEvents) to rebuild the thread; `llmMessages` is
 * what gets sent to the LLM on the next prompt. `usage` accumulates across
 * turns so the ACP `usage_update` cost (CNY) is cumulative, matching what
 * Zed's SessionCost expects ("Total cumulative cost for session").
 */
export interface StoredSession {
  sessionId: string;
  cwd: string;
  createdAt: string;
  updatedAt: string;
  title: string | null;
  events: SessionUpdate[];
  llmMessages: LlmMessage[];
  config: SessionConfig;
  usage: SessionUsage;
  /** Per-turn stats, one entry per completed LLM-backed turn. */
  turnStats: TurnStats[];
  /** Per-step cache diagnostics, newest last (ring-buffered at 50 entries). */
  cacheDiagnostics?: CacheDiagnosticEntry[];
}

interface SessionIndex {
  [sessionId: string]: {
    cwd: string;
    updatedAt: string;
  };
}

/**
 * Session storage layout (one directory per session):
 *
 *   <project>/.sessions/<sessionId>/state.json
 *   <project>/.sessions/<sessionId>/llm.jsonl
 *   <project>/.sessions/<sessionId>/terminals/input-<timestamp>-<callId>.sh
 *   <project>/.sessions/<sessionId>/terminals/output-<timestamp>-<callId>.log
 *   <project>/.sessions/client/<startupTimestamp>_<uuid>/log.jsonl
 */
export function sessionDirectory(cwd: string): string {
  return join(cwd, ".sessions");
}

/** Per-session root: <project>/.sessions/<sessionId>/ */
export function sessionRootDirectory(cwd: string, sessionId: string): string {
  return join(sessionDirectory(cwd), sessionId);
}

/** Terminal artifacts for a session: <project>/.sessions/<sessionId>/terminals/ */
export function terminalDirectory(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), "terminals");
}

/** Session state: <project>/.sessions/<sessionId>/state.json */
export function sessionPath(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), "state.json");
}

/** LLM request/response transcript: <project>/.sessions/<sessionId>/llm.jsonl */
export function sessionLlmLogPath(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), "llm.jsonl");
}

/**
 * Zen Agent's own per-startup debug log:
 * <project>/.sessions/client/<startupTimestamp>_<uuid>/log.jsonl
 * `startupKey` is created once per agent process (local-time timestamp
 * "YYYY-MM-DD-HH-mm-ss" plus UUID) so every run of the agent gets its own
 * log directory, e.g. 2026-08-21-23-06-04_<uuid>.
 */
export function clientLogPath(cwd: string, startupKey: string): string {
  return join(sessionDirectory(cwd), "client", startupKey, "log.jsonl");
}

/**
 * Cached OpenRouter model catalog shared across agent restarts:
 * <project>/.sessions/client/models.openrouter.json
 */
export function clientModelsPath(cwd: string): string {
  return join(sessionDirectory(cwd), "client", "models.openrouter.json");
}

function generateSessionId(): string {
  return `sess_${randomBytes(12).toString("hex")}`;
}

async function ensureDirectory(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
}

/**
 * Crash-safe file write: write to a unique temp file in the destination
 * directory, then atomically rename over the target. A crash mid-write can
 * at worst leave a stray .tmp file — never a truncated/corrupt target. This
 * matters for state.json, which holds the full session history including
 * base64 media and is rewritten multiple times per turn.
 */
async function writeFileAtomic(filePath: string, contents: string): Promise<void> {
  await ensureDirectory(dirname(filePath));
  const tmp = `${filePath}.${process.pid}.${randomBytes(4).toString("hex")}.tmp`;
  try {
    await writeFile(tmp, contents, "utf8");
    await rename(tmp, filePath);
  } catch (error) {
    // Best-effort cleanup so failed writes do not litter the directory.
    await unlink(tmp).catch(() => {});
    throw error;
  }
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
  await writeFileAtomic(indexPath(), `${JSON.stringify(index, null, 2)}\n`);
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

export async function createStoredSession(
  cwd: string,
  provider: ProviderId = DEFAULT_PROVIDER,
): Promise<StoredSession> {
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
    config: {
      provider,
      model: provider === "openrouter" ? DEFAULT_OPENROUTER_MODEL : DEFAULT_DEEPSEEK_MODEL,
      thinkingEffort: DEFAULT_THINKING_EFFORT,
      systemPrompt: "",
      sandbox: false,
    },
    usage: emptySessionUsage(),
    turnStats: [],
    cacheDiagnostics: [],
  };
  await writeSession(session);
  await rememberSession(session);
  return session;
}

export async function writeSession(session: StoredSession): Promise<void> {
  await writeFileAtomic(
    sessionPath(session.cwd, session.sessionId),
    `${JSON.stringify(session, null, 2)}\n`,
  );
  await rememberSession(session);
}

function defaultConfig(provider: ProviderId = DEFAULT_PROVIDER): SessionConfig {
  return {
    provider,
    model: provider === "openrouter" ? DEFAULT_OPENROUTER_MODEL : DEFAULT_DEEPSEEK_MODEL,
    thinkingEffort: DEFAULT_THINKING_EFFORT,
    systemPrompt: "",
    sandbox: false,
  };
}

const THINKING_EFFORTS: readonly ThinkingEffort[] = ["off", "high", "max"];

/**
 * Validate a parsed state.json and backfill fields missing from older
 * sessions or partially damaged files, so one bad field degrades to its
 * default instead of a TypeError deep inside the agent. Unrecoverable shapes
 * (wrong session id / cwd / not an object) throw a clean, actionable error.
 */
function normalizeStoredSession(
  parsed: unknown,
  cwd: string,
  sessionId: string,
): StoredSession {
  const corrupt = (detail: string): Error =>
    new Error(`Session file for ${sessionId} is corrupted: ${detail}`);

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw corrupt("not a session object");
  }
  const raw = parsed as Record<string, unknown>;

  if (raw.sessionId !== sessionId) {
    throw new Error(`Session file ${sessionId} has an invalid sessionId`);
  }
  if (raw.cwd !== cwd) {
    throw new Error(`Session ${sessionId} belongs to ${String(raw.cwd)}, not ${cwd}`);
  }

  // --- config ---
  const rawConfig =
    typeof raw.config === "object" && raw.config !== null
      ? (raw.config as Record<string, unknown>)
      : {};
  // Sessions created before providers existed have no `provider`; they are
  // DeepSeek sessions by definition (the only provider back then).
  const provider: ProviderId =
    rawConfig.provider === "deepseek" || rawConfig.provider === "openrouter"
      ? rawConfig.provider
      : DEFAULT_PROVIDER;
  const config: SessionConfig = {
    provider,
    model:
      typeof rawConfig.model === "string" && rawConfig.model.length > 0
        ? rawConfig.model
        : defaultConfig(provider).model,
    thinkingEffort: THINKING_EFFORTS.includes(rawConfig.thinkingEffort as ThinkingEffort)
      ? (rawConfig.thinkingEffort as ThinkingEffort)
      : DEFAULT_THINKING_EFFORT,
    systemPrompt: typeof rawConfig.systemPrompt === "string" ? rawConfig.systemPrompt : "",
    // Older sessions predate the flag; absent means off.
    sandbox: rawConfig.sandbox === true,
  };

  // --- usage: keep every known numeric field, default anything else ---
  const emptyUsage = emptySessionUsage();
  const usage: SessionUsage = { ...emptyUsage };
  if (typeof raw.usage === "object" && raw.usage !== null) {
    for (const key of Object.keys(emptyUsage) as Array<keyof SessionUsage>) {
      if (key === "estimated") continue;
      const value = (raw.usage as Record<string, unknown>)[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        usage[key] = value as never;
      }
    }
  }

  const nowIso = new Date().toISOString();
  return {
    sessionId,
    cwd,
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : nowIso,
    updatedAt: typeof raw.updatedAt === "string" ? raw.updatedAt : nowIso,
    title: typeof raw.title === "string" ? raw.title : null,
    events: Array.isArray(raw.events) ? (raw.events as StoredSession["events"]) : [],
    llmMessages: Array.isArray(raw.llmMessages)
      ? (raw.llmMessages as StoredSession["llmMessages"])
      : [],
    config,
    usage,
    turnStats: Array.isArray(raw.turnStats) ? (raw.turnStats as StoredSession["turnStats"]) : [],
    cacheDiagnostics: Array.isArray(raw.cacheDiagnostics)
      ? (raw.cacheDiagnostics as StoredSession["cacheDiagnostics"])
      : [],
  };
}

export async function readStoredSession(
  cwd: string,
  sessionId: string,
): Promise<StoredSession> {
  let raw: string;
  try {
    raw = await readFile(sessionPath(cwd, sessionId), "utf8");
  } catch {
    throw new Error(`Session file not found for ${sessionId}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `Session file for ${sessionId} is corrupted: not valid JSON (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  return normalizeStoredSession(parsed, cwd, sessionId);
}

export async function findSessionCwd(sessionId: string): Promise<string | undefined> {
  const index = await readIndex();
  return index[sessionId]?.cwd;
}

export async function listStoredSessions(cwd?: string): Promise<SessionInfo[]> {
  if (cwd) {
    const seen = new Set<string>();
    const sessions: SessionInfo[] = [];

    // Current layout: <project>/.sessions/<sessionId>/state.json
    let entries: string[];
    try {
      entries = await readdir(sessionDirectory(cwd));
    } catch {
      entries = [];
    }
    for (const entry of entries) {
      if (seen.has(entry)) continue;
      let raw: string;
      try {
        raw = await readFile(
          join(sessionDirectory(cwd), entry, "state.json"),
          "utf8",
        );
      } catch {
        // Not a per-session directory (client/, llm/, logs/, ...).
        continue;
      }
      seen.add(entry);
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
  // Remove the whole per-session tree: state.json plus the terminal
  // artifacts (input-*.sh / output-*.log) and llm.jsonl that would otherwise
  // be orphaned forever (the index forgets the cwd right after).
  await rm(sessionRootDirectory(cwd, sessionId), { recursive: true, force: true });
  await forgetSession(sessionId);
}
