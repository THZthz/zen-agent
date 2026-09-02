import { randomBytes } from 'node:crypto';
import type { SessionInfo, SessionUpdate } from '@agentclientprotocol/sdk';
import type { CacheDiagnosticEntry } from '../providers/cache-diagnostics.js';
import type { TurnStats } from '../agent/stats.js';
import { openDb } from './db.js';
import { getDefaultProviderId, getProviderDefinition } from '../providers/registry.js';

/**
 * Session persistence on a single shared SQLite database (see db.ts): one row
 * per session in `sessions` (JSON columns for the collections that used to be
 * state.json fields), append-only transcripts in `llm_log` / `runtime_log`,
 * and bash tool calls in `terminal_calls`. The public async API is unchanged
 * from the former per-project state.json files so callers keep working.
 */

/**
 * One part of a multi-part user message. Media parts carry base64 payloads
 * attached by the client (Zed paste / drag & drop / @-mention all arrive as
 * ACP image blocks) and are sent to multimodal models as OpenAI-compatible
 * `image_url` (data URI) / `input_audio` content parts.
 */
export type UserContentPart =
  | { type: 'text'; text: string }
  | {
      type: 'image';
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
      type: 'audio';
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
  role: 'user';
  content: string | UserContentPart[];
  name?: string;
}

export interface AssistantMessage {
  role: 'assistant';
  content: Array<
    | { type: 'text'; text: string }
    | {
        type: 'reasoning';
        text: string;
        /** Opaque provider replay metadata; persist and return unchanged. */
        reasoningSignature?: string;
      }
    | {
        type: 'tool-call';
        toolCallId: string;
        toolName: string;
        input: unknown;
      }
  >;
}

export interface ToolMessage {
  role: 'tool';
  content: Array<{
    type: 'tool-result';
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

/**
 * LLM provider backing a session. Any registered provider id is valid:
 * built-ins are `deepseek` and `openrouter`; users can add more via
 * ZEN_AGENT_PROVIDERS / ZEN_AGENT_PROVIDERS_FILE (see provider-registry.ts).
 */
export type ProviderId = string;

/**
 * Model identifier for the active provider, e.g. "deepseek-v4-flash" or
 * "anthropic/claude-sonnet-4" (OpenRouter uses its own model slugs).
 */
export type ModelId = string;

/**
 * Session thinking-effort level.
 *
 * `off` is the session-level "disabled" value and maps to whatever the active
 * model/provider offers as its closest no-reasoning mode (OpenRouter: `none`
 * when supported, the model's lowest supported effort on mandatory-reasoning
 * models, otherwise the omitted field; DeepSeek: omitted field). The
 * remaining values cover the full OpenRouter gateway ladder (`minimal` <
 * `low` < `medium` < `high` < `xhigh` < `max`); DeepSeek accepts only a
 * subset (`low`/`high`/`max`) and other values are clamped when sending (see
 * deepseek.ts).
 */
export type ThinkingEffort = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'max' | 'xhigh';

export const DEFAULT_THINKING_EFFORT: ThinkingEffort = 'off';

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
  /**
   * Paths mounted read-write inside the bash sandbox as `--bind <path> <path>`;
   * everything else (the whole rootfs) is read-only. Set at runtime with the
   * `/writable` slash command (per session) and persisted so a resumed session
   * keeps its list. The default is `/tmp` and `/var/tmp` so the agent always
   * has scratch space; an empty list means no writable paths at all.
   */
  writablePaths: string[];
  /**
   * Whether the session may use tools at all (bash + read_media). Toggled
   * at runtime with the `/tools` slash command and persisted so a session
   * resumed after a restart keeps its choice. When off, no tool schemas are
   * sent to the model and any tool call is refused with a failed result.
   */
  toolsEnabled: boolean;
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
  /**
   * Cumulative cost in the session's billing currency (CNY for DeepSeek,
   * USD for OpenRouter). Persisted as `costYuan` before OpenRouter existed;
   * loading maps the legacy key.
   */
  cost: number;
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
    cost: 0,
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

/**
 * Database rows (see db.ts for the schema):
 *
 *   sessions        one row per session - scalar columns (session_id, cwd,
 *                   created_at, updated_at, title) plus JSON columns (config,
 *                   usage, events, llm_messages, turn_stats,
 *                   cache_diagnostics)
 *   llm_log         LLM request/response transcript, one JSON entry per row
 *   runtime_log     per-process diagnostic log, keyed by startup_key
 *   terminal_calls  one row per bash tool call (command + full output)
 */

/**
 * Require a session id to be exactly one path-like token.
 *
 * Session ids are persisted and may later arrive back from an ACP client or a
 * hand-created session, so this intentionally accepts legacy safe ids such as
 * `sess_manual` rather than enforcing only the current generated format.
 */

export function validateSessionId(sessionId: string): string {
  if (
    typeof sessionId !== 'string' ||
    sessionId.length === 0 ||
    sessionId === '.' ||
    sessionId === '..' ||
    sessionId.includes('/') ||
    sessionId.includes('\\') ||
    sessionId.includes('\0')
  ) {
    throw new Error(`Invalid session ID: ${JSON.stringify(sessionId)}`);
  }
  return sessionId;
}

function generateSessionId(): string {
  return validateSessionId(`sess_${randomBytes(12).toString('hex')}`);
}

export async function createStoredSession(
  cwd: string,
  provider: ProviderId = getDefaultProviderId(),
): Promise<StoredSession> {
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
      model: getProviderDefinition(provider)?.defaultModel ?? '',
      thinkingEffort: DEFAULT_THINKING_EFFORT,
      systemPrompt: '',
      sandbox: false,
      writablePaths: ['/tmp', '/var/tmp'],
      toolsEnabled: true,
    },
    usage: emptySessionUsage(),
    turnStats: [],
    cacheDiagnostics: [],
  };
  await writeSession(session);
  return session;
}

/** Insert or replace the session row. JSON columns mirror the former
 * state.json fields; normalization on load keeps old shapes readable. */
export async function writeSession(session: StoredSession): Promise<void> {
  openDb()
    .prepare(
      `INSERT INTO sessions
         (session_id, cwd, created_at, updated_at, title, config, usage,
          events, llm_messages, turn_stats, cache_diagnostics)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(session_id) DO UPDATE SET
         cwd = excluded.cwd, created_at = excluded.created_at,
         updated_at = excluded.updated_at, title = excluded.title,
         config = excluded.config, usage = excluded.usage,
         events = excluded.events, llm_messages = excluded.llm_messages,
         turn_stats = excluded.turn_stats,
         cache_diagnostics = excluded.cache_diagnostics`,
    )
    .run(
      session.sessionId,
      session.cwd,
      session.createdAt,
      session.updatedAt,
      session.title,
      JSON.stringify(session.config),
      JSON.stringify(session.usage),
      JSON.stringify(session.events),
      JSON.stringify(session.llmMessages),
      JSON.stringify(session.turnStats),
      JSON.stringify(session.cacheDiagnostics ?? []),
    );
}

function defaultConfig(provider: ProviderId = getDefaultProviderId()): SessionConfig {
  return {
    provider,
    model: getProviderDefinition(provider)?.defaultModel ?? '',
    thinkingEffort: DEFAULT_THINKING_EFFORT,
    systemPrompt: '',
    sandbox: false,
    writablePaths: ['/tmp', '/var/tmp'],
    toolsEnabled: true,
  };
}

const THINKING_EFFORTS: readonly ThinkingEffort[] = [
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
];

/** What a pure element-level normalizer returns: the kept items plus how many
 * structurally unusable elements were dropped. */
export interface NormalizedItems<T> {
  items: T[];
  dropped: number;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Element-level pass over the persisted `events` array: an event survives
 * only when it is an object with a non-empty `sessionUpdate` discriminator —
 * the one field every downstream consumer dereferences (replay, ACP
 * dispatch). Everything else inside a kept event is passed through untouched:
 * validation stays syntactic, and unknown (e.g. newer) update kinds keep
 * loading so a session written by a newer agent version is not destroyed.
 */
export function normalizeEvents(raw: unknown): NormalizedItems<SessionUpdate> {
  if (!Array.isArray(raw)) return { items: [], dropped: 0 };
  const items: SessionUpdate[] = [];
  let dropped = 0;
  for (const element of raw) {
    const kind = isPlainObject(element) ? element.sessionUpdate : undefined;
    if (typeof kind === 'string' && kind.length > 0) {
      items.push(element as SessionUpdate);
    } else {
      dropped += 1;
    }
  }
  return { items, dropped };
}

function isValidUserPart(part: unknown): boolean {
  if (!isPlainObject(part)) return false;
  if (part.type === 'text') return typeof part.text === 'string';
  if (part.type === 'image' || part.type === 'audio') {
    return typeof part.mimeType === 'string' && typeof part.data === 'string';
  }
  return false;
}

function isValidAssistantPart(part: unknown): boolean {
  if (!isPlainObject(part)) return false;
  if (part.type === 'text' || part.type === 'reasoning') {
    return typeof part.text === 'string';
  }
  if (part.type === 'tool-call') return typeof part.toolCallId === 'string';
  return false;
}

/** Tool messages are only useful as complete tool-result arrays; a single
 * malformed part makes the whole message unusable for the LLM request. */
function isValidToolContent(content: unknown): boolean {
  return (
    Array.isArray(content) &&
    content.every(
      (part) =>
        isPlainObject(part) && part.type === 'tool-result' && typeof part.toolCallId === 'string',
    )
  );
}

/** Keep a user message when it carries usable content (string, or a part
 * array with at least one valid part, invalid parts filtered). Valid messages
 * are returned unchanged (same reference) so healthy sessions round-trip
 * byte-identically; only messages that need repair are rebuilt. A non-string
 * `name` is dropped. */
function normalizeUserMessage(element: Record<string, unknown>): LlmMessage | undefined {
  const content = element.content;
  let keptContent: string | unknown[];
  let changed = false;
  if (typeof content === 'string') {
    keptContent = content;
  } else if (Array.isArray(content)) {
    const valid = content.filter(isValidUserPart);
    // Nothing survived and there is no usable content left.
    if (valid.length === 0) return undefined;
    if (valid.length !== content.length) {
      keptContent = valid;
      changed = true;
    } else {
      keptContent = content;
    }
  } else {
    return undefined;
  }
  if (element.name !== undefined && typeof element.name !== 'string') {
    changed = true;
  }
  if (!changed) return element as unknown as LlmMessage;
  const cleaned: Record<string, unknown> = { ...element, content: keptContent };
  if (cleaned.name !== undefined && typeof cleaned.name !== 'string') {
    delete cleaned.name;
  }
  return cleaned as unknown as LlmMessage;
}

/**
 * Element-level pass over the persisted `llmMessages` array. Messages whose
 * role is missing/unknown, whose content is structurally unusable, or whose
 * parts are all invalid are dropped; assistant messages with a mix of valid
 * and invalid parts keep the valid parts. Dropped assistant tool-calls can
 * orphan tool messages — tolerated at request time by healMessages.
 */
export function normalizeLlmMessages(raw: unknown): NormalizedItems<LlmMessage> {
  if (!Array.isArray(raw)) return { items: [], dropped: 0 };
  const items: LlmMessage[] = [];
  let dropped = 0;
  for (const element of raw) {
    if (!isPlainObject(element)) {
      dropped += 1;
      continue;
    }
    if (element.role === 'user') {
      const kept = normalizeUserMessage(element);
      if (kept === undefined) dropped += 1;
      else items.push(kept);
      continue;
    }
    if (element.role === 'assistant') {
      const content = element.content;
      if (Array.isArray(content)) {
        const valid = content.filter(isValidAssistantPart);
        if (valid.length === 0) {
          dropped += 1;
        } else if (valid.length === content.length) {
          items.push(element as unknown as LlmMessage);
        } else {
          items.push({ ...element, content: valid } as LlmMessage);
        }
      } else {
        dropped += 1;
      }
      continue;
    }
    if (element.role === 'tool') {
      if (isValidToolContent(element.content)) {
        items.push(element as unknown as LlmMessage);
      } else {
        dropped += 1;
      }
      continue;
    }
    dropped += 1;
  }
  return { items, dropped };
}

/** Shared element pass for `turnStats` / `cacheDiagnostics`: drop anything
 * that is not a plain object. Field-level shape stays the owner's concern. */
function normalizeObjectEntries<T>(raw: unknown): NormalizedItems<T> {
  if (!Array.isArray(raw)) return { items: [], dropped: 0 };
  const items: T[] = [];
  let dropped = 0;
  for (const element of raw) {
    if (isPlainObject(element)) items.push(element as T);
    else dropped += 1;
  }
  return { items, dropped };
}

/**
 * Backfill fields missing from older sessions or partially damaged rows, so
 * one bad field degrades to its default instead of a TypeError deep inside
 * the agent. Identity (session id / cwd) is checked by the caller against the
 * row columns before this runs.
 *
 * Collection fields additionally get an element-level pass (see
 * normalizeEvents / normalizeLlmMessages): structurally unusable elements are
 * dropped instead of failing the whole load — a failed load blocks resume
 * entirely, which is worse than losing one malformed entry. The dropped
 * count is returned so the caller can surface it in the runtime log.
 */

function normalizeStoredSession(parsed: { [key: string]: unknown }): {
  session: StoredSession;
  droppedEntries: number;
} {
  const raw = parsed;

  // --- config ---
  const rawConfig =
    typeof raw.config === 'object' && raw.config !== null
      ? (raw.config as Record<string, unknown>)
      : {};
  // Sessions created before providers existed have no `provider`; they are
  // DeepSeek sessions by definition (the only provider back then).
  const provider: ProviderId =
    typeof rawConfig.provider === 'string' && rawConfig.provider.length > 0
      ? rawConfig.provider
      : getDefaultProviderId();
  const config: SessionConfig = {
    provider,
    model:
      typeof rawConfig.model === 'string' && rawConfig.model.length > 0
        ? rawConfig.model
        : defaultConfig(provider).model,
    thinkingEffort: THINKING_EFFORTS.includes(rawConfig.thinkingEffort as ThinkingEffort)
      ? (rawConfig.thinkingEffort as ThinkingEffort)
      : DEFAULT_THINKING_EFFORT,
    systemPrompt: typeof rawConfig.systemPrompt === 'string' ? rawConfig.systemPrompt : '',
    // Older sessions predate the flag; absent means off.
    sandbox: rawConfig.sandbox === true,
    // The sandbox is deny-by-default (root read-only): writable paths are an
    // explicit allowlist. Absent or a legacy `roBindPaths` list (the old
    // `/robind` semantics, gone) fall back to the default scratch paths.
    writablePaths: Array.isArray(rawConfig.writablePaths)
      ? (rawConfig.writablePaths as unknown[]).filter(
          (p): p is string => typeof p === 'string' && p.trim() !== '',
        )
      : ['/tmp', '/var/tmp'],
    // Older sessions predate the flag; absent means enabled (the default).
    toolsEnabled: rawConfig.toolsEnabled !== false,
  };

  // --- usage: keep every known numeric field, default anything else ---
  const emptyUsage = emptySessionUsage();
  const usage: SessionUsage = { ...emptyUsage };
  if (typeof raw.usage === 'object' && raw.usage !== null) {
    for (const key of Object.keys(emptyUsage) as Array<keyof SessionUsage>) {
      if (key === 'estimated') continue;
      const value = (raw.usage as Record<string, unknown>)[key];
      if (typeof value === 'number' && Number.isFinite(value)) {
        usage[key] = value as never;
      }
    }
    // Legacy persisted shape named the cost field `costYuan` (from the
    // CNY-only era); map it so old sessions keep their accumulated totals.
    const legacyCost = (raw.usage as Record<string, unknown>).costYuan;
    if (
      typeof legacyCost === 'number' &&
      Number.isFinite(legacyCost) &&
      usage.cost === emptyUsage.cost
    ) {
      usage.cost = legacyCost;
    }
  }

  const nowIso = new Date().toISOString();
  const events = normalizeEvents(raw.events);
  const llmMessages = normalizeLlmMessages(raw.llmMessages);
  const turnStats = normalizeObjectEntries<TurnStats>(raw.turnStats);
  const cacheDiagnostics = normalizeObjectEntries<CacheDiagnosticEntry>(raw.cacheDiagnostics);
  return {
    session: {
      sessionId: String(raw.sessionId),
      cwd: String(raw.cwd),
      createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : nowIso,
      updatedAt: typeof raw.updatedAt === 'string' ? raw.updatedAt : nowIso,
      title: typeof raw.title === 'string' ? raw.title : null,
      events: events.items,
      llmMessages: llmMessages.items,
      config,
      usage,
      turnStats: turnStats.items,
      cacheDiagnostics: cacheDiagnostics.items,
    },
    droppedEntries:
      events.dropped + llmMessages.dropped + turnStats.dropped + cacheDiagnostics.dropped,
  };
}

interface SessionRow {
  session_id: string;
  cwd: string;
  created_at: string;
  updated_at: string;
  title: string | null;
  config: string;
  usage: string;
  events: string;
  llm_messages: string;
  turn_stats: string;
  cache_diagnostics: string;
}

/** JSON.parse a session row column; a corrupt column fails the load with the
 * same "corrupted" error surface the former state.json files had. */
function parseColumn(raw: string, sessionId: string, column: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch (error) {
    throw new Error(
      `Session record for ${sessionId} is corrupted: ${column} is not valid JSON (${error instanceof Error ? error.message : String(error)})`,
    );
  }
}

function selectSessionRow(sessionId: string): SessionRow | undefined {
  return openDb().prepare('SELECT * FROM sessions WHERE session_id = ?').get(sessionId) as
    SessionRow | undefined;
}

/** Load and validate a session; `droppedEntries` counts persisted elements
 * (events, messages, stats, diagnostics) that were dropped as structurally
 * unusable, so the caller can make the data loss visible in the runtime log. */
export async function readStoredSession(
  cwd: string,
  sessionId: string,
): Promise<{ session: StoredSession; droppedEntries: number }> {
  // Keep validation outside the query so an unsafe id is reported as invalid
  // rather than being disguised as a missing row.
  validateSessionId(sessionId);
  const row = selectSessionRow(sessionId);
  if (!row) {
    throw new Error(`Session not found for ${sessionId}`);
  }
  if (row.cwd !== cwd) {
    throw new Error(`Session ${sessionId} belongs to ${row.cwd}, not ${cwd}`);
  }
  const parsed: Record<string, unknown> = {
    sessionId: row.session_id,
    cwd: row.cwd,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    title: row.title,
    config: parseColumn(row.config, sessionId, 'config'),
    usage: parseColumn(row.usage, sessionId, 'usage'),
    events: parseColumn(row.events, sessionId, 'events'),
    llmMessages: parseColumn(row.llm_messages, sessionId, 'llm_messages'),
    turnStats: parseColumn(row.turn_stats, sessionId, 'turn_stats'),
    cacheDiagnostics: parseColumn(row.cache_diagnostics, sessionId, 'cache_diagnostics'),
  };
  return normalizeStoredSession(parsed);
}

/** Remove the session row and everything that belongs to it: the terminal
 * records and the LLM transcript entries a live session accumulates. */
export async function deleteStoredSession(sessionId: string): Promise<void> {
  validateSessionId(sessionId);
  const db = openDb();
  db.prepare('DELETE FROM sessions WHERE session_id = ?').run(sessionId);
  db.prepare('DELETE FROM terminal_calls WHERE session_id = ?').run(sessionId);
  db.prepare('DELETE FROM llm_log WHERE session_id = ?').run(sessionId);
}

/** The project a session belongs to (the former global index's job). */
export async function findSessionCwd(sessionId: string): Promise<string | undefined> {
  validateSessionId(sessionId);
  const row = openDb().prepare('SELECT cwd FROM sessions WHERE session_id = ?').get(sessionId) as
    { cwd: string } | undefined;
  return row?.cwd;
}

function byUpdatedAtDesc(a: SessionInfo, b: SessionInfo): number {
  const at = a.updatedAt ? Date.parse(a.updatedAt) : 0;
  const bt = b.updatedAt ? Date.parse(b.updatedAt) : 0;
  return bt - at;
}

/** Sessions for one project (or all projects when no cwd is given), newest
 * first. The sessions table itself is the index; no separate listing exists. */
export async function listStoredSessions(cwd?: string): Promise<SessionInfo[]> {
  const sql =
    'SELECT session_id, cwd, title, updated_at FROM sessions' +
    (cwd === undefined ? '' : ' WHERE cwd = ?');
  const statement = openDb().prepare(sql);
  const rows = (cwd === undefined ? statement.all() : statement.all(cwd)) as Array<{
    session_id: string;
    cwd: string;
    title: string | null;
    updated_at: string;
  }>;
  return rows
    .map((row) => ({
      sessionId: row.session_id,
      cwd: row.cwd,
      title: row.title,
      updatedAt: row.updated_at,
    }))
    .sort(byUpdatedAtDesc);
}
