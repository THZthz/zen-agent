import { randomBytes } from 'node:crypto';
import { mkdir, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import type { SessionUpdate } from '@agentclientprotocol/sdk';
import type { CacheDiagnosticEntry } from './cache-diagnostics.js';
import type { TurnStats } from './turn-stats.js';
import { forgetSession, rememberSession, writeFileAtomic } from './session-index.js';

export { findSessionCwd, listStoredSessions } from './session-index.js';

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

export type ProviderId = 'deepseek' | 'openrouter';

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

export const DEFAULT_DEEPSEEK_MODEL: ModelId = 'deepseek-v4-flash';
export const DEFAULT_OPENROUTER_MODEL: ModelId = 'openrouter/free';
export const DEFAULT_PROVIDER: ProviderId = 'deepseek';
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
 * Session storage layout (one directory per session):
 *
 *   <project>/.sessions/<sessionId>/state.json
 *   <project>/.sessions/<sessionId>/llm.jsonl
 *   <project>/.sessions/<sessionId>/terminals/input-<timestamp>-<callId>.sh
 *   <project>/.sessions/<sessionId>/terminals/output-<timestamp>-<callId>.log
 *   <project>/.sessions/client/<startupTimestamp>_<uuid>/log.jsonl
 */
export function sessionDirectory(cwd: string): string {
  return join(cwd, '.sessions');
}

/**
 * Require a session id to be exactly one filesystem path component.
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

/** Per-session root: <project>/.sessions/<sessionId>/ */
export function sessionRootDirectory(cwd: string, sessionId: string): string {
  validateSessionId(sessionId);
  return join(sessionDirectory(cwd), sessionId);
}

/** Terminal artifacts for a session: <project>/.sessions/<sessionId>/terminals/ */
export function terminalDirectory(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), 'terminals');
}

/** Session state: <project>/.sessions/<sessionId>/state.json */
export function sessionPath(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), 'state.json');
}

/** LLM request/response transcript: <project>/.sessions/<sessionId>/llm.jsonl */
export function sessionLlmLogPath(cwd: string, sessionId: string): string {
  return join(sessionRootDirectory(cwd, sessionId), 'llm.jsonl');
}

/**
 * Zen Agent's own per-startup debug log:
 * <project>/.sessions/client/<startupTimestamp>_<uuid>/log.jsonl
 * `startupKey` is created once per agent process (local-time timestamp
 * "YYYY-MM-DD-HH-mm-ss" plus UUID) so every run of the agent gets its own
 * log directory, e.g. 2026-08-21-23-06-04_<uuid>.
 */
export function clientLogPath(cwd: string, startupKey: string): string {
  return join(sessionDirectory(cwd), 'client', startupKey, 'log.jsonl');
}

/**
 * Cached OpenRouter model catalog shared across agent restarts:
 * <project>/.sessions/client/models.openrouter.json
 */
export function clientModelsPath(cwd: string): string {
  return join(sessionDirectory(cwd), 'client', 'models.openrouter.json');
}

function generateSessionId(): string {
  return validateSessionId(`sess_${randomBytes(12).toString('hex')}`);
}

async function ensureDirectory(dir: string): Promise<void> {
  await mkdir(dir, { recursive: true });
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
      model: provider === 'openrouter' ? DEFAULT_OPENROUTER_MODEL : DEFAULT_DEEPSEEK_MODEL,
      thinkingEffort: DEFAULT_THINKING_EFFORT,
      systemPrompt: '',
      sandbox: false,
      toolsEnabled: true,
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
    model: provider === 'openrouter' ? DEFAULT_OPENROUTER_MODEL : DEFAULT_DEEPSEEK_MODEL,
    thinkingEffort: DEFAULT_THINKING_EFFORT,
    systemPrompt: '',
    sandbox: false,
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

/**
 * Validate a parsed state.json and backfill fields missing from older
 * sessions or partially damaged files, so one bad field degrades to its
 * default instead of a TypeError deep inside the agent. Unrecoverable shapes
 * (wrong session id / cwd / not an object) throw a clean, actionable error.
 */
function normalizeStoredSession(parsed: unknown, cwd: string, sessionId: string): StoredSession {
  const corrupt = (detail: string): Error =>
    new Error(`Session file for ${sessionId} is corrupted: ${detail}`);

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw corrupt('not a session object');
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
    typeof raw.config === 'object' && raw.config !== null
      ? (raw.config as Record<string, unknown>)
      : {};
  // Sessions created before providers existed have no `provider`; they are
  // DeepSeek sessions by definition (the only provider back then).
  const provider: ProviderId =
    rawConfig.provider === 'deepseek' || rawConfig.provider === 'openrouter'
      ? rawConfig.provider
      : DEFAULT_PROVIDER;
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
  return {
    sessionId,
    cwd,
    createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : nowIso,
    updatedAt: typeof raw.updatedAt === 'string' ? raw.updatedAt : nowIso,
    title: typeof raw.title === 'string' ? raw.title : null,
    events: Array.isArray(raw.events) ? (raw.events as StoredSession['events']) : [],
    llmMessages: Array.isArray(raw.llmMessages)
      ? (raw.llmMessages as StoredSession['llmMessages'])
      : [],
    config,
    usage,
    turnStats: Array.isArray(raw.turnStats) ? (raw.turnStats as StoredSession['turnStats']) : [],
    cacheDiagnostics: Array.isArray(raw.cacheDiagnostics)
      ? (raw.cacheDiagnostics as StoredSession['cacheDiagnostics'])
      : [],
  };
}

export async function readStoredSession(cwd: string, sessionId: string): Promise<StoredSession> {
  // Keep validation outside the filesystem error handler so an unsafe id is
  // reported as invalid rather than being disguised as a missing file.
  validateSessionId(sessionId);
  let raw: string;
  try {
    raw = await readFile(sessionPath(cwd, sessionId), 'utf8');
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

export async function deleteStoredSession(cwd: string, sessionId: string): Promise<void> {
  // Remove the whole per-session tree: state.json plus the terminal
  // artifacts (input-*.sh / output-*.log) and llm.jsonl that would otherwise
  // be orphaned forever (the index forgets the cwd right after).
  await rm(sessionRootDirectory(cwd, sessionId), { recursive: true, force: true });
  await forgetSession(sessionId);
}
