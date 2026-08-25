import * as acp from "@agentclientprotocol/sdk";
import { randomBytes, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { hostname } from "node:os";
import { isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Sonyflake } from 'sonyflake';
import {
  createStoredSession,
  DEFAULT_DEEPSEEK_MODEL,
  DEFAULT_PROVIDER,
  DEFAULT_THINKING_EFFORT,
  deleteStoredSession,
  findSessionCwd,
  listStoredSessions,
  readStoredSession,
  clientLogPath,
  sessionLlmLogPath,
  terminalDirectory,
  writeSession,
  type NamedUserMessage,
  type ProviderId,
  type StoredSession,
  type ThinkingEffort,
  type UserContentPart,
} from "./storage.js";
import { appendJsonLine, makeLogEntry } from "./logger.js";
import { promptBlocksToPromptContent } from "./prompt-content.js";
import { executeLlmToolCall, type ToolExecutionResult } from "./tool-execution.js";
import {
  costFromUsage,
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getDefaultModel,
  getModelModalities,
  getModelPricing,
  runLlmStep,
  type LlmToolCall,
  type LlmUsage,
} from "./provider.js";
import { getOpenRouterModelOptions } from "./openrouter.js";
import { READ_MEDIA_TOOL_SCHEMA } from "./llm-client.js";
import { formatLlmError } from "./llm-errors.js";
import { BASH_TOOL_SCHEMA } from "./llm-client.js";

import {
  appendCacheDiagnostic,
  buildCacheDiagnostic,
  latestCacheDiagnostic,
  prefixDiagnosticHashes,
} from "./cache-diagnostics.js";
import { prepareReplayEvents, coalesceReplayEvents } from "./replay.js";
import { StreamThrottle } from "./stream-throttle.js";
import {
  buildEnvironmentMessage,
  buildSessionContinuedMessage,
  buildSystemPrompt,
  ENVIRONMENT_MESSAGE_NAME,
  getUserMessageName,
  isEnvironmentMessage,
} from "./system-prompt.js";
import { buildSkillInvocationPrompt, listSkills } from "./skills.js";
import {
  cacheHitPercent,
  emptyTurnStats,
  formatMs,
  formatTokens,
  formatYuan,
  roundYuan,
  showTurnStats,
  type TurnStats,
} from "./turn-stats.js";

interface ActiveSession {
  session: StoredSession;
  abortController: AbortController | null;
  /**
   * Lazily resolved input modalities of the session's model (provider/model
   * are locked after the first message, so this is stable per session).
   * Drives the read_media tool offering and prompt media gating.
   */
  mediaModalities: { image: boolean; audio: boolean } | null;
  /**
   * Set when the client sends `session/cancel`. Instead of aborting the
   * current unit of work immediately (which would kill an in-flight bash
   * command or cut the model off mid-thought), the running turn stops at the
   * next safe boundary and responds with stopReason "cancelled".
   *
   * Zed's own flow relies on this: `thread_view.rs` (`stop_current_and_send_new_message`
   * and `dispatch_queued_entry`) sends `session/cancel`, awaits the prompt
   * response, and only then sends the follow-up `session/prompt`. So a
   * graceful stop is what lets the new user message be "inserted" after the
   * current tool call step / thinking step instead of interrupting it.
   */
  gracefulCancel: boolean;
  /** Hard-abort escape hatch scheduled when a graceful cancel is requested. */
  cancelTimer: NodeJS.Timeout | null;
  /**
   * The in-flight turn started by the current prompt (null when idle). New
   * prompts and load/resume/close/delete await it after a hard abort, so the
   * old turn's final history mutations and state.json saves complete before
   * anyone else touches the same session.
   */
  turnPromise: Promise<acp.PromptResponse> | null;
  /** Set while modality lookups keep failing, so the warn is logged once. */
  mediaModalitiesUnknownLogged: boolean;
}

/** Safety valve for graceful cancel: hard-abort after this long. 0 = wait forever. */
const GRACEFUL_CANCEL_TIMEOUT_MS = parseGracefulCancelTimeoutMs();

function parseGracefulCancelTimeoutMs(): number {
  const raw = process.env.ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS;
  if (!raw) {
    return 0;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

const MAX_TURN_STEPS = parseMaxTurnSteps();

function parseMaxTurnSteps(): number {
  const raw = process.env.ZEN_AGENT_MAX_TURN_STEPS;
  if (!raw) {
    return 25;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 25;
  }
  return parsed;
}

/**
 * Per-session provider selector. DeepSeek and OpenRouter can be used side by
 * side: each session picks its provider here (locked once the user sent the
 * first message, like model and thinking effort). New sessions default to
 * deepseek.
 */
const PROVIDER_CONFIG_OPTION = {
  id: "provider",
  name: "Provider",
  description: "LLM provider used for this session (locked after the first message)",
  category: "model",
  type: "select",
  currentValue: DEFAULT_PROVIDER,
  options: [
    {
      value: "deepseek",
      name: "DeepSeek",
      description: "DeepSeek's own API, billed in CNY",
    },
    {
      value: "openrouter",
      name: "OpenRouter",
      description: "OpenRouter model aggregator, billed in USD",
    },
  ],
};

const DEEPSEEK_MODEL_CONFIG_OPTION = {
  id: "model",
  name: "Model",
  description: "Deepseek model used for this session",
  category: "model",
  type: "select",
  currentValue: DEFAULT_DEEPSEEK_MODEL,
  options: [
    {
      value: "deepseek-v4-flash",
      name: "Deepseek V4 Flash",
      description: "Fast model for everyday coding tasks",
    },
    {
      value: "deepseek-v4-pro",
      name: "Deepseek V4 Pro",
      description: "More powerful model for complex tasks",
    },
  ],
};

/**
 * OpenRouter models offered in the session selector. Any OpenRouter model
 * slug can be used beyond this list via OPENROUTER_MODEL or
 * session/set_config_option; `openrouter/free` routes to OpenRouter's
 * free-tier models.
 */
const OPENROUTER_MODEL_OPTIONS: Array<{
  value: string;
  name: string;
  description: string;
}> = [
  {
    value: "openrouter/free",
    name: "OpenRouter Free",
    description: "OpenRouter's free-tier routing model",
  },
];

/**
 * Model selector for a provider. OpenRouter sessions get the live model
 * catalog (auto-fetched, cached on disk, static list as fallback); DeepSeek
 * sessions get the two fixed models.
 */
async function modelConfigOption(provider: ProviderId, cwd: string) {
  if (provider === "openrouter") {
    const options = (await getOpenRouterModelOptions(cwd)) ?? OPENROUTER_MODEL_OPTIONS;
    return {
      id: "model",
      name: "Model",
      description: "OpenRouter model used for this session",
      category: "model",
      type: "select",
      currentValue: getDefaultModel("openrouter"),
      options,
    };
  }
  return {
    ...DEEPSEEK_MODEL_CONFIG_OPTION,
    currentValue: DEFAULT_DEEPSEEK_MODEL,
  };
}

const THINKING_CONFIG_OPTION = {
  id: "thinking_effort",
  name: "Thinking Effort",
  description: "Reasoning effort used by the model",
  category: "thought_level",
  type: "select",
  currentValue: DEFAULT_THINKING_EFFORT,
  options: [
    { value: "off", name: "Off", description: "Disable extended thinking" },
    { value: "high", name: "High", description: "Use high reasoning effort" },
    { value: "max", name: "Max", description: "Use maximum reasoning effort" },
  ],
};

const AVAILABLE_COMMANDS: acp.AvailableCommand[] = [
  {
    name: "prompt",
    description: "Set a custom system prompt / instructions for this session",
    input: {
      hint: "custom system prompt or instructions",
    },
  },
  {
    name: "sandbox",
    description:
      "Run every bash tool call inside a bubblewrap sandbox for this session",
    input: {
      hint: "on | off | (empty for status)",
    },
  },
];

/**
 * Sonyflake's default machine-id space is 16 bits (max 0xFFFF); the old fixed
 * constant 0x0d000721 overflowed it and made the constructor throw. Deriving
 * the id from hostname:pid keeps ids unique across containers on one host and
 * across processes in one container. FNV-1a keeps this dependency-free.
 */
function sonyflakeMachineId(): number {
  let hash = 0x811c9dc5;
  const input = `${hostname()}:${process.pid}`;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0) & 0xffff;
}

const sonyflake = new Sonyflake({ machineId: sonyflakeMachineId() });

const BASE62_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

function toBase62(bigintNum: bigint): string {
  if (bigintNum === 0n) return "0";
  let result = "";
  let num = bigintNum;
  while (num > 0n) {
    result = BASE62_CHARS[Number(num % 62n)] + result;
    num = num / 62n;
  }
  return result;
}

function newMessageId(): string {
  const id = sonyflake.nextId(); // Returns a BigInt or stringified BigInt
  const bigintId = BigInt(id);
  return `msg_${toBase62(bigintId)}`; // Example: msg_7zK4X9p2Q
}

/**
 * Local-time startup timestamp in the same shape Zed's terminal artifacts
 * use, e.g. 2026-08-21-23-06-04, so client debug logs sort chronologically
 * and are human-readable next to terminal logs.
 */
function formatStartupTimestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join("-");
}

export class ZenAgent {
  private sessions = new Map<string, ActiveSession>();
  private clientCapabilities: acp.ClientCapabilities = {};
  /**
   * Most recent balance snapshot for the session's provider (CNY for
   * DeepSeek, USD for OpenRouter), captured at the end of each turn. The
   * delta to the next turn's snapshot is compared against the locally
   * estimated cost to verify token accounting (see verifyTurnCost). Null
   * until the first turn completes.
   */
  private lastObservedBalance: { currency: string; total: number } | null = null;
  /**
   * Per-startup debug log identity: "YYYY-MM-DD-HH-mm-ss_<uuid>", e.g.
   * 2026-08-21-23-06-04_<uuid>. Created once per agent process; all runtime
   * diagnostics for this run are appended to
   * <project>/.sessions/client/<startupKey>/log.jsonl.
   */
  private readonly startupLogKey = `${formatStartupTimestamp(new Date())}_${randomUUID()}`;

  private makeActiveSession(session: StoredSession): ActiveSession {
    return {
      session,
      abortController: null,
      gracefulCancel: false,
      cancelTimer: null,
      mediaModalities: null,
      turnPromise: null,
      mediaModalitiesUnknownLogged: false,
    };
  }

  /**
   * Input modalities of the session's model (memoized on the active session:
   * provider and model are locked after the first message).
   */
  private async mediaModalities(
    active: ActiveSession,
  ): Promise<{ image: boolean; audio: boolean }> {
    if (active.mediaModalities) {
      return active.mediaModalities;
    }
    // Only DEFINITIVE answers are memoized: an unknown lookup (OpenRouter
    // catalog fetch failed, slug not in any table) returns null and is
    // retried on every later call, so a transient offline start cannot pin a
    // wrong text-only answer — and hide read_media / media passthrough — for
    // the rest of the session.
    const resolved = await getModelModalities(
      active.session.config.provider,
      active.session.config.model,
    );
    if (!resolved) {
      if (!active.mediaModalitiesUnknownLogged) {
        active.mediaModalitiesUnknownLogged = true;
        void this.logRuntime(
          active.session.cwd,
          "warn",
          "model input modalities unknown; assuming text-only until lookup succeeds",
          {
            sessionId: active.session.sessionId,
            provider: active.session.config.provider,
            model: active.session.config.model,
          },
        );
      }
      return { image: false, audio: false };
    }
    if (active.mediaModalitiesUnknownLogged) {
      active.mediaModalitiesUnknownLogged = false;
      void this.logRuntime(
        active.session.cwd,
        "info",
        "model input modalities resolved after earlier unknown",
        {
          sessionId: active.session.sessionId,
          provider: active.session.config.provider,
          model: active.session.config.model,
          image: resolved.image,
          audio: resolved.audio,
        },
      );
    }
    active.mediaModalities = resolved;
    return resolved;
  }

  /**
   * Tool schemas for the session's LLM requests. read_media is appended when
   * the model accepts image or audio input; the list is part of the cached
   * prefix, so it must stay stable within a session.
   */
  private async sessionToolSchemas(active: ActiveSession): Promise<unknown[]> {
    const modalities = await this.mediaModalities(active);
    return modalities.image || modalities.audio
      ? [BASH_TOOL_SCHEMA, READ_MEDIA_TOOL_SCHEMA]
      : [BASH_TOOL_SCHEMA];
  }

  /** Clears any pending graceful-cancel state (flag + hard-abort timer). */
  private clearGracefulCancel(active: ActiveSession): void {
    active.gracefulCancel = false;
    if (active.cancelTimer) {
      clearTimeout(active.cancelTimer);
      active.cancelTimer = null;
    }
  }

  async initialize(
    params: acp.InitializeRequest,
  ): Promise<acp.InitializeResponse> {
    this.clientCapabilities = params.clientCapabilities ?? {};
    return {
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities: {
        loadSession: true,
        // Zed gates its paste / drag & drop / @-mention image UI on these
        // (message_editor.rs supports_images); whether an attached block is
        // actually sent to the model is decided per prompt by the active
        // model's input modalities (see getModelModalities).
        promptCapabilities: {
          image: true,
          audio: true,
        },
        sessionCapabilities: {
          list: {},
          delete: {},
          resume: {},
          close: {},
        },
      },
      agentInfo: {
        name: "zen-agent",
        title: "Zen Agent",
        version: "0.1.0",
      },
      authMethods: [],
    };
  }

  async authenticate(
    _params: acp.AuthenticateRequest,
  ): Promise<acp.AuthenticateResponse | void> {
    return {};
  }

  async newSession(
    params: acp.NewSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.NewSessionResponse> {
    if (!isAbsolute(params.cwd)) {
      throw new Error("cwd must be an absolute path");
    }
    const session = await createStoredSession(params.cwd);
    // Freeze the environment snapshot into the persisted conversation at
    // session creation. It sits right after the system prompt, so it must
    // stay byte-identical for the provider's context cache to keep hitting
    // across steps and restarts (a per-request regenerated message — e.g.
    // with a changing git status — would break the whole cached prefix).
    session.llmMessages.push({
      role: "user",
      name: ENVIRONMENT_MESSAGE_NAME,
      content: await buildEnvironmentMessage(session),
    });
    await writeSession(session);
    this.sessions.set(session.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session created", {
      sessionId: session.sessionId,
    });
    this.scheduleAvailableCommands(session.sessionId, cx);
    return {
      sessionId: session.sessionId,
      configOptions: await this.getConfigOptions(session),
    };
  }

  async loadSession(
    params: acp.LoadSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.LoadSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    // The previous ActiveSession (same id, older in-memory storage object)
    // may still be unwinding a hard-aborted turn that saves to state.json;
    // let it finish before writing the freshly loaded session to disk.
    const previous = this.sessions.get(params.sessionId);
    this.abortActiveSession(params.sessionId);
    await this.settlePreviousTurn(previous);
    await this.prepareResumedSession(session);
    this.sessions.set(params.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session loaded", {
      sessionId: session.sessionId,
    });

    const replayEvents = coalesceReplayEvents(
      prepareReplayEvents(session.events, session.cwd),
    );
    for (const update of replayEvents) {
      await cx.notify(acp.methods.client.session.update, {
        sessionId: session.sessionId,
        update,
      });
    }
    this.scheduleAvailableCommands(session.sessionId, cx);

    return {
      configOptions: await this.getConfigOptions(session),
    };
  }

  async resumeSession(
    params: acp.ResumeSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.ResumeSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    // See loadSession: wait out the aborted turn before rewriting disk state.
    const previous = this.sessions.get(params.sessionId);
    this.abortActiveSession(params.sessionId);
    await this.settlePreviousTurn(previous);
    await this.prepareResumedSession(session);
    this.sessions.set(params.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session resumed", {
      sessionId: session.sessionId,
    });
    this.scheduleAvailableCommands(session.sessionId, cx);
    return {
      configOptions: await this.getConfigOptions(session),
    };
  }

  async listSessions(
    params: acp.ListSessionsRequest,
  ): Promise<acp.ListSessionsResponse> {
    const sessions = await listStoredSessions(params.cwd ?? undefined);
    return { sessions };
  }

  async deleteSession(
    params: acp.DeleteSessionRequest,
  ): Promise<acp.DeleteSessionResponse> {
    const active = this.sessions.get(params.sessionId);
    const cwd = active?.session.cwd ?? (await findSessionCwd(params.sessionId));
    if (!cwd) {
      throw new Error(`Session ${params.sessionId} not found`);
    }
    this.abortActiveSession(params.sessionId);
    // The aborted turn would otherwise recreate state.json on its final save
    // after we deleted it.
    await this.settlePreviousTurn(active);
    this.sessions.delete(params.sessionId);
    await deleteStoredSession(cwd, params.sessionId);
    return {};
  }

  async closeSession(
    params: acp.CloseSessionRequest,
  ): Promise<acp.CloseSessionResponse> {
    const previous = this.sessions.get(params.sessionId);
    this.abortActiveSession(params.sessionId);
    await this.settlePreviousTurn(previous);
    this.sessions.delete(params.sessionId);
    return {};
  }

  async setSessionConfigOption(
    params: acp.SetSessionConfigOptionRequest,
  ): Promise<acp.SetSessionConfigOptionResponse> {
    const active = this.sessions.get(params.sessionId);
    if (!active) {
      throw new Error(`Session ${params.sessionId} not found`);
    }

    const value = String(params.value);

    // Provider, model and thinking effort are fixed once the user sent their
    // first message: changing them mid-conversation would mix model
    // behaviors and billing currencies within one thread.
    if (this.sessionHasStarted(active.session)) {
      throw new Error(
        `${params.configId} cannot be changed after the first message of a session`,
      );
    }

    switch (params.configId) {
      case "provider": {
        if (value !== "deepseek" && value !== "openrouter") {
          throw new Error(`Unknown provider: ${value}`);
        }
        if (value !== active.session.config.provider) {
          active.session.config.provider = value as ProviderId;
          // The previous provider's model likely does not exist on the new
          // one; reset to the provider's default so the selector stays valid.
          active.session.config.model = getDefaultModel(value as ProviderId);
        }
        break;
      }
      case "model": {
        if (active.session.config.provider === "openrouter") {
          // OpenRouter accepts any model slug; only the curated list is
          // offered in the selector.
          if (value.trim().length === 0) {
            throw new Error("Model must not be empty");
          }
        } else if (value !== "deepseek-v4-flash" && value !== "deepseek-v4-pro") {
          throw new Error(`Unknown model: ${value}`);
        }
        active.session.config.model = value;
        break;
      }
      case "thinking_effort": {
        if (value !== "off" && value !== "high" && value !== "max") {
          throw new Error(`Unknown thinking effort: ${value}`);
        }
        active.session.config.thinkingEffort = value as ThinkingEffort;
        break;
      }
      default:
        throw new Error(`Unknown config option: ${params.configId}`);
    }

    await this.save(active);
    return { configOptions: await this.getConfigOptions(active.session) };
  }

  /**
   * Whether the user has sent their first message to this session: the
   * conversation contains a user message that is not an auto-generated
   * environment snapshot. Skill invocations are injected as environment-named
   * messages, so only real user prompts (and their follow-ups) lock the
   * provider/model/thinking settings.
   */
  private sessionHasStarted(session: StoredSession): boolean {
    return session.llmMessages.some(
      (message) => message.role === "user" && !isEnvironmentMessage(message),
    );
  }

  /**
   * Graceful cancel (ACP `session/cancel`).
   *
   * Zed sends this notification for ALL interruption paths — the Stop button
   * (`thread_view.rs::cancel_generation`), force-send while generating
   * (`thread_view.rs::stop_current_and_send_new_message`) and "steer" queued
   * messages (`thread_view.rs::dispatch_queued_entry`) — and it carries no
   * reason field, so they are indistinguishable here.
   *
   * We therefore never hard-abort on cancel: the running turn keeps executing
   * until the current LLM step (thinking/answering) or bash tool call
   * completes, then responds `stopReason: "cancelled"`. Zed awaits that
   * response before delivering the follow-up message, which is what makes the
   * new message appear after the interrupted unit of work rather than
   * mid-way through it.
   *
   * A hard abort still happens on `session/close` / `session/delete`, and
   * optionally after `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` as an escape
   * hatch for runaway commands.
   */
  cancel(params: acp.CancelNotification): void {
    const active = this.sessions.get(params.sessionId);
    if (!active || active.gracefulCancel) {
      return;
    }
    if (!active.abortController) {
      // No turn is running; nothing to stop. The flag is not set so a later
      // prompt starts fresh.
      return;
    }
    active.gracefulCancel = true;
    void this.logRuntime(active.session.cwd, "info", "graceful cancel requested", {
      sessionId: params.sessionId,
    });
    if (GRACEFUL_CANCEL_TIMEOUT_MS > 0) {
      active.cancelTimer = setTimeout(() => {
        active.cancelTimer = null;
        void this.logRuntime(active.session.cwd, "warn", "graceful cancel timed out; hard abort", {
          sessionId: params.sessionId,
          timeoutMs: GRACEFUL_CANCEL_TIMEOUT_MS,
        });
        active.abortController?.abort();
      }, GRACEFUL_CANCEL_TIMEOUT_MS);
      active.cancelTimer.unref?.();
    }
  }

  async prompt(
    params: acp.PromptRequest,
    cx: acp.AgentContext,
  ): Promise<acp.PromptResponse> {
    const active = this.sessions.get(params.sessionId);
    if (!active) {
      throw new Error(`Session ${params.sessionId} not found`);
    }

    // A new prompt can only arrive after the previous turn's response in
    // Zed's flow (it awaits the cancelled turn first), but defensively abort
    // any still-running turn and WAIT until it fully settles before touching
    // the shared history: until it unwinds, the old turn keeps mutating
    // llmMessages/events and re-saving state.json, and interleaved writes
    // would interleave two turns' entries in one conversation.
    this.abortActiveSession(params.sessionId);
    await this.settlePreviousTurn(active);
    this.clearGracefulCancel(active);
    const controller = new AbortController();
    active.abortController = controller;

    try {
      const { text: userText, parts } = await promptBlocksToPromptContent(params.prompt);
      void this.logRuntime(active.session.cwd, "info", "prompt received", {
        sessionId: params.sessionId,
        text: userText,
      });
      const slashCommand = this.parseSlashCommand(userText);

      if (slashCommand) {
        active.session.events.push({
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text: userText },
        });
        await this.save(active);

        return this.handleSlashCommand(active, cx, slashCommand);
      }

      const mediaParts = parts.filter((part) => part.type !== "text");

      // Transcript events show the original blocks (Zed renders attached
      // images/audio in the thread), while the LLM message carries whatever
      // the active model can actually consume. Pure-text prompts keep the
      // single combined chunk shape used before media support.
      if (mediaParts.length === 0) {
        active.session.events.push({
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text: userText },
        });
      } else {
        for (const part of parts) {
          const block: acp.ContentBlock =
            part.type === "text"
              ? { type: "text", text: part.text }
              : part.type === "image"
                ? {
                    type: "image",
                    data: part.data,
                    mimeType: part.mimeType,
                    ...(part.uri ? { uri: part.uri } : {}),
                  }
                : { type: "audio", data: part.data, mimeType: part.mimeType };
          active.session.events.push({
            sessionUpdate: "user_message_chunk",
            content: block,
          });
        }
      }

      let llmContent: string | UserContentPart[];
      if (mediaParts.length === 0) {
        // Plain-string shape stays byte-compatible with existing history
        // and the provider's prefix cache.
        llmContent = userText;
      } else {
        const modalities = await this.mediaModalities(active);
        llmContent = [];
        const rawText = parts
          .filter((part): part is Extract<UserContentPart, { type: "text" }> => part.type === "text")
          .map((part) => part.text)
          .join("\n\n");
        if (rawText.length > 0) {
          llmContent.push({ type: "text", text: rawText });
        }
        for (const part of mediaParts) {
          if (
            (part.type === "image" && !modalities.image) ||
            (part.type === "audio" && !modalities.audio)
          ) {
            // The model cannot consume this modality; degrade to a note so
            // the turn still runs (and the user learns why).
            llmContent.push({
              type: "text",
              text:
                part.type === "image"
                  ? `[image attached${part.uri ? `: ${part.uri}` : ""} (${part.mimeType}) omitted: current model does not support image input]`
                  : `[audio attached (${part.mimeType}) omitted: current model does not support audio input]`,
            });
            continue;
          }
          llmContent.push(part);
        }
      }

      const userMessage: NamedUserMessage = {
        role: "user",
        content: llmContent,
        name: await getUserMessageName(active.session.cwd),
      };
      active.session.llmMessages.push(userMessage);
      await this.save(active);

      // IMPORTANT: must `await` here. `return promise` inside try/finally runs
      // the finally block IMMEDIATELY (the returned promise is adopted
      // asynchronously outside the function), which would null
      // active.abortController while the turn is still running and break
      // graceful cancel (cancel() would see no controller).
      const response = await this.runTurnTracked(active, cx, controller.signal);
      return response;
    } catch (error) {
      if (controller.signal.aborted) {
        void this.logRuntime(active.session.cwd, "warn", "prompt cancelled", {
          sessionId: params.sessionId,
        });
        return { stopReason: "cancelled" };
      }
      // Classify API failures (401/402/429/5xx/context overflow) into
      // actionable guidance for the user; the original error is kept as the
      // thrown instance so its stack and cause survive.
      const formatted = await formatLlmError(error, {
        provider: active.session.config.provider,
      });
      void this.logRuntime(active.session.cwd, "error", "prompt failed", {
        sessionId: params.sessionId,
        error: formatted,
      });
      if (error instanceof Error && formatted !== error.message) {
        error.message = formatted;
      }
      throw error;
    } finally {
      if (active.abortController === controller) {
        active.abortController = null;
      }
      this.clearGracefulCancel(active);
      active.session.updatedAt = new Date().toISOString();
      await this.save(active).catch(() => {});
    }
  }

  /**
   * Runs one turn and records its promise on the session so concurrent entry
   * points (new prompt, load/resume, close/delete) can wait out the old unit
   * of work instead of racing it on the shared history.
   */
  private runTurnTracked(
    active: ActiveSession,
    cx: acp.AgentContext,
    signal: AbortSignal,
  ): Promise<acp.PromptResponse> {
    const turn = this.runTurn(active, cx, signal).finally(() => {
      if (active.turnPromise === turn) {
        active.turnPromise = null;
      }
    });
    active.turnPromise = turn;
    return turn;
  }

  /**
   * Waits out a still-running turn (already hard-aborted by the caller) so
   * its final history mutations and state.json saves complete before the
   * caller starts mutating the same session. No-op when idle or undefined.
   */
  private async settlePreviousTurn(
    active: ActiveSession | undefined,
  ): Promise<void> {
    const inflight = active?.turnPromise;
    if (!inflight) {
      return;
    }
    const startedAt = Date.now();
    void this.logRuntime(
      active.session.cwd,
      "warn",
      "previous turn still running; waiting for it to settle",
      { sessionId: active.session.sessionId },
    );
    try {
      await inflight;
    } catch {
      // The aborted turn's own caller handles (and logs) its error.
    }
    void this.logRuntime(active.session.cwd, "info", "previous turn settled", {
      sessionId: active.session.sessionId,
      waitedMs: Date.now() - startedAt,
    });
  }

  private async runTurn(
    active: ActiveSession,
    cx: acp.AgentContext,
    signal: AbortSignal,
  ): Promise<acp.PromptResponse> {
    const turn = emptyTurnStats();
    let contextUsed: number | undefined;
    const tools = await this.sessionToolSchemas(active);

    const finalize = async (
      stopReason: acp.StopReason,
    ): Promise<acp.PromptResponse> => {
      this.mergeTurnStats(active, turn);
      this.logTurnStats(active, turn, stopReason);
      await this.reportUsage(active, cx, contextUsed);
      if (showTurnStats()) {
        await this.emitTurnStats(active, cx, turn);
      }
      return {
        stopReason,
        usage: this.buildExperimentalUsage(active.session),
      };
    };

    for (let step = 0; step < MAX_TURN_STEPS; step++) {
      // Graceful-cancel boundary: stop before starting a new LLM step (i.e.
      // when the previous tool call step already completed, or when the
      // cancel arrived between steps).
      if (active.gracefulCancel) {
        return finalize("cancelled");
      }

      const assistantMessageId = newMessageId();
      // Captured once per step: the system prompt is part of the byte-stable
      // prefix the provider's context cache keys on, so it must be identical
      // for the request, the transcript, and the cache diagnostics.
      const mediaModalities = await this.mediaModalities(active);
      const system = buildSystemPrompt(active.session, {
        media: mediaModalities.image || mediaModalities.audio,
      });

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_request",
        timestamp: new Date().toISOString(),
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system,
        messages: active.session.llmMessages,
      });

      const stream = new StreamThrottle(async (kind, delta) => {
        if (kind === "thought") {
          await this.emit(active, cx, {
            sessionUpdate: "agent_thought_chunk",
            messageId: assistantMessageId,
            content: { type: "text", text: delta },
          });
        } else {
          await this.emit(active, cx, {
            sessionUpdate: "agent_message_chunk",
            messageId: assistantMessageId,
            content: { type: "text", text: delta },
          });
        }
      });

      const llmResult = await runLlmStep(active.session.config.provider, {
        messages: active.session.llmMessages,
        tools,
        signal,
        logRuntime: (level, message, details) => {
          void this.logRuntime(active.session.cwd, level, message, details);
        },
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system,
        onTextDelta: async (delta) => {
          stream.push("message", delta);
        },
        onReasoningDelta: async (delta) => {
          stream.push("thought", delta);
        },
      });

      await stream.drain();

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_response",
        timestamp: new Date().toISOString(),
        text: llmResult.text,
        toolCalls: llmResult.toolCalls,
        finishReason: llmResult.finishReason,
        usage: llmResult.usage,
      });

      if (llmResult.usage) {
        await this.accumulateTurnUsage(active, turn, llmResult.usage);
        await this.recordCacheDiagnostic(active, llmResult.usage, system);
        contextUsed = llmResult.usage.inputTokens;
        await this.reportUsage(active, cx, contextUsed);
        await this.logStepStats(
          active,
          step,
          llmResult.usage,
          llmResult.finishReason,
          llmResult.toolCalls.length,
        );
      }

      if (llmResult.toolCalls.length === 0) {
        // The model finished its answer. If a cancel arrived while it was
        // streaming, the turn still ends "cancelled" (the client interrupted
        // it), but the completed answer is kept in the conversation history.
        const content =
          llmResult.text.length > 0
            ? [{ type: "text" as const, text: llmResult.text }]
            : [];
        active.session.llmMessages.push({
          role: "assistant",
          content,
        });
        await this.save(active);
        return finalize(
          active.gracefulCancel ? "cancelled" : this.stopReasonFromFinish(llmResult.finishReason),
        );
      }

      // A cancel requested during the LLM step means "stop after thinking
      // completes". The step is done, so stop here and DISCARD the tool calls
      // it proposed: the user's follow-up supersedes them. They were never
      // executed, so nothing is persisted (and nothing was emitted to Zed
      // either — tool_call entries are only created in executeLlmToolCall).
      if (active.gracefulCancel) {
        return finalize("cancelled");
      }

      const assistantParts: Array<{
        type: "text";
        text: string;
      } | {
        type: "reasoning";
        text: string;
      } | {
        type: "tool-call";
        toolCallId: string;
        toolName: string;
        input: unknown;
      }> = [];
      if (llmResult.reasoning.length > 0 || active.session.config.thinkingEffort !== "off") {
        assistantParts.push({ type: "reasoning", text: llmResult.reasoning });
      }
      if (llmResult.text.length > 0) {
        assistantParts.push({ type: "text", text: llmResult.text });
      }
      const toolResults: ToolExecutionResult[] = [];

      for (const call of llmResult.toolCalls) {
        // The bash tool runs to completion even when a graceful cancel is
        // pending: the abort listener in executeLlmToolCall only fires on a
        // HARD abort (session close/delete or the timeout escape hatch), so
        // the command is never killed mid-run by a user follow-up.
        const toolStart = Date.now();
        const result = await this.executeLlmToolCall(active, cx, call, signal);
        turn.toolMs += Date.now() - toolStart;
        toolResults.push(result);
        assistantParts.push({
          type: "tool-call",
          toolCallId: call.id,
          toolName: call.name,
          input: call.input,
        });
        if (signal.aborted) {
          throw new Error("aborted");
        }
        // Graceful-cancel boundary: stop right after this tool call finishes
        // ("insert after the tool call step"). assistantParts only includes
        // the calls that actually ran, keeping the LLM message history
        // consistent (no unresolved tool calls).
        if (active.gracefulCancel) {
          break;
        }
      }

      active.session.llmMessages.push({
        role: "assistant",
        content: assistantParts,
      });
      active.session.llmMessages.push({
        role: "tool",
        content: toolResults.map((result) => ({
          type: "tool-result",
          toolCallId: result.toolCallId,
          toolName: result.toolName,
          output: result.output,
        })),
      });
      // read_media payload injection: the OpenAI-compatible tool role only
      // allows text content, so the actual image/audio parts ride in a
      // synthetic user message right after the tool results (the same trick
      // Claude Code's Read tool uses). The tool role messages above keep the
      // assistant's tool_calls paired - DeepSeek 400s on unpaired calls.
      const mediaResults = toolResults.filter((result) => result.attachedMedia);
      if (mediaResults.length > 0) {
        active.session.llmMessages.push({
          role: "user",
          content: mediaResults.flatMap((result) => {
            const media = result.attachedMedia!;
            return [
              {
                type: "text" as const,
                text: `[read_media] ${media.path} (${media.mimeType}, ${media.decodedBytes} bytes):`,
              },
              media.modality === "image"
                ? { type: "image" as const, mimeType: media.mimeType, data: media.data }
                : { type: "audio" as const, mimeType: media.mimeType, data: media.data },
            ];
          }),
        });
      }
      await this.save(active);

      // The completed tool call's results were persisted above, so the next
      // turn (the user's follow-up) has full context of what ran.
      if (active.gracefulCancel) {
        return finalize("cancelled");
      }
    }

    return finalize("max_turn_requests");
  }

  private async accumulateTurnUsage(
    active: ActiveSession,
    turn: TurnStats,
    usage: LlmUsage,
  ): Promise<void> {
    turn.steps += 1;
    turn.inputTokens += usage.inputTokens;
    turn.outputTokens += usage.outputTokens;
    turn.cacheReadTokens += usage.cacheReadTokens;
    turn.cacheMissTokens += usage.cacheMissTokens;
    turn.reasoningTokens += usage.reasoningTokens;
    turn.llmMs += usage.llmMs;
    turn.thinkingMs += usage.thinkingMs;
    turn.answeringMs += usage.answeringMs;
    turn.costYuan += costFromUsage(
      usage,
      await getModelPricing(active.session.config.provider, active.session.config.model),
    );
  }

  /**
   * Per-step cache diagnostics: hash the stable prefix (system prompt, tool
   * schemas, frozen environment snapshot) that the provider's context cache
   * keys on, infer the miss reason locally from hash deltas, and warn when
   * actionable churn is detected. Entries persist on the session (ring-
   * buffered at 50) so the history survives restarts.
   */
  private async recordCacheDiagnostic(
    active: ActiveSession,
    usage: LlmUsage,
    system: string,
  ): Promise<void> {
    const pricing = await getModelPricing(
      active.session.config.provider,
      active.session.config.model,
    );
    const prefix = prefixDiagnosticHashes({
      system,
      toolSpecs: await this.sessionToolSchemas(active),
      env: active.session.llmMessages.find(isEnvironmentMessage) ?? null,
    });
    const entry = buildCacheDiagnostic({
      turn: active.session.usage.turns + 1,
      model: active.session.config.model,
      usage: {
        inputTokens: usage.inputTokens,
        cachedTokens: usage.cacheReadTokens,
        missTokens: usage.cacheMissTokens,
      },
      pricing,
      prefix,
      previous: latestCacheDiagnostic(active.session.cacheDiagnostics),
    });
    active.session.cacheDiagnostics = appendCacheDiagnostic(
      active.session.cacheDiagnostics,
      entry,
    );
    if (
      entry.missReason === "system-prompt-changed" ||
      entry.missReason === "env-snapshot-changed" ||
      entry.missReason === "tool-list-changed" ||
      entry.missReason === "tool-schema-or-order-changed"
    ) {
      await this.logRuntime(active.session.cwd, "warn", "cache miss", {
        sessionId: active.session.sessionId,
        turn: entry.turn,
        model: entry.model,
        hitRate: entry.cacheHitRate,
        reason: entry.missReason,
        detail: entry.missReasonDetail,
      });
    }
  }

  private mergeTurnStats(active: ActiveSession, turn: TurnStats): void {
    const usage = active.session.usage;
    usage.turns += 1;
    usage.steps += turn.steps;
    usage.inputTokens += turn.inputTokens;
    usage.outputTokens += turn.outputTokens;
    usage.cacheReadTokens += turn.cacheReadTokens;
    usage.cacheMissTokens += turn.cacheMissTokens;
    usage.reasoningTokens += turn.reasoningTokens;
    usage.costYuan += turn.costYuan;
    usage.llmMs += turn.llmMs;
    usage.thinkingMs += turn.thinkingMs;
    usage.answeringMs += turn.answeringMs;
    usage.toolMs += turn.toolMs;
    // Keep per-turn stats so they survive process restarts (resume/load).
    active.session.turnStats.push(turn);
  }

  /**
   * Report context window usage and cumulative cost to the client.
   *
   * Zed maps this ACP update in `acp_thread.rs` (SessionUpdate::UsageUpdate)
   * into its TokenUsage/SessionCost and renders it in the agent panel header
   * (`agent_ui/.../thread_view.rs::render_token_usage`) as a token-usage ring
   * whose tooltip shows "Context: used / max" and "Cost: amount <currency>"
   * (CNY for DeepSeek, USD for OpenRouter). The ring also warns at 80% and
   * marks the thread exceeded at 100% of `size`.
   */
  private async reportUsage(
    active: ActiveSession,
    cx: acp.AgentContext,
    contextUsed: number | undefined,
  ): Promise<void> {
    if (contextUsed === undefined) {
      return;
    }
    const size = await getContextWindowTokens(
      active.session.config.provider,
      active.session.config.model,
    );
    await this.emit(active, cx, {
      sessionUpdate: "usage_update",
      used: Math.min(contextUsed, size),
      size,
      cost: {
        amount: roundYuan(active.session.usage.costYuan),
        currency: this.costCurrency(active),
      },
    });
  }

  /** Billing currency of the session's provider (CNY for DeepSeek, USD for OpenRouter). */
  private costCurrency(active: ActiveSession): "CNY" | "USD" {
    return active.session.config.provider === "openrouter" ? "USD" : "CNY";
  }

  /**
   * Emit a compact per-turn stats line as its own message in the thread.
   *
   * There is no ACP field for turns/steps/timing/cache-ratio, so this is
   * displayed as a regular agent message. It is deliberately NOT pushed to
   * `llmMessages` (it would waste context tokens on the next LLM request),
   * and it uses a fresh messageId so Zed renders it as its own bubble. Note
   * Zed auto-sends queued follow-up messages on the turn's Stopped event
   * (`agent_ui/.../conversation_view.rs`), so this stats bubble is always
   * followed by the user's next message rather than being dropped.
   */
  private async emitTurnStats(
    active: ActiveSession,
    cx: acp.AgentContext,
    turn: TurnStats,
  ): Promise<void> {
    if (turn.steps === 0) {
      return;
    }
    const symbol = active.session.config.provider === "openrouter" ? "$" : "¥";
    const text = [
      `Turn ${active.session.usage.turns} · ${turn.steps} step${turn.steps === 1 ? "" : "s"} · think ${formatMs(turn.thinkingMs)} · answer ${formatMs(turn.answeringMs)} · tools ${formatMs(turn.toolMs)}`,
      `in ${formatTokens(turn.inputTokens)} · out ${formatTokens(turn.outputTokens)} · cache hit ${cacheHitPercent(turn)} · ${symbol}${formatYuan(turn.costYuan)} (session ${symbol}${formatYuan(active.session.usage.costYuan)})`,
    ].join("\n");
    await this.emit(active, cx, {
      sessionUpdate: "agent_message_chunk",
      messageId: newMessageId(),
      content: { type: "text", text },
    });
    // Cross-check the locally estimated cost against the provider's actual
    // billing; fire-and-forget so a slow/failed balance request never
    // delays or breaks the stats bubble.
    void this.verifyTurnCost(active, turn);
  }

  /**
   * Verify turn stats against the provider's billing and log the result to
   * log.jsonl ("turn stats balance verify").
   *
   * DeepSeek's /user/balance (and OpenRouter's /auth/key) returns the current
   * account balance; the delta between this turn's snapshot and the previous
   * turn's snapshot should equal the turn's locally estimated cost
   * (turn.costYuan, derived from the usage fields the provider streams back
   * per step). A persistent mismatch means the pricing table or token
   * counting is wrong. Balance values are only two-decimal-precise, so
   * single-turn deltas are noisy — this is data gathering only, no behavior
   * change.
   */
  private async verifyTurnCost(
    active: ActiveSession,
    turn: TurnStats,
  ): Promise<void> {
    try {
      const provider = active.session.config.provider;
      const snapshot = await fetchBalanceSnapshot(provider);
      const details: Record<string, unknown> = {
        sessionId: active.session.sessionId,
        turn: active.session.usage.turns,
        provider,
        model: active.session.config.model,
        estimatedTurnCost: roundYuan(turn.costYuan),
        sessionEstimatedCost: roundYuan(active.session.usage.costYuan),
        balanceIsAvailable: snapshot.isAvailable,
        balanceCurrency: snapshot.currency,
        balanceTotal: snapshot.total,
        ...snapshot.details,
      };
      const before = this.lastObservedBalance;
      if (before !== null && before.currency === snapshot.currency) {
        const balanceDelta = before.total - snapshot.total;
        details.balanceBefore = before.total;
        details.balanceDelta = roundYuan(balanceDelta);
        details.deltaVsEstimated = roundYuan(balanceDelta - turn.costYuan);
      }
      this.lastObservedBalance = { currency: snapshot.currency, total: snapshot.total };
      await this.logRuntime(active.session.cwd, "info", "turn stats balance verify", details);
    } catch (error) {
      await this.logRuntime(active.session.cwd, "warn", "turn stats balance verify failed", {
        sessionId: active.session.sessionId,
        turn: active.session.usage.turns,
        provider: active.session.config.provider,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Experimental ACP PromptResponse.usage.
   *
   * The field is marked UNSTABLE in the ACP spec. Zed only consumes it behind
   * its `AcpBetaFeatureFlag` (`acp_thread.rs`, "response.usage"), where it
   * updates the thread's cumulative input/output token counters. We still
   * send it unconditionally — a stable fallback for other clients and for
   * when Zed's beta flag is enabled.
   */
  private buildExperimentalUsage(session: StoredSession): acp.Usage | null {
    const usage = session.usage;
    if (usage.turns === 0 && usage.inputTokens === 0 && usage.outputTokens === 0) {
      return null;
    }
    return {
      totalTokens: usage.inputTokens + usage.outputTokens,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thoughtTokens: usage.reasoningTokens,
      cachedReadTokens: usage.cacheReadTokens,
      cachedWriteTokens: null,
    };
  }

  /**
   * Prepare a stored session for continuation after a restart.
   *
   * 1. Backfill: sessions created before environment messages existed have
   *    none in their history; prepend a frozen snapshot so the cache prefix
   *    is stable (system + environment + history).
   * 2. Notify: append a fresh environment notification at the END of the
   *    conversation so the model knows the session was continued. Being
   *    appended after all history, it does not disturb the cached prefix —
   *    the system prompt, frozen environment message and persisted history
   *    stay byte-identical, so the provider's context cache keeps hitting.
   */
  private async prepareResumedSession(session: StoredSession): Promise<void> {
    if (
      session.llmMessages.length === 0 ||
      !isEnvironmentMessage(session.llmMessages[0]!)
    ) {
      session.llmMessages.unshift({
        role: "user",
        name: ENVIRONMENT_MESSAGE_NAME,
        content: await buildEnvironmentMessage(session),
      });
    }
    session.llmMessages.push({
      role: "user",
      name: ENVIRONMENT_MESSAGE_NAME,
      content: await buildSessionContinuedMessage(session),
    });
    await writeSession(session);
  }

  private async executeLlmToolCall(
    active: ActiveSession,
    cx: acp.AgentContext,
    call: LlmToolCall,
    signal: AbortSignal,
  ): Promise<ToolExecutionResult> {
    return executeLlmToolCall(
      {
        session: active.session,
        sandbox: this.sessionSandboxEnabled(active.session),
        mediaModalities: await this.mediaModalities(active),
        clientCapabilities: this.clientCapabilities,
        emit: (update) => this.emit(active, cx, update),
        logRuntime: (level, message, details) =>
          this.logRuntime(active.session.cwd, level, message, details),
      },
      cx,
      call,
      signal,
    );
  }

  private stopReasonFromFinish(finishReason: string): acp.StopReason {
    switch (finishReason) {
      case "length":
        return "max_tokens";
      case "content-filter":
        return "refusal";
      case "error":
        throw new Error(`Language model finished with error: ${finishReason}`);
      default:
        return "end_turn";
    }
  }

  private abortActiveSession(sessionId: string): void {
    const active = this.sessions.get(sessionId);
    if (!active) {
      return;
    }
    // Hard abort: used by session/close, session/delete and defensively when a
    // new prompt arrives while a turn is still running. Kills any in-flight
    // bash terminal (via the abort listener in executeLlmToolCall) and aborts
    // the LLM stream.
    this.clearGracefulCancel(active);
    active.abortController?.abort();
  }

  /**
   * Per-LLM-step stats for the per-startup debug log (log.jsonl): token
   * usage, cache hit ratio, cost and timing for one model request inside a
   * turn. Skipped when the provider reported no usage (the numbers would
   * all be zero/meaningless).
   */
  private async logStepStats(
    active: ActiveSession,
    step: number,
    usage: LlmUsage,
    finishReason: string,
    toolCallCount: number,
  ): Promise<void> {
    const pricing = await getModelPricing(
      active.session.config.provider,
      active.session.config.model,
    );
    void this.logRuntime(active.session.cwd, "info", "llm step stats", {
      sessionId: active.session.sessionId,
      step: step + 1,
      provider: active.session.config.provider,
      model: active.session.config.model,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      cacheReadTokens: usage.cacheReadTokens,
      cacheMissTokens: usage.cacheMissTokens,
      cacheHitPercent: cacheHitPercent(usage),
      reasoningTokens: usage.reasoningTokens,
      costYuan: costFromUsage(usage, pricing),
      llmMs: usage.llmMs,
      thinkingMs: usage.thinkingMs,
      answeringMs: usage.answeringMs,
      finishReason,
      toolCalls: toolCallCount,
    });
  }

  /**
   * Per-turn stats for the per-startup debug log (log.jsonl): aggregate of
   * all LLM steps in the turn plus tool execution time and the stop reason.
   * Called from finalize() after mergeTurnStats() so `usage.turns` already
   * points at this turn.
   */
  private logTurnStats(
    active: ActiveSession,
    turn: TurnStats,
    stopReason: acp.StopReason,
  ): void {
    void this.logRuntime(active.session.cwd, "info", "turn stats", {
      sessionId: active.session.sessionId,
      turn: active.session.usage.turns,
      provider: active.session.config.provider,
      model: active.session.config.model,
      stopReason,
      steps: turn.steps,
      inputTokens: turn.inputTokens,
      outputTokens: turn.outputTokens,
      cacheReadTokens: turn.cacheReadTokens,
      cacheMissTokens: turn.cacheMissTokens,
      cacheHitPercent: cacheHitPercent(turn),
      reasoningTokens: turn.reasoningTokens,
      costYuan: turn.costYuan,
      llmMs: turn.llmMs,
      thinkingMs: turn.thinkingMs,
      answeringMs: turn.answeringMs,
      toolMs: turn.toolMs,
    });
  }

  private async logRuntime(
    cwd: string,
    level: "debug" | "info" | "warn" | "error",
    message: string,
    details?: Record<string, unknown>,
  ): Promise<void> {
    try {
      await appendJsonLine(
        clientLogPath(cwd, this.startupLogKey),
        makeLogEntry(level, message, details),
      );
    } catch {
      // Logging must never break the agent.
    }
  }

  private async logLlmExchange(
    cwd: string,
    sessionId: string,
    entry: Record<string, unknown>,
  ): Promise<void> {
    try {
      await appendJsonLine(
        sessionLlmLogPath(cwd, sessionId),
        entry,
      );
    } catch {
      // Logging must never break the agent.
    }
  }

  private scheduleAvailableCommands(
    sessionId: string,
    cx: acp.AgentContext,
  ): void {
    setTimeout(() => {
      void this.sendAvailableCommands(sessionId, cx).catch((error) => {
        console.error("Failed to send available commands:", error);
      });
    }, 0);
  }

  private async sendAvailableCommands(
    sessionId: string,
    cx: acp.AgentContext,
  ): Promise<void> {
    const active = this.sessions.get(sessionId);
    const availableCommands = active
      ? await this.buildAvailableCommands(active.session)
      : AVAILABLE_COMMANDS;
    await cx.notify(acp.methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands,
      },
    });
  }

  /**
   * Built-in slash commands plus one `/skill-name` command per installed
   * Agent Skill (discovered live from the session cwd, independent of the
   * `ZEN_AGENT_SHOW_SKILLS_CATALOG` environment flag).
   */
  private async buildAvailableCommands(
    session: StoredSession,
  ): Promise<acp.AvailableCommand[]> {
    const commands: acp.AvailableCommand[] = [...AVAILABLE_COMMANDS];
    for (const skill of await listSkills(session.cwd)) {
      commands.push({
        name: skill.name,
        description:
          skill.description ||
          `Run the "${skill.name}" skill (installed Agent Skill)`,
        input: {
          hint:
            skill.description ||
            `what to pass to the ${skill.name} skill`,
        },
      });
    }
    return commands;
  }

  private parseSlashCommand(text: string): {
    name: string;
    argument: string;
  } | null {
    const trimmed = text.trim();
    if (!trimmed.startsWith("/")) {
      return null;
    }
    const match = trimmed.match(/^\/(\S+)\s*([\s\S]*)$/);
    if (!match) {
      return { name: trimmed.slice(1).trim(), argument: "" };
    }
    return {
      name: match[1].toLowerCase(),
      argument: match[2].trim(),
    };
  }

  private async handleSlashCommand(
    active: ActiveSession,
    cx: acp.AgentContext,
    command: { name: string; argument: string },
  ): Promise<acp.PromptResponse> {
    switch (command.name) {
      case "prompt":
        return {
          stopReason: await this.handlePromptSlashCommand(active, cx, command.argument),
        };
      case "sandbox":
        return {
          stopReason: await this.handleSandboxSlashCommand(active, cx, command.argument),
        };
      default: {
        const skillStop = await this.handleSkillSlashCommand(active, cx, command);
        if (skillStop) {
          return skillStop;
        }
        await this.emit(active, cx, {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: `Unknown slash command: /${command.name}`,
          },
        });
        return { stopReason: "end_turn" };
      }
    }
  }

  /**
   * `/skill-name <argument>` invokes an installed Agent Skill: the skill's
   * `SKILL.md` is read and injected as the user message that starts the
   * next model turn, so the model follows the skill's instructions with its
   * bash tool. Works for every installed skill, regardless of
   * `ZEN_AGENT_SHOW_SKILLS_CATALOG` (that flag only controls the frozen
   * environment catalog, not slash commands). Returns null when no installed
   * skill matches the command name.
   */
  private async handleSkillSlashCommand(
    active: ActiveSession,
    cx: acp.AgentContext,
    command: { name: string; argument: string },
  ): Promise<acp.PromptResponse | null> {
    const skills = await listSkills(active.session.cwd);
    const skill = skills.find((s) => s.name.toLowerCase() === command.name);
    if (!skill) {
      return null;
    }

    active.session.llmMessages.push({
      role: "user",
      content: await buildSkillInvocationPrompt(skill, command.argument),
      name: ENVIRONMENT_MESSAGE_NAME,
    });
    await this.save(active);
    void this.logRuntime(active.session.cwd, "info", "skill invoked via slash command", {
      sessionId: active.session.sessionId,
      skill: skill.name,
      scope: skill.scope,
      disableModelInvocation: skill.disableModelInvocation,
    });

    const signal = active.abortController?.signal ?? new AbortController().signal;
    return this.runTurnTracked(active, cx, signal);
  }

  private async handlePromptSlashCommand(
    active: ActiveSession,
    cx: acp.AgentContext,
    argument: string,
  ): Promise<acp.StopReason> {
    if (!argument) {
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: buildSystemPrompt(active.session),
        },
      });
      return "end_turn";
    }

    active.session.config.systemPrompt = argument;
    await this.save(active);
    void this.logRuntime(active.session.cwd, "info", "system prompt updated", {
      sessionId: active.session.sessionId,
    });

    await this.emit(active, cx, {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: "System prompt updated for this session.",
      },
    });

    return "end_turn";
  }

  /**
   * `/sandbox` toggles bwrap wrapping of bash tool calls for this session.
   *
   * Unlike `ZEN_AGENT_SANDBOX=1` (a global env policy set at startup), the
   * slash command changes the per-session `config.sandbox` flag at runtime
   * and persists it across restarts. The env policy still applies on top:
   * when `ZEN_AGENT_SANDBOX=1` the sandbox cannot be turned off per session.
   */
  private async handleSandboxSlashCommand(
    active: ActiveSession,
    cx: acp.AgentContext,
    argument: string,
  ): Promise<acp.StopReason> {
    const normalized = argument.trim().toLowerCase();
    const envForced = process.env.ZEN_AGENT_SANDBOX === "1";
    const effective = this.sessionSandboxEnabled(active.session);

    if (normalized === "" || normalized === "status") {
      const via =
        envForced && !active.session.config.sandbox
          ? " (enforced by ZEN_AGENT_SANDBOX=1)"
          : active.session.config.sandbox
            ? " (session)"
            : " (off)";
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text:
            `Bash tool sandbox: ${effective ? "ON" : "OFF"}${via}\n` +
            "Usage: /sandbox on | off",
        },
      });
      return "end_turn";
    }

    const enabled =
      normalized === "on" || normalized === "1" || normalized === "true" || normalized === "yes";
    const disabled =
      normalized === "off" || normalized === "0" || normalized === "false" || normalized === "no";

    if (!enabled && !disabled) {
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: `Unknown /sandbox argument: ${argument}
Usage: /sandbox on | off`,
        },
      });
      return "end_turn";
    }

    if (envForced) {
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: enabled
            ? "Bash tool sandbox is already ON (enforced by ZEN_AGENT_SANDBOX=1)."
            : "Cannot disable: ZEN_AGENT_SANDBOX=1 forces the bash tool sandbox on.",
        },
      });
      return "end_turn";
    }

    active.session.config.sandbox = enabled;
    await this.save(active);
    void this.logRuntime(active.session.cwd, "info", "bash sandbox toggled", {
      sessionId: active.session.sessionId,
      sandbox: enabled,
    });

    await this.emit(active, cx, {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: enabled
          ? "Bash tool calls are now sandboxed with bubblewrap for this session."
          : "Bash tool sandbox disabled for this session.",
      },
    });

    return "end_turn";
  }

  /**
   * Effective sandbox state for a session: the per-session `/sandbox` flag
   * OR the global `ZEN_AGENT_SANDBOX=1` env policy (which always applies).
   */
  private sessionSandboxEnabled(session: StoredSession): boolean {
    return session.config.sandbox || process.env.ZEN_AGENT_SANDBOX === "1";
  }

  private async getConfigOptions(session: StoredSession): Promise<acp.SessionConfigOption[]> {
    return [
      {
        ...PROVIDER_CONFIG_OPTION,
        currentValue: session.config.provider,
      } as acp.SessionConfigOption,
      {
        ...(await modelConfigOption(session.config.provider, session.cwd)),
        currentValue: session.config.model,
      } as acp.SessionConfigOption,
      {
        ...THINKING_CONFIG_OPTION,
        currentValue: session.config.thinkingEffort,
      } as acp.SessionConfigOption,
    ];
  }

  private async emit(
    active: ActiveSession,
    cx: acp.AgentContext,
    update: acp.SessionUpdate,
  ): Promise<void> {
    await cx.notify(acp.methods.client.session.update, {
      sessionId: active.session.sessionId,
      update,
    });
    active.session.events.push(update);
  }

  private async save(active: ActiveSession): Promise<void> {
    active.session.updatedAt = new Date().toISOString();
    await writeSession(active.session);
  }
}
