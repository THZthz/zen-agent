import * as acp from '@agentclientprotocol/sdk';
import { randomUUID } from 'node:crypto';
import {
  writeSession,
  validateSessionId,
  type ProviderId,
  type StoredSession,
} from '../session/storage.js';
import { makeLogEntry } from '../util/logger.js';
import { insertLlmLogEntry, insertRuntimeLogEntry } from '../session/db.js';
import { executeLlmToolCall, type ToolExecutionResult } from '../tools/execution.js';
import { getModelModalities, type LlmToolCall } from '../providers/index.js';
import { BASH_TOOL_SCHEMA, READ_MEDIA_TOOL_SCHEMA } from '../providers/llm-client.js';
import {
  buildEnvironmentMessage,
  ENVIRONMENT_MESSAGE_NAME,
  isEnvironmentMessage,
} from '../session/system-prompt.js';
import { formatStartupTimestamp } from './config.js';

/** Base class shape used by the ZenAgent mixins (see agent.ts). */
export type Constructor<T = object> = new (...args: any[]) => T;

export interface ActiveSession {
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

/**
 * ZenAgent shared state and low-level plumbing. The ACP-facing behavior is
 * layered on top by the session / turn / prompt mixins in agent.ts, so this
 * class deliberately has no session lifecycle, LLM turn loop or slash-command
 * logic of its own.
 */
export class ZenAgentCore {
  protected sessions = new Map<string, ActiveSession>();
  private readonly sessionOperationTails = new Map<string, Promise<void>>();
  protected clientCapabilities: acp.ClientCapabilities = {};
  /**
   * Most recent balance snapshot per PROVIDER (CNY for DeepSeek, USD for
   * OpenRouter), captured at the end of each turn. The delta to the next
   * turn's snapshot is compared against the locally estimated cost to verify
   * token accounting (see verifyTurnCost). Keyed by provider so concurrent
   * sessions on different providers never compare each other's deltas.
   */
  protected readonly lastBalanceByProvider = new Map<
    ProviderId,
    { currency: string; total: number }
  >();
  /**
   * Per-startup log identity: "YYYY-MM-DD-HH-mm-ss_<uuid>", e.g.
   * 2026-08-21-23-06-04_<uuid>. Created once per agent process; all runtime
   * diagnostics for this run are inserted into the runtime_log table grouped
   * by this key.
   */
  protected readonly startupLogKey = `${formatStartupTimestamp(new Date())}_${randomUUID()}`;

  async initialize(params: acp.InitializeRequest): Promise<acp.InitializeResponse> {
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
        name: 'zen-agent',
        title: 'Zen Agent',
        version: '0.1.0',
      },
      authMethods: [],
    };
  }

  async authenticate(_params: acp.AuthenticateRequest): Promise<acp.AuthenticateResponse | void> {
    return {};
  }

  protected makeActiveSession(session: StoredSession): ActiveSession {
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
   * Serialize every stateful operation for one session. The tail is installed
   * synchronously before the first await, closing the check-then-await race that
   * allowed simultaneous prompts and lifecycle operations to share one history.
   */
  protected async withSessionOperation<T>(
    sessionId: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    validateSessionId(sessionId);
    const previous = this.sessionOperationTails.get(sessionId) ?? Promise.resolve();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.sessionOperationTails.set(sessionId, gate);

    await previous;
    try {
      return await operation();
    } finally {
      release();
      if (this.sessionOperationTails.get(sessionId) === gate) {
        this.sessionOperationTails.delete(sessionId);
      }
    }
  }

  /**
   * Input modalities of the session's model (memoized on the active session:
   * provider and model are locked after the first message).
   */
  protected async mediaModalities(
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
          'warn',
          'model input modalities unknown; assuming text-only until lookup succeeds',
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
        'info',
        'model input modalities resolved after earlier unknown',
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
  protected async sessionToolSchemas(active: ActiveSession): Promise<unknown[]> {
    // /tools off disables every tool: no schemas are sent, so the model
    // cannot legitimately propose a tool call this turn.
    if (!active.session.config.toolsEnabled) {
      return [];
    }
    const modalities = await this.mediaModalities(active);
    return modalities.image || modalities.audio
      ? [BASH_TOOL_SCHEMA, READ_MEDIA_TOOL_SCHEMA]
      : [BASH_TOOL_SCHEMA];
  }

  /** Clears any pending graceful-cancel state (flag + hard-abort timer). */
  protected clearGracefulCancel(active: ActiveSession): void {
    active.gracefulCancel = false;
    if (active.cancelTimer) {
      clearTimeout(active.cancelTimer);
      active.cancelTimer = null;
    }
  }

  /**
   * Whether the user has sent their first message to this session: the
   * conversation contains a user message that is not an auto-generated
   * environment snapshot. Skill invocations are injected as environment-named
   * messages, so only real user prompts (and their follow-ups) lock the
   * provider/model/thinking settings.
   */
  protected sessionHasStarted(session: StoredSession): boolean {
    return session.llmMessages.some(
      (message) => message.role === 'user' && !isEnvironmentMessage(message),
    );
  }

  /** Drop every auto-generated environment snapshot/continuation message. */
  protected removeEnvironmentMessages(session: StoredSession): void {
    session.llmMessages = session.llmMessages.filter((message) => !isEnvironmentMessage(message));
  }

  /**
   * Inject the frozen environment snapshot at the FRONT of the conversation
   * (right after the system prompt, where newSession places it) when it is
   * missing. Used when a chat-only session re-enables tools.
   */
  protected async ensureEnvironmentMessage(session: StoredSession): Promise<void> {
    if (!session.llmMessages.some(isEnvironmentMessage)) {
      session.llmMessages.unshift({
        role: 'user',
        name: ENVIRONMENT_MESSAGE_NAME,
        content: await buildEnvironmentMessage(session),
      });
    }
  }

  /**
   * Waits out a still-running turn (already hard-aborted by the caller) so
   * its final history mutations and state.json saves complete before the
   * caller starts mutating the same session. No-op when idle or undefined.
   */
  protected async settlePreviousTurn(active: ActiveSession | undefined): Promise<void> {
    const inflight = active?.turnPromise;
    if (!inflight) {
      return;
    }
    const startedAt = Date.now();
    void this.logRuntime(
      active.session.cwd,
      'warn',
      'previous turn still running; waiting for it to settle',
      { sessionId: active.session.sessionId },
    );
    try {
      await inflight;
    } catch {
      // The aborted turn's own caller handles (and logs) its error.
    }
    void this.logRuntime(active.session.cwd, 'info', 'previous turn settled', {
      sessionId: active.session.sessionId,
      waitedMs: Date.now() - startedAt,
    });
  }

  protected abortActiveSession(sessionId: string): void {
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

  protected stopReasonFromFinish(finishReason: string): acp.StopReason {
    switch (finishReason) {
      case 'length':
        return 'max_tokens';
      case 'content-filter':
        return 'refusal';
      case 'error':
        throw new Error(`Language model finished with error: ${finishReason}`);
      default:
        return 'end_turn';
    }
  }

  /**
   * Effective sandbox state for a session: the per-session `/sandbox` flag
   * OR the global `ZEN_AGENT_SANDBOX=1` env policy (which always applies).
   */
  protected sessionSandboxEnabled(session: StoredSession): boolean {
    return session.config.sandbox || process.env.ZEN_AGENT_SANDBOX === '1';
  }

  protected async executeLlmToolCall(
    active: ActiveSession,
    cx: acp.AgentContext,
    call: LlmToolCall,
    signal: AbortSignal,
  ): Promise<ToolExecutionResult> {
    return executeLlmToolCall(
      {
        session: active.session,
        sandbox: this.sessionSandboxEnabled(active.session),
        toolsEnabled: active.session.config.toolsEnabled,
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

  protected async logRuntime(
    cwd: string,
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    details?: Record<string, unknown>,
  ): Promise<void> {
    insertRuntimeLogEntry(this.startupLogKey, cwd, makeLogEntry(level, message, details));
  }

  protected async logLlmExchange(sessionId: string, entry: Record<string, unknown>): Promise<void> {
    insertLlmLogEntry(sessionId, entry);
  }

  protected async emit(
    active: ActiveSession,
    cx: acp.AgentContext,
    update: acp.SessionUpdate,
  ): Promise<void> {
    await cx.notify(acp.methods.client.session.update, {
      sessionId: active.session.sessionId,
      update,
    });
    active.session.events.push(this.eventForTranscript(update));
  }

  /**
   * Transcript copy of an update for state.json.
   *
   * The final bash `tool_call_update` carries the full terminal output THREE
   * times on the wire (rawOutput.output, _meta.terminal_output.data and the
   * terminal log file on disk). The transcript keeps only rawOutput.output:
   * replay (prepareReplayEvents/remapTerminalContent) re-derives
   * _meta.terminal_output/_meta.terminal_exit from it when they are missing —
   * exactly its legacy-session path — so replayed cards are byte-identical
   * while state.json halves in size per bash call. Attached media is still
   * stored twice by design (transcript blocks for Zed rendering + llmMessages
   * for the model); deduplicating that would change what either side sees.
   */
  protected eventForTranscript(update: acp.SessionUpdate): acp.SessionUpdate {
    if (update.sessionUpdate !== 'tool_call_update') {
      return update;
    }
    const meta = (update as { _meta?: Record<string, unknown> })._meta;
    if (!meta || !('terminal_output' in meta || 'terminal_exit' in meta)) {
      return update;
    }
    const stripped = Object.fromEntries(
      Object.entries(meta).filter(([key]) => key !== 'terminal_output' && key !== 'terminal_exit'),
    );
    // Spread order matters: the base must exclude _meta, otherwise an empty
    // replacement silently keeps the original payload.
    const { _meta: _dropped, ...rest } = update as typeof update & Record<string, unknown>;
    return {
      ...rest,
      ...(Object.keys(stripped).length > 0 ? { _meta: stripped } : {}),
    } as acp.SessionUpdate;
  }

  protected async save(active: ActiveSession): Promise<void> {
    active.session.updatedAt = new Date().toISOString();
    await writeSession(active.session);
  }

  /**
   * Shutdown path: hard-abort every active turn and wait (bounded) for them
   * to unwind. Aborting kills any in-flight client terminal via the abort
   * listeners in executeLlmToolCall and ends the LLM streams, so the process
   * can exit without leaving terminals running in Zed or state.json writes
   * half-done.
   */
  async dispose(timeoutMs = 5_000): Promise<void> {
    const actives = [...this.sessions.values()];
    for (const active of actives) {
      void this.logRuntime(active.session.cwd, 'info', 'shutdown: aborting session turn', {
        sessionId: active.session.sessionId,
        hadRunningTurn: active.turnPromise !== null,
      });
      this.abortActiveSession(active.session.sessionId);
    }
    await Promise.race([
      Promise.allSettled(actives.map((active) => active.turnPromise ?? Promise.resolve())),
      new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, timeoutMs);
        timer.unref?.();
      }),
    ]);
  }
}
