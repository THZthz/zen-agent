import * as acp from "@agentclientprotocol/sdk";
import { randomBytes } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ModelMessage } from "ai";
import {
  createStoredSession,
  DEFAULT_MODEL,
  DEFAULT_THINKING_EFFORT,
  deleteStoredSession,
  emptySessionUsage,
  findSessionCwd,
  listStoredSessions,
  readStoredSession,
  runtimeLogPath,
  sessionDirectory,
  sessionLlmLogPath,
  terminalDirectory,
  writeSession,
  type ModelId,
  type StoredSession,
  type ThinkingEffort,
} from "./storage.js";
import { appendJsonLine, makeLogEntry } from "./logger.js";
import {
  costYuan,
  getContextWindowTokens,
  getModelPricing,
  runLlmStep,
  SYSTEM_PROMPT,
  type LlmToolCall,
  type LlmUsage,
} from "./llm/deepseek.js";

interface ActiveSession {
  session: StoredSession;
  abortController: AbortController | null;
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

const MODEL_CONFIG_OPTION = {
  id: "model",
  name: "Model",
  description: "Deepseek model used for this session",
  category: "model",
  type: "select",
  currentValue: DEFAULT_MODEL,
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
];

function newMessageId(): string {
  return `msg_${randomBytes(8).toString("hex")}`;
}

function newSessionIdForPrompt(): string {
  return newMessageId();
}

type StreamKind = "thought" | "message";

/**
 * Batches streamed LLM deltas into small, throttled chunks.
 *
 * Zed renders each `agent_message_chunk`/`agent_thought_chunk` update
 * incrementally (`agent_ui/.../thread_view.rs`), and flooding it with
 * per-token updates across a slow JSON-RPC pipe makes the UI janky and drops
 * chunks. Throttling to ~24 chars per 16ms tick keeps the output visibly
 * streaming while staying well within Zed's per-update handling budget.
 */
class StreamThrottle {
  private queue: Array<{ kind: StreamKind; text: string }> = [];
  private timer: NodeJS.Timeout | null = null;
  private running = false;

  constructor(
    private emit: (kind: StreamKind, text: string) => Promise<void>,
    private intervalMs = 16,
    private maxCharsPerTick = 24,
  ) {}

  push(kind: StreamKind, text: string): void {
    if (!text) {
      return;
    }
    this.queue.push({ kind, text });
    this.schedule();
  }

  async drain(): Promise<void> {
    while (this.queue.length > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.intervalMs));
    }
  }

  private schedule(): void {
    if (this.running || this.timer) {
      return;
    }
    this.timer = setTimeout(() => {
      void this.tick();
    }, this.intervalMs);
  }

  private async tick(): Promise<void> {
    this.timer = null;
    if (this.queue.length === 0) {
      this.running = false;
      return;
    }

    this.running = true;
    let remaining = this.maxCharsPerTick;

    while (remaining > 0 && this.queue.length > 0) {
      const item = this.queue[0];
      if (item.text.length <= remaining) {
        this.queue.shift();
        await this.emit(item.kind, item.text);
        remaining -= item.text.length;
      } else {
        const chunk = item.text.slice(0, remaining);
        item.text = item.text.slice(remaining);
        await this.emit(item.kind, chunk);
        remaining = 0;
      }
    }

    this.running = false;
    if (this.queue.length > 0) {
      this.schedule();
    }
  }
}

interface TurnStats {
  steps: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheMissTokens: number;
  reasoningTokens: number;
  costYuan: number;
  llmMs: number;
  thinkingMs: number;
  answeringMs: number;
  toolMs: number;
}

function emptyTurnStats(): TurnStats {
  return {
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

function formatMs(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTokens(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  }
  if (count >= 1000) {
    return `${(count / 1000).toFixed(1)}K`;
  }
  return String(count);
}

function formatYuan(amount: number): string {
  if (amount > 0 && amount < 0.01) {
    return amount.toFixed(4);
  }
  return amount.toFixed(3);
}

function roundYuan(amount: number): number {
  return Math.round(amount * 10_000) / 10_000;
}

function cacheHitPercent(turn: TurnStats): string {
  const total = turn.cacheReadTokens + turn.cacheMissTokens;
  if (total === 0) {
    return "n/a";
  }
  return `${Math.round((turn.cacheReadTokens / total) * 100)}%`;
}

function showTurnStats(): boolean {
  const raw = process.env.ZEN_AGENT_SHOW_STATS;
  if (raw === undefined) {
    return true;
  }
  return !["0", "false", "no", "off"].includes(raw.trim().toLowerCase());
}

export class ZenAgent {
  private sessions = new Map<string, ActiveSession>();
  private clientCapabilities: acp.ClientCapabilities = {};

  private makeActiveSession(session: StoredSession): ActiveSession {
    return {
      session,
      abortController: null,
      gracefulCancel: false,
      cancelTimer: null,
    };
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
    this.sessions.set(session.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session created", {
      sessionId: session.sessionId,
    });
    this.scheduleAvailableCommands(session.sessionId, cx);
    return {
      sessionId: session.sessionId,
      configOptions: this.getConfigOptions(session),
    };
  }

  async loadSession(
    params: acp.LoadSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.LoadSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    this.abortActiveSession(params.sessionId);
    this.sessions.set(params.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session loaded", {
      sessionId: session.sessionId,
    });

    const replayEvents = this.coalesceReplayEvents(
      this.prepareReplayEvents(session.events),
    );
    for (const update of replayEvents) {
      await cx.notify(acp.methods.client.session.update, {
        sessionId: session.sessionId,
        update,
      });
    }
    this.scheduleAvailableCommands(session.sessionId, cx);

    return {
      configOptions: this.getConfigOptions(session),
    };
  }

  async resumeSession(
    params: acp.ResumeSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.ResumeSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    this.abortActiveSession(params.sessionId);
    this.sessions.set(params.sessionId, this.makeActiveSession(session));
    void this.logRuntime(params.cwd, "info", "session resumed", {
      sessionId: session.sessionId,
    });
    this.scheduleAvailableCommands(session.sessionId, cx);
    return {
      configOptions: this.getConfigOptions(session),
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
    this.sessions.delete(params.sessionId);
    await deleteStoredSession(cwd, params.sessionId);
    return {};
  }

  async closeSession(
    params: acp.CloseSessionRequest,
  ): Promise<acp.CloseSessionResponse> {
    this.abortActiveSession(params.sessionId);
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

    switch (params.configId) {
      case "model": {
        if (value !== "deepseek-v4-flash" && value !== "deepseek-v4-pro") {
          throw new Error(`Unknown model: ${value}`);
        }
        active.session.config.model = value as ModelId;
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
    return { configOptions: this.getConfigOptions(active.session) };
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
    // any still-running turn and always start with a clean cancel state.
    this.abortActiveSession(params.sessionId);
    this.clearGracefulCancel(active);
    const controller = new AbortController();
    active.abortController = controller;

    try {
      const userText = await this.promptBlocksToText(params.prompt);
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

        const stopReason = await this.handleSlashCommand(
          active,
          cx,
          slashCommand,
        );
        return { stopReason };
      }

      const userMessage: ModelMessage = {
        role: "user",
        content: userText,
      };
      active.session.llmMessages.push(userMessage);
      active.session.events.push({
        sessionUpdate: "user_message_chunk",
        content: { type: "text", text: userText },
      });
      await this.save(active);

      // IMPORTANT: must `await` here. `return promise` inside try/finally runs
      // the finally block IMMEDIATELY (the returned promise is adopted
      // asynchronously outside the function), which would null
      // active.abortController while the turn is still running and break
      // graceful cancel (cancel() would see no controller).
      const response = await this.runTurn(active, cx, controller.signal);
      return response;
    } catch (error) {
      if (controller.signal.aborted) {
        void this.logRuntime(active.session.cwd, "warn", "prompt cancelled", {
          sessionId: params.sessionId,
        });
        return { stopReason: "cancelled" };
      }
      void this.logRuntime(active.session.cwd, "error", "prompt failed", {
        sessionId: params.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    } finally {
      active.abortController = null;
      this.clearGracefulCancel(active);
      active.session.updatedAt = new Date().toISOString();
      await this.save(active).catch(() => {});
    }
  }

  private async runTurn(
    active: ActiveSession,
    cx: acp.AgentContext,
    signal: AbortSignal,
  ): Promise<acp.PromptResponse> {
    const turn = emptyTurnStats();
    let contextUsed: number | undefined;

    const finalize = async (
      stopReason: acp.StopReason,
    ): Promise<acp.PromptResponse> => {
      this.mergeTurnStats(active, turn);
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

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_request",
        timestamp: new Date().toISOString(),
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system: this.buildSystemPrompt(active.session),
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

      const llmResult = await runLlmStep({
        messages: active.session.llmMessages,
        signal,
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system: this.buildSystemPrompt(active.session),
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
        this.accumulateTurnUsage(active, turn, llmResult.usage);
        contextUsed = llmResult.usage.inputTokens;
        await this.reportUsage(active, cx, contextUsed);
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
        type: "tool-call";
        toolCallId: string;
        toolName: string;
        input: unknown;
      }> = [];
      if (llmResult.text.length > 0) {
        assistantParts.push({ type: "text", text: llmResult.text });
      }
      const toolResults: Array<{
        toolCallId: string;
        toolName: string;
        output: { type: "text"; value: string };
      }> = [];

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
      await this.save(active);

      // The completed tool call's results were persisted above, so the next
      // turn (the user's follow-up) has full context of what ran.
      if (active.gracefulCancel) {
        return finalize("cancelled");
      }
    }

    return finalize("max_turn_requests");
  }

  private accumulateTurnUsage(
    active: ActiveSession,
    turn: TurnStats,
    usage: LlmUsage,
  ): void {
    turn.steps += 1;
    turn.inputTokens += usage.inputTokens;
    turn.outputTokens += usage.outputTokens;
    turn.cacheReadTokens += usage.cacheReadTokens;
    turn.cacheMissTokens += usage.cacheMissTokens;
    turn.reasoningTokens += usage.reasoningTokens;
    turn.llmMs += usage.llmMs;
    turn.thinkingMs += usage.thinkingMs;
    turn.answeringMs += usage.answeringMs;
    turn.costYuan += costYuan(usage, getModelPricing(active.session.config.model));
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
  }

  /**
   * Report context window usage and cumulative cost to the client.
   *
   * Zed maps this ACP update in `acp_thread.rs` (SessionUpdate::UsageUpdate)
   * into its TokenUsage/SessionCost and renders it in the agent panel header
   * (`agent_ui/.../thread_view.rs::render_token_usage`) as a token-usage ring
   * whose tooltip shows "Context: used / max" and "Cost: amount CNY". The
   * ring also warns at 80% and marks the thread exceeded at 100% of `size`.
   */
  private async reportUsage(
    active: ActiveSession,
    cx: acp.AgentContext,
    contextUsed: number | undefined,
  ): Promise<void> {
    if (contextUsed === undefined) {
      return;
    }
    const size = getContextWindowTokens();
    await this.emit(active, cx, {
      sessionUpdate: "usage_update",
      used: Math.min(contextUsed, size),
      size,
      cost: {
        amount: roundYuan(active.session.usage.costYuan),
        currency: "CNY",
      },
    });
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
    const text = [
      `Turn ${active.session.usage.turns} · ${turn.steps} step${turn.steps === 1 ? "" : "s"} · think ${formatMs(turn.thinkingMs)} · answer ${formatMs(turn.answeringMs)} · tools ${formatMs(turn.toolMs)}`,
      `in ${formatTokens(turn.inputTokens)} · out ${formatTokens(turn.outputTokens)} · cache hit ${cacheHitPercent(turn)} · ¥${formatYuan(turn.costYuan)} (session ¥${formatYuan(active.session.usage.costYuan)})`,
    ].join("\n");
    await this.emit(active, cx, {
      sessionUpdate: "agent_message_chunk",
      messageId: newMessageId(),
      content: { type: "text", text },
    });
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

  private shellQuote(value: string): string {
    return `'${value.replace(/'/g, `'\\''`)}'`;
  }

  /**
   * Runs a bash tool call through Zed's ACP terminal API.
   *
   * Zed exposes client-side terminals (`acp::methods.client.terminal.*`) that
   * run in Zed's own PTY on the host; we create one per call, stream its
   * output to Zed as a `tool_call_update` (which Zed renders as a live
   * terminal card), and wait for exit before collecting output. The abort
   * listener below kills the terminal ONLY on a hard abort — a graceful
   * cancel (user follow-up / Stop) lets the command finish.
   */
  private async executeLlmToolCall(
    active: ActiveSession,
    cx: acp.AgentContext,
    call: LlmToolCall,
    signal: AbortSignal,
  ): Promise<{
    toolCallId: string;
    toolName: string;
    output: { type: "text"; value: string };
  }> {
    if (call.name !== "bash") {
      const message = `Unknown tool: ${call.name}`;
      await this.emit(active, cx, {
        sessionUpdate: "tool_call",
        toolCallId: call.id,
        title: `Unknown tool ${call.name}`,
        kind: "other",
        status: "failed",
        rawInput: call.input,
      });
      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status: "failed",
        content: [
          {
            type: "content",
            content: { type: "text", text: message },
          },
        ],
        rawOutput: { error: message },
      });
      return {
        toolCallId: call.id,
        toolName: call.name,
        output: { type: "text", value: message },
      };
    }

    const command = (call.input as { command?: unknown }).command;
    if (typeof command !== "string" || command.trim().length === 0) {
      const message = "bash tool requires a non-empty string command";
      await this.emit(active, cx, {
        sessionUpdate: "tool_call",
        toolCallId: call.id,
        title: "Invalid bash command",
        kind: "execute",
        status: "failed",
        rawInput: call.input,
      });
      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status: "failed",
        content: [
          {
            type: "content",
            content: { type: "text", text: message },
          },
        ],
        rawOutput: { error: message },
      });
      return {
        toolCallId: call.id,
        toolName: "bash",
        output: { type: "text", value: message },
      };
    }

    const terminalDir = terminalDirectory(
      active.session.cwd,
      active.session.sessionId,
    );
    const logPath = join(terminalDir, `terminal-${call.id}.log`);
    const commandScriptPath = join(terminalDir, `terminal-${call.id}.sh`);
    await mkdir(terminalDir, { recursive: true });
    await writeFile(commandScriptPath, command, "utf8");
    const scriptCommand = `bash ${this.shellQuote(commandScriptPath)}`;
    const wrappedCommand = `script -q -e -c ${this.shellQuote(scriptCommand)} ${this.shellQuote(logPath)}`;
    const toolStart = Date.now();

    await this.emit(active, cx, {
      sessionUpdate: "tool_call",
      toolCallId: call.id,
      title: `$ ${command}`,
      kind: "execute",
      status: "pending",
      rawInput: { command },
    });

    if (!this.clientCapabilities.terminal) {
      const message =
        "Zed terminal support is required for the bash tool, but the client did not advertise terminal: true";
      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status: "failed",
        content: [
          {
            type: "content",
            content: { type: "text", text: message },
          },
        ],
        rawOutput: { error: message },
      });
      return {
        toolCallId: call.id,
        toolName: "bash",
        output: { type: "text", value: message },
      };
    }

    let terminalId: string | undefined;
    let cancelledBySignal = false;

    const killTerminal = async () => {
      if (!terminalId) return;
      try {
        await cx.request(acp.methods.client.terminal.kill, {
          sessionId: active.session.sessionId,
          terminalId,
        });
      } catch {
        // The terminal may already have exited.
      }
    };

    const releaseTerminal = async () => {
      if (!terminalId) return;
      try {
        await cx.request(acp.methods.client.terminal.release, {
          sessionId: active.session.sessionId,
          terminalId,
        });
      } catch {
        // The terminal may already have been released.
      }
    };

    const onAbort = () => {
      cancelledBySignal = true;
      void killTerminal();
    };

    if (signal.aborted) {
      onAbort();
    } else {
      signal.addEventListener("abort", onAbort, { once: true });
    }

    try {
      const createResp = await cx.request(acp.methods.client.terminal.create, {
        sessionId: active.session.sessionId,
        command: "/bin/bash",
        args: ["-lc", wrappedCommand],
        cwd: active.session.cwd,
        env: [],
        outputByteLimit: 1_000_000,
      });
      terminalId = createResp.terminalId;
      void this.logRuntime(active.session.cwd, "info", "terminal created", {
        sessionId: active.session.sessionId,
        terminalId,
        command,
      });

      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status: "in_progress",
        content: [{ type: "terminal", terminalId }],
      });

      const exit = await cx.request(acp.methods.client.terminal.waitForExit, {
        sessionId: active.session.sessionId,
        terminalId,
      });

      const outputResp = await cx.request(acp.methods.client.terminal.output, {
        sessionId: active.session.sessionId,
        terminalId,
      });

      await releaseTerminal();
      void this.logRuntime(active.session.cwd, "info", "terminal finished", {
        sessionId: active.session.sessionId,
        terminalId,
        command,
        exitCode: exit.exitCode,
        signal: exit.signal,
      });

      const cancelled = cancelledBySignal || signal.aborted;
      const status = cancelled || exit.exitCode !== 0 ? "failed" : "completed";
      const durationMs = Date.now() - toolStart;
      const outputText =
        outputResp.output ||
        (status === "completed" ? "(no output)" : `exit code ${exit.exitCode ?? "unknown"}`);
      const outputForModel = outputText + `\n\n[Full output saved to ${logPath}]`;
      const displayText = `${outputText}\n\n⏱ ${formatMs(durationMs)}`;

      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status,
        content: [
          { type: "terminal", terminalId },
          {
            type: "content",
            content: { type: "text", text: displayText },
          },
        ],
        rawOutput: {
          output: outputResp.output,
          exitCode: exit.exitCode,
          signal: exit.signal,
          truncated: outputResp.truncated,
          cancelled,
          durationMs,
          fullOutputPath: logPath,
          commandScriptPath,
        },
      });

      return {
        toolCallId: call.id,
        toolName: "bash",
        output: { type: "text", value: outputForModel },
      };
    } catch (error) {
      await releaseTerminal();
      const message = error instanceof Error ? error.message : String(error);
      const durationMs = Date.now() - toolStart;
      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status: "failed",
        content: [
          {
            type: "content",
            content: {
              type: "text",
              text: `${message}\n\n⏱ ${formatMs(durationMs)}`,
            },
          },
        ],
        rawOutput: { error: message, durationMs },
      });
      return {
        toolCallId: call.id,
        toolName: "bash",
        output: { type: "text", value: message },
      };
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
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

  private async promptBlocksToText(
    blocks: acp.ContentBlock[],
  ): Promise<string> {
    const parts: string[] = [];

    for (const block of blocks) {
      switch (block.type) {
        case "text":
          parts.push(block.text);
          break;
        case "resource_link": {
          const text = await this.readResourceLink(block);
          parts.push(text);
          break;
        }
        case "resource": {
          const resource = block.resource;
          if ("text" in resource && typeof resource.text === "string") {
            parts.push(resource.text);
          } else if ("blob" in resource && typeof resource.blob === "string") {
            parts.push(
              `[Embedded binary resource ${resource.uri} (base64, ${resource.blob.length} chars)]`,
            );
          } else {
            parts.push(`[Embedded resource ${resource.uri}]`);
          }
          break;
        }
        case "image":
        case "audio":
          throw new Error(
            `${block.type} content is not supported by zen-agent yet`,
          );
        default:
          throw new Error(`Unsupported content block: ${(block as { type: string }).type}`);
      }
    }

    return parts.join("\n\n");
  }

  private async readResourceLink(block: {
    type: "resource_link";
    uri: string;
    name?: string;
    mimeType?: string | null;
  }): Promise<string> {
    if (!block.uri.startsWith("file://")) {
      return block.name ?? block.uri;
    }

    try {
      const path = fileURLToPath(block.uri);
      const content = await readFile(path, "utf8");
      return `File: ${path}\n${content}`;
    } catch {
      return block.name ?? block.uri;
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

  private prepareReplayEvents(events: acp.SessionUpdate[]): acp.SessionUpdate[] {
    // Only replay tool calls that have BOTH an initial `tool_call` event and a
    // final (completed/failed) `tool_call_update`. Calls interrupted mid-run
    // (e.g. hard-aborted) have no result worth showing, and replaying them as
    // "pending" would leave phantom entries in the thread. Zed also creates a
    // spurious "Tool call not found" failed entry for any update without a
    // matching initial event (`acp_thread.rs::update_tool_call`), which is
    // exactly what we want to avoid on reload.
    //
    // Note: a tool call that finished during a GRACEFUL cancel has both its
    // initial event and a final completed update, so it replays normally.
    const initialCallIds = new Set<string>();
    const finalizedCallIds = new Set<string>();
    for (const event of events) {
      if (event.sessionUpdate === "tool_call") {
        initialCallIds.add(event.toolCallId);
      } else if (
        event.sessionUpdate === "tool_call_update" &&
        (event.status === "completed" || event.status === "failed")
      ) {
        finalizedCallIds.add(event.toolCallId);
      }
    }

    const replayableCallIds = new Set(
      [...initialCallIds].filter((id) => finalizedCallIds.has(id)),
    );

    const result: acp.SessionUpdate[] = [];
    for (const event of events) {
      if (event.sessionUpdate === "tool_call") {
        if (replayableCallIds.has(event.toolCallId)) {
          result.push(event);
        }
        continue;
      }
      if (event.sessionUpdate === "tool_call_update") {
        if (
          !replayableCallIds.has(event.toolCallId) ||
          event.status === "in_progress"
        ) {
          // Drop transient in-progress updates. The final update carries the
          // full status and content, so replaying the intermediate one would
          // only cause extra work (and fail on stale terminal references).
          continue;
        }
        result.push(this.stripTerminalContent(event));
        continue;
      }
      result.push(event);
    }
    return result;
  }

  private stripTerminalContent(event: acp.SessionUpdate): acp.SessionUpdate {
    // Zed binds terminal cards to live terminals by ID; a stale terminalId
    // from a previous Zed run cannot be re-attached, so replay the text
    // output instead of the terminal reference.
    if (event.sessionUpdate !== "tool_call_update" || !event.content) {
      return event;
    }
    const content = event.content.filter((item) => item.type !== "terminal");
    if (content.length === event.content.length) {
      return event;
    }
    return { ...event, content } as acp.SessionUpdate;
  }

  private coalesceReplayEvents(events: acp.SessionUpdate[]): acp.SessionUpdate[] {
    const result: acp.SessionUpdate[] = [];

    for (const event of events) {
      const enriched = this.enrichReplayEvent(event);
      const last = result[result.length - 1];

      if (
        last &&
        (enriched.sessionUpdate === "agent_thought_chunk" ||
          enriched.sessionUpdate === "agent_message_chunk") &&
        last.sessionUpdate === enriched.sessionUpdate &&
        "messageId" in last &&
        "messageId" in enriched &&
        last.messageId === enriched.messageId &&
        last.content.type === "text" &&
        enriched.content.type === "text"
      ) {
        result[result.length - 1] = {
          ...last,
          content: {
            type: "text",
            text: last.content.text + enriched.content.text,
          },
        } as acp.SessionUpdate;
      } else {
        result.push(enriched);
      }
    }

    return result;
  }

  private enrichReplayEvent(event: acp.SessionUpdate): acp.SessionUpdate {
    if (event.sessionUpdate !== "tool_call_update") {
      return event;
    }
    const rawOutput = event.rawOutput as { output?: unknown } | undefined;
    if (!rawOutput || typeof rawOutput.output !== "string") {
      return event;
    }

    const hasTextContent = (event.content ?? []).some(
      (item) =>
        item.type === "content" &&
        item.content.type === "text" &&
        typeof item.content.text === "string",
    );
    if (hasTextContent) {
      return event;
    }

    return {
      ...event,
      content: [
        ...(event.content ?? []),
        {
          type: "content",
          content: { type: "text", text: rawOutput.output },
        },
      ],
    } as acp.SessionUpdate;
  }

  private async logRuntime(
    cwd: string,
    level: "debug" | "info" | "warn" | "error",
    message: string,
    details?: Record<string, unknown>,
  ): Promise<void> {
    try {
      await appendJsonLine(
        runtimeLogPath(cwd),
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
    await cx.notify(acp.methods.client.session.update, {
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands: AVAILABLE_COMMANDS,
      },
    });
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
  ): Promise<acp.StopReason> {
    if (command.name !== "prompt") {
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: `Unknown slash command: /${command.name}`,
        },
      });
      return "end_turn";
    }

    if (!command.argument) {
      await this.emit(active, cx, {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: this.buildSystemPrompt(active.session),
        },
      });
      return "end_turn";
    }

    active.session.config.systemPrompt = command.argument;
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

  private buildSystemPrompt(session: StoredSession): string {
    const environmentInfo = [
      `Working directory: ${session.cwd}`,
      `Current date/time: ${session.createdAt}`,
    ].join("\n");
    const base = session.config.systemPrompt || SYSTEM_PROMPT;
    return `${base}\n---\n${environmentInfo}`;
  }

  private getConfigOptions(session: StoredSession): acp.SessionConfigOption[] {
    return [
      {
        ...MODEL_CONFIG_OPTION,
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
