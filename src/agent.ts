import * as acp from "@agentclientprotocol/sdk";
import { randomBytes, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createStoredSession,
  DEFAULT_MODEL,
  DEFAULT_THINKING_EFFORT,
  deleteStoredSession,
  findSessionCwd,
  listStoredSessions,
  readStoredSession,
  clientLogPath,
  sessionLlmLogPath,
  terminalDirectory,
  writeSession,
  type ModelId,
  type NamedUserMessage,
  type StoredSession,
  type ThinkingEffort,
} from "./storage.js";
import { appendJsonLine, makeLogEntry } from "./logger.js";
import { promptBlocksToText } from "./prompt-content.js";
import { executeLlmToolCall } from "./tool-execution.js";
import {
  costYuan,
  getContextWindowTokens,
  getModelPricing,
  runLlmStep,
  type LlmToolCall,
  type LlmUsage,
} from "./llm/deepseek.js";
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
    // Freeze the environment snapshot into the persisted conversation at
    // session creation. It sits right after the system prompt, so it must
    // stay byte-identical for DeepSeek's prefix cache to keep hitting
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
      configOptions: this.getConfigOptions(session),
    };
  }

  async loadSession(
    params: acp.LoadSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.LoadSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    this.abortActiveSession(params.sessionId);
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
      configOptions: this.getConfigOptions(session),
    };
  }

  async resumeSession(
    params: acp.ResumeSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.ResumeSessionResponse> {
    const session = await readStoredSession(params.cwd, params.sessionId);
    this.abortActiveSession(params.sessionId);
    await this.prepareResumedSession(session);
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
      const userText = await promptBlocksToText(params.prompt);
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

      const userMessage: NamedUserMessage = {
        role: "user",
        content: userText,
        name: await getUserMessageName(active.session.cwd),
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

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_request",
        timestamp: new Date().toISOString(),
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system: buildSystemPrompt(active.session),
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
        system: buildSystemPrompt(active.session),
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
        this.logStepStats(
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
    // Keep per-turn stats so they survive process restarts (resume/load).
    active.session.turnStats.push(turn);
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
   *    stay byte-identical, so DeepSeek's context cache keeps hitting.
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
  ): Promise<{
    toolCallId: string;
    toolName: string;
    output: { type: "text"; value: string };
  }> {
    return executeLlmToolCall(
      {
        session: active.session,
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
  private logStepStats(
    active: ActiveSession,
    step: number,
    usage: LlmUsage,
    finishReason: string,
    toolCallCount: number,
  ): void {
    void this.logRuntime(active.session.cwd, "info", "llm step stats", {
      sessionId: active.session.sessionId,
      step: step + 1,
      model: active.session.config.model,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      cacheReadTokens: usage.cacheReadTokens,
      cacheMissTokens: usage.cacheMissTokens,
      cacheHitPercent: cacheHitPercent(usage),
      reasoningTokens: usage.reasoningTokens,
      costYuan: costYuan(usage, getModelPricing(active.session.config.model)),
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
          text: buildSystemPrompt(active.session),
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
