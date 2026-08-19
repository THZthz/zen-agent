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
import { runLlmStep, SYSTEM_PROMPT, type LlmToolCall } from "./llm/deepseek.js";

interface ActiveSession {
  session: StoredSession;
  abortController: AbortController | null;
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

export class ZenAgent {
  private sessions = new Map<string, ActiveSession>();
  private clientCapabilities: acp.ClientCapabilities = {};

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
    this.sessions.set(session.sessionId, {
      session,
      abortController: null,
    });
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
    this.sessions.set(params.sessionId, {
      session,
      abortController: null,
    });
    void this.logRuntime(params.cwd, "info", "session loaded", {
      sessionId: session.sessionId,
    });

    for (const update of this.coalesceReplayEvents(session.events)) {
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
    this.sessions.set(params.sessionId, {
      session,
      abortController: null,
    });
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

  cancel(params: acp.CancelNotification): void {
    this.abortActiveSession(params.sessionId);
  }

  async prompt(
    params: acp.PromptRequest,
    cx: acp.AgentContext,
  ): Promise<acp.PromptResponse> {
    const active = this.sessions.get(params.sessionId);
    if (!active) {
      throw new Error(`Session ${params.sessionId} not found`);
    }

    this.abortActiveSession(params.sessionId);
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

      const stopReason = await this.runTurn(active, cx, controller.signal);
      return { stopReason };
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
      active.session.updatedAt = new Date().toISOString();
      await this.save(active).catch(() => {});
    }
  }

  private async runTurn(
    active: ActiveSession,
    cx: acp.AgentContext,
    signal: AbortSignal,
  ): Promise<acp.StopReason> {
    for (let step = 0; step < MAX_TURN_STEPS; step++) {
      const assistantMessageId = newMessageId();

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_request",
        timestamp: new Date().toISOString(),
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system: this.buildSystemPrompt(active.session),
        messages: active.session.llmMessages,
      });

      const llmResult = await runLlmStep({
        messages: active.session.llmMessages,
        signal,
        model: active.session.config.model,
        thinkingEffort: active.session.config.thinkingEffort,
        system: this.buildSystemPrompt(active.session),
        onTextDelta: async (delta) => {
          await this.emit(active, cx, {
            sessionUpdate: "agent_message_chunk",
            messageId: assistantMessageId,
            content: { type: "text", text: delta },
          });
        },
        onReasoningDelta: async (delta) => {
          await this.emit(active, cx, {
            sessionUpdate: "agent_thought_chunk",
            messageId: assistantMessageId,
            content: { type: "text", text: delta },
          });
        },
      });

      void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
        type: "llm_response",
        timestamp: new Date().toISOString(),
        text: llmResult.text,
        toolCalls: llmResult.toolCalls,
        finishReason: llmResult.finishReason,
      });

      if (llmResult.toolCalls.length === 0) {
        const content =
          llmResult.text.length > 0
            ? [{ type: "text" as const, text: llmResult.text }]
            : [];
        active.session.llmMessages.push({
          role: "assistant",
          content,
        });
        await this.save(active);
        return this.stopReasonFromFinish(llmResult.finishReason);
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
      for (const call of llmResult.toolCalls) {
        assistantParts.push({
          type: "tool-call",
          toolCallId: call.id,
          toolName: call.name,
          input: call.input,
        });
      }
      const toolResults: Array<{
        toolCallId: string;
        toolName: string;
        output: { type: "text"; value: string };
      }> = [];

      for (const call of llmResult.toolCalls) {
        const result = await this.executeLlmToolCall(active, cx, call, signal);
        toolResults.push(result);
        if (signal.aborted) {
          throw new Error("aborted");
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
    }

    return "max_turn_requests";
  }

  private shellQuote(value: string): string {
    return `'${value.replace(/'/g, `'\\''`)}'`;
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
      const outputText =
        outputResp.output ||
        (status === "completed" ? "(no output)" : `exit code ${exit.exitCode ?? "unknown"}`);
      const outputForModel = outputText + `\n\n[Full output saved to ${logPath}]`;

      await this.emit(active, cx, {
        sessionUpdate: "tool_call_update",
        toolCallId: call.id,
        status,
        content: [
          { type: "terminal", terminalId },
          {
            type: "content",
            content: { type: "text", text: outputText },
          },
        ],
        rawOutput: {
          output: outputResp.output,
          exitCode: exit.exitCode,
          signal: exit.signal,
          truncated: outputResp.truncated,
          cancelled,
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
    active?.abortController?.abort();
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
      `Current date/time: ${new Date().toISOString()}`,
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
