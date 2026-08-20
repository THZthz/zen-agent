import * as acp from "@agentclientprotocol/sdk";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { LlmToolCall } from "./llm/deepseek.js";
import { terminalDirectory, type StoredSession } from "./storage.js";
import { formatMs } from "./turn-stats.js";

export interface ToolExecutionResult {
  toolCallId: string;
  toolName: string;
  output: { type: "text"; value: string };
}

export interface ToolExecutorContext {
  session: StoredSession;
  clientCapabilities: acp.ClientCapabilities;
  emit: (update: acp.SessionUpdate) => Promise<void>;
  logRuntime: (
    level: "debug" | "info" | "warn" | "error",
    message: string,
    details?: Record<string, unknown>,
  ) => Promise<void>;
}

export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/**
 * Default bubblewrap sandbox used for bash tool calls when
 * `ZEN_AGENT_SANDBOX=1`:
 *   --bind / /           rootfs behaves exactly as on the host
 *   --ro-bind /mnt /mnt  /mnt becomes read-only (reads OK, writes fail)
 *   --dev /dev           fresh devtmpfs (host /dev is unusable in a userns)
 *
 * The sandboxed process runs as the invoking uid in a new user+mount
 * namespace, so it cannot remount /mnt read-write. Override the entire
 * bwrap command with `ZEN_AGENT_SANDBOX_CMD` if a different policy is
 * needed (e.g. `--ro-bind / /` plus explicit writable binds).
 */
const DEFAULT_BASH_SANDBOX =
  "bwrap --die-with-parent --bind / / --ro-bind /mnt /mnt --dev /dev " +
  "--bind /dev/pts /dev/pts --tmpfs /dev/shm";

function bashSandboxPrefix(): string {
  const custom = process.env.ZEN_AGENT_SANDBOX_CMD;
  if (custom !== undefined && custom.trim() !== "") {
    return `${custom.trim()} `;
  }
  if (process.env.ZEN_AGENT_SANDBOX === "1") {
    return `${DEFAULT_BASH_SANDBOX} `;
  }
  return "";
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
 *
 * Because these terminals run in Zed (on the host), sandboxing the agent
 * process with bwrap does NOT constrain the bash tool. When
 * `ZEN_AGENT_SANDBOX=1`, every bash call is therefore wrapped in its own
 * bubblewrap invocation (see `bashSandboxPrefix`) that bind-mounts /mnt
 * read-only: the tool can read /mnt but every write to it fails with EROFS.
 */
export async function executeLlmToolCall(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
  signal: AbortSignal,
): Promise<ToolExecutionResult> {
  const { session, clientCapabilities, emit, logRuntime } = context;

  if (call.name !== "bash") {
    const message = `Unknown tool: ${call.name}`;
    await emit({
      sessionUpdate: "tool_call",
      toolCallId: call.id,
      title: `Unknown tool ${call.name}`,
      kind: "other",
      status: "failed",
      rawInput: call.input,
    });
    await emit({
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
    await emit({
      sessionUpdate: "tool_call",
      toolCallId: call.id,
      title: "Invalid bash command",
      kind: "execute",
      status: "failed",
      rawInput: call.input,
    });
    await emit({
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

  const terminalDir = terminalDirectory(session.cwd, session.sessionId);
  // The same timestamp pairs an input script with its output log, and keeps
  // multiple runs of the same tool call id distinct on disk.
  const timestamp = Date.now();
  const logPath = join(terminalDir, `output-${timestamp}-${call.id}.log`);
  const commandScriptPath = join(terminalDir, `input-${timestamp}-${call.id}.sh`);
  await mkdir(terminalDir, { recursive: true });
  await writeFile(commandScriptPath, command, "utf8");
  const scriptCommand = `${bashSandboxPrefix()}bash ${shellQuote(commandScriptPath)}`;
  const wrappedCommand = `script -q -e -c ${shellQuote(scriptCommand)} ${shellQuote(logPath)}`;
  const toolStart = Date.now();

  // Zed-specific display-only terminal id (see `_meta.terminal_info` below).
  //
  // Why this exists: after a Zed restart, the REAL terminal id returned by
  // `terminal/create` no longer exists, so replaying a `tool_call_update`
  // that references it makes Zed fail with "Terminal with id not found" and
  // drop the whole update. And Zed only renders execute-kind cards with an
  // expand/unfold toggle when they contain a `terminal` content item, so a
  // text-only replay card is not even expandable. Zed therefore supports
  // "display-only" terminals (Codex uses them too): if the `tool_call`
  // carries `_meta.terminal_info`, Zed re-creates that terminal on every
  // notification — including replay during `session/load` — and we stream
  // the output into it via `_meta.terminal_output`/`_meta.terminal_exit` on
  // the final update. The replayed card then resolves, auto-expands and
  // shows the output.
  //
  // The id is deterministic (`zen-<toolCallId>`) so replay can rewrite the
  // stale real terminal id to it (see `prepareReplayEvents` in replay.ts).
  // The REAL terminal (id returned by terminal/create) is only an execution
  // vehicle: it runs the command, we wait for exit and fetch its output,
  // then release it. It is never embedded in tool-call content.
  const displayTerminalId = `zen-${call.id}`;

  await emit({
    sessionUpdate: "tool_call",
    toolCallId: call.id,
    title: `$ ${command}`,
    kind: "execute",
    status: "pending",
    rawInput: { command },
    // Zed pre-registers a display-only terminal for this id on every
    // session/update notification, including the replayed ones during
    // session/load. Without it, a replayed `terminal` content item would
    // fail with "Terminal with id not found" and the whole update would be
    // dropped, leaving the tool call with no visible result after a restart.
    _meta: {
      terminal_info: {
        terminal_id: displayTerminalId,
        cwd: session.cwd,
      },
    },
  });

  if (!clientCapabilities.terminal) {
    const message =
      "Zed terminal support is required for the bash tool, but the client did not advertise terminal: true";
    await emit({
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
        sessionId: session.sessionId,
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
        sessionId: session.sessionId,
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
      sessionId: session.sessionId,
      command: "/bin/bash",
      args: ["-lc", wrappedCommand],
      cwd: session.cwd,
      env: [],
      outputByteLimit: 1_000_000,
    });
    terminalId = createResp.terminalId;
    void logRuntime("info", "terminal created", {
      sessionId: session.sessionId,
      terminalId,
      command,
    });

    await emit({
      sessionUpdate: "tool_call_update",
      toolCallId: call.id,
      status: "in_progress",
      content: [{ type: "terminal", terminalId }],
    });

    const exit = await cx.request(acp.methods.client.terminal.waitForExit, {
      sessionId: session.sessionId,
      terminalId,
    });

    const outputResp = await cx.request(acp.methods.client.terminal.output, {
      sessionId: session.sessionId,
      terminalId,
    });

    await releaseTerminal();
    void logRuntime("info", "terminal finished", {
      sessionId: session.sessionId,
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

    await emit({
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
      // Stream the recorded output into the display-only terminal so that
      // after a Zed restart the replayed tool call shows the output inside
      // the terminal card (replay re-registers the terminal from the
      // `tool_call` event's terminal_info above).
      _meta: {
        terminal_output: {
          terminal_id: displayTerminalId,
          data: outputText,
        },
        terminal_exit: {
          terminal_id: displayTerminalId,
          exit_code: exit.exitCode ?? null,
          signal: exit.signal ?? null,
        },
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
    await emit({
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
