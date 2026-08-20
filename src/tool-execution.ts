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
  const logPath = join(terminalDir, `terminal-${call.id}.log`);
  const commandScriptPath = join(terminalDir, `terminal-${call.id}.sh`);
  await mkdir(terminalDir, { recursive: true });
  await writeFile(commandScriptPath, command, "utf8");
  const scriptCommand = `${bashSandboxPrefix()}bash ${shellQuote(commandScriptPath)}`;
  const wrappedCommand = `script -q -e -c ${shellQuote(scriptCommand)} ${shellQuote(logPath)}`;
  const toolStart = Date.now();

  await emit({
    sessionUpdate: "tool_call",
    toolCallId: call.id,
    title: `$ ${command}`,
    kind: "execute",
    status: "pending",
    rawInput: { command },
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
