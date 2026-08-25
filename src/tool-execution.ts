import * as acp from "@agentclientprotocol/sdk";
import { existsSync, realpathSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { LlmToolCall } from "./deepseek.js";
import { resolveMedia, type ResolvedMedia } from "./media.js";
import { terminalDirectory, type StoredSession } from "./storage.js";
import { envPositiveInt } from "./env.js";
import { formatMs } from "./turn-stats.js";

export interface ToolExecutionResult {
  toolCallId: string;
  toolName: string;
  output: { type: "text"; value: string };
  /**
   * Set by read_media: the media to inject as parts of the synthetic user
   * message that follows the tool result (the tool role only allows text).
   */
  attachedMedia?: ResolvedMedia;
}

export interface ToolExecutorContext {
  session: StoredSession;
  /**
   * Input modalities of the active model. Gates read_media: the tool is
   * only offered to (and executed for) models that accept image/audio.
   */
  mediaModalities: { image: boolean; audio: boolean };
  /**
   * Whether bash tool calls in this session run inside their own bwrap
   * sandbox. The agent computes this as `session.config.sandbox || env
   * ZEN_AGENT_SANDBOX=1`, so the environment policy always applies.
   */
  sandbox: boolean;
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

/** Default cap on terminal output text sent to the model, in bytes. */
export const DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 50_000;

export interface TruncatedTerminalOutput {
  /** The kept text (the tail when truncated, otherwise the original). */
  text: string;
  truncated: boolean;
  /** UTF-8 byte length of the original text. */
  originalBytes: number;
  /** UTF-8 byte length of the kept text. */
  keptBytes: number;
}

/**
 * Keep only the tail of `text` so it fits within `maxBytes` UTF-8 bytes.
 * The cut is adjusted forward to a UTF-8 lead byte so the kept text never
 * contains a split multi-byte sequence (and never exceeds `maxBytes`).
 *
 * Used for the model-visible tool result only: the full output stays on
 * disk at the log path (and in the terminal card), so the model can read
 * more with bash whenever it needs to.
 */
export function truncateTerminalOutput(
  text: string,
  maxBytes: number,
): TruncatedTerminalOutput {
  const bytes = Buffer.from(text, "utf8");
  const originalBytes = bytes.length;
  if (originalBytes <= maxBytes) {
    return { text, truncated: false, originalBytes, keptBytes: originalBytes };
  }
  let start = bytes.length - maxBytes;
  // Continuation bytes are 0b10xxxxxx; skip forward to the next lead byte
  // so we never split a multi-byte character.
  while (start < bytes.length && (bytes[start]! & 0b1100_0000) === 0b1000_0000) {
    start++;
  }
  const kept = bytes.subarray(start).toString("utf8");
  return {
    text: kept,
    truncated: true,
    originalBytes,
    keptBytes: Buffer.byteLength(kept, "utf8"),
  };
}

/**
 * Model-visible terminal output byte budget. Override with
 * `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` (bytes; must be > 0).
 */
function terminalOutputByteLimit(): number {
  return envPositiveInt(
    "ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT",
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
  );
}

/**
 * Commands the bash tool may not run; their real binaries are shadowed
 * inside the bwrap namespace by `sandboxBlockShim`, which refuses to run
 * and points the agent at the substitute. The host binaries are untouched,
 * so every other script on the machine keeps using the real `rm`, `grep`
 * and `find`.
 */
const BLOCKED_COMMANDS: Record<string, string> = {
  rm: "trash",
  grep: "rg",
  find: "fdfind",
};

/** Standard locations of the blocked binaries (checked at runtime). */
const BLOCKED_BINARY_DIRS = ["/usr/bin", "/bin"];

/**
 * Path to the shim mounted over the blocked binaries. Override with
 * `ZEN_AGENT_SANDBOX_BLOCK_SHIM`; the default resolves relative to this
 * file, so it works both from `src/` (tsx) and `dist/` (compiled).
 */
function sandboxBlockShimPath(): string {
  const override = process.env.ZEN_AGENT_SANDBOX_BLOCK_SHIM;
  if (override !== undefined && override.trim() !== "") {
    return override.trim();
  }
  return fileURLToPath(
    new URL("../bin/zen-agent-sandbox-block.sh", import.meta.url),
  );
}

/**
 * `--ro-bind` arguments that replace each blocked binary with the shim
 * inside the sandbox. `/bin` is a symlink to `/usr/bin` on many distros,
 * so destinations are deduplicated by their resolved path. Only existing
 * binaries are shadowed (bwrap refuses to bind onto a missing file).
 */
function blockedBinaryBinds(): string[] {
  const shim = sandboxBlockShimPath();
  const seen = new Set<string>();
  const binds: string[] = [];
  for (const command of Object.keys(BLOCKED_COMMANDS)) {
    for (const dir of BLOCKED_BINARY_DIRS) {
      const dest = join(dir, command);
      if (!existsSync(dest)) continue;
      const resolved = realpathSync(dest);
      if (seen.has(resolved)) continue;
      seen.add(resolved);
      binds.push(`--ro-bind ${shim} ${dest}`);
    }
  }
  return binds;
}

/**
 * Default bubblewrap sandbox used for bash tool calls when
 * `ZEN_AGENT_SANDBOX=1`:
 *   --bind / /           rootfs behaves exactly as on the host
 *   --ro-bind /mnt /mnt  /mnt becomes read-only (reads OK, writes fail)
 *   --dev /dev           fresh devtmpfs (host /dev is unusable in a userns)
 *   --ro-bind shim ...   `rm`, `grep` and `find` are replaced by a shim
 *                        that refuses to run (use trash/rg/fdfind instead)
 *
 * The sandboxed process runs as the invoking uid in a new user+mount
 * namespace, so it cannot remount /mnt read-write or escape the namespace.
 * Shadowing the binaries only affects processes inside the namespace:
 * scripts on the host keep using the real `rm`/`grep`/`find`. Override the
 * entire bwrap command with `ZEN_AGENT_SANDBOX_CMD` if a different policy
 * is needed (e.g. `--ro-bind / /` plus explicit writable binds).
 */
function defaultBashSandbox(): string {
  return [
    "bwrap --die-with-parent --bind / / --ro-bind /mnt /mnt --dev /dev",
    "--bind /dev/pts /dev/pts --tmpfs /dev/shm",
    ...blockedBinaryBinds(),
  ].join(" ");
}

function bashSandboxPrefix(enabled: boolean): string {
  if (!enabled) {
    return "";
  }
  const custom = process.env.ZEN_AGENT_SANDBOX_CMD;
  if (custom !== undefined && custom.trim() !== "") {
    return `${custom.trim()} `;
  }
  return `${defaultBashSandbox()} `;
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
 * process with bwrap does NOT constrain the bash tool. Every bash call is
 * therefore wrapped in its own bubblewrap invocation (see
 * `bashSandboxPrefix`) whenever sandboxing is enabled — either per session
 * via the `/sandbox` slash command or globally via `ZEN_AGENT_SANDBOX=1`.
 * The sandbox bind-mounts /mnt read-only: the tool can read /mnt but every
 * write to it fails with EROFS. It also shadows `rm`, `grep` and `find`
 * with refusing shims (see `blockedBinaryBinds`), so the agent can only
 * use `trash`, `rg` and `fdfind` for those operations; host scripts are
 * unaffected because the shadowing lives in the bwrap mount namespace.
 */
export async function executeLlmToolCall(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
  signal: AbortSignal,
): Promise<ToolExecutionResult> {
  const { session, sandbox, clientCapabilities, emit, logRuntime } = context;

  // A streamed tool call whose JSON arguments never parsed cannot be
  // dispatched; report that clearly instead of a confusing per-tool
  // "requires a string" error (the client used to assume bash).
  const malformedArguments = (
    call.input as { malformed_arguments?: unknown } | null
  )?.malformed_arguments;
  if (typeof malformedArguments === "string") {
    const message = `Tool ${call.name} produced malformed JSON arguments: ${malformedArguments.slice(0, 200)}`;
    await emit({
      sessionUpdate: "tool_call",
      toolCallId: call.id,
      title: `Malformed arguments for ${call.name}`,
      kind: "other",
      status: "failed",
      rawInput: call.input,
    });
    await emit({
      sessionUpdate: "tool_call_update",
      toolCallId: call.id,
      status: "failed",
      content: [{ type: "content", content: { type: "text", text: message } }],
      rawOutput: { error: message },
    });
    void logRuntime("warn", "tool call had malformed JSON arguments", {
      sessionId: session.sessionId,
      toolCallId: call.id,
      toolName: call.name,
    });
    return {
      toolCallId: call.id,
      toolName: call.name,
      output: { type: "text", value: message },
    };
  }

  if (call.name === "read_media") {
    return executeReadMedia(context, cx, call);
  }

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
  const scriptCommand = `${bashSandboxPrefix(sandbox)}bash ${shellQuote(commandScriptPath)}`;
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
    const modelOutput = truncateTerminalOutput(outputText, terminalOutputByteLimit());
    const outputForModel = modelOutput.truncated
      ? `${modelOutput.text}\n\n[Terminal output truncated: showing the last ${modelOutput.keptBytes} of ${modelOutput.originalBytes} bytes; full output saved to ${logPath}]`
      : `${outputText}\n\n[Full output saved to ${logPath}]`;
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
        truncatedForModel: modelOutput.truncated,
        outputBytes: modelOutput.originalBytes,
        outputKeptBytes: modelOutput.keptBytes,
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

/**
 * The read_media tool: load a local image/audio file into the conversation.
 * The payload is returned via attachedMedia and injected as parts of the
 * synthetic user message following the tool result - the OpenAI-compatible
 * tool role only accepts text content.
 */
async function executeReadMedia(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
): Promise<ToolExecutionResult> {
  void cx;
  const { session, mediaModalities, emit } = context;
  const rawPath = (call.input as { path?: unknown }).path;
  const displayPath = typeof rawPath === "string" ? rawPath : String(rawPath ?? "");

  await emit({
    sessionUpdate: "tool_call",
    toolCallId: call.id,
    title: `read_media ${displayPath}`,
    kind: "read",
    status: "pending",
    rawInput: call.input,
  });

  try {
    if (typeof rawPath !== "string" || rawPath.trim().length === 0) {
      throw new Error("read_media requires a non-empty string path");
    }
    const allowed: Array<"image" | "audio"> = [];
    if (mediaModalities.image) allowed.push("image");
    if (mediaModalities.audio) allowed.push("audio");
    const media = await resolveMedia(session.cwd, rawPath, allowed);

    const summary = `loaded ${media.path} (${media.mimeType}, ${(media.decodedBytes / 1024).toFixed(1)} KB); media attached below`;
    await emit({
      sessionUpdate: "tool_call_update",
      toolCallId: call.id,
      status: "completed",
      content: [
        {
          type: "content",
          content: { type: "text", text: summary },
        },
      ],
      rawOutput: { path: media.path, mimeType: media.mimeType, bytes: media.decodedBytes },
    });
    return {
      toolCallId: call.id,
      toolName: "read_media",
      output: { type: "text", value: summary },
      attachedMedia: media,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
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
      toolName: "read_media",
      output: { type: "text", value: `read_media failed: ${message}` },
    };
  }
}
