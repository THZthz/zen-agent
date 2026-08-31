import * as acp from '@agentclientprotocol/sdk';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { LlmToolCall } from './llm-client.js';
import { terminalDirectory } from './storage.js';
import { formatMs } from './turn-stats.js';
import {
  bashSandboxPrefix,
  shellQuote,
  terminalOutputByteLimit,
  truncateTerminalOutput,
} from './sandbox.js';
import type { ToolExecutionResult, ToolExecutorContext } from './tool-execution.js';

/**
 * Bash tool execution (see the ownership map in agent.ts): terminal
 * create/wait/output/kill/release against the ACP client, the abort
 * listener that kills the terminal on a HARD abort, and the result
 * formatting (model-visible text, display text, `_meta` terminal streaming
 * for replay). The dispatch front door and its refusal paths live in
 * tool-execution.ts.
 *
 * Only `import type` from tool-execution.js — the context/result types are
 * erased at runtime, so there is no import cycle with the dispatcher.
 */

export async function executeBashToolCall(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
  command: string,
  signal: AbortSignal,
): Promise<ToolExecutionResult> {
  const { session, clientCapabilities, emit, logRuntime } = context;

  const terminalDir = terminalDirectory(session.cwd, session.sessionId);
  // The same timestamp pairs an input script with its output log, and keeps
  // multiple runs of the same tool call id distinct on disk.
  const timestamp = Date.now();
  const logPath = join(terminalDir, `output-${timestamp}-${call.id}.log`);
  const commandScriptPath = join(terminalDir, `input-${timestamp}-${call.id}.sh`);
  await mkdir(terminalDir, { recursive: true });
  await writeFile(commandScriptPath, command, 'utf8');
  const scriptCommand = `${bashSandboxPrefix(context.sandbox)}bash ${shellQuote(commandScriptPath)}`;
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
    sessionUpdate: 'tool_call',
    toolCallId: call.id,
    title: `$ ${command}`,
    kind: 'execute',
    status: 'pending',
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
      'Zed terminal support is required for the bash tool, but the client did not advertise terminal: true';
    await emit({
      sessionUpdate: 'tool_call_update',
      toolCallId: call.id,
      status: 'failed',
      content: [
        {
          type: 'content',
          content: { type: 'text', text: message },
        },
      ],
      rawOutput: { error: message },
    });
    return {
      toolCallId: call.id,
      toolName: 'bash',
      output: { type: 'text', value: message },
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
    signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    const createResp = await cx.request(acp.methods.client.terminal.create, {
      sessionId: session.sessionId,
      command: '/bin/bash',
      args: ['-lc', wrappedCommand],
      cwd: session.cwd,
      env: [],
      outputByteLimit: 1_000_000,
    });
    terminalId = createResp.terminalId;
    if (cancelledBySignal || signal.aborted) {
      await killTerminal();
      throw signal.reason instanceof Error
        ? signal.reason
        : new Error('bash tool cancelled during terminal creation');
    }
    void logRuntime('info', 'terminal created', {
      sessionId: session.sessionId,
      terminalId,
      command,
    });

    await emit({
      sessionUpdate: 'tool_call_update',
      toolCallId: call.id,
      status: 'in_progress',
      content: [{ type: 'terminal', terminalId }],
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
    void logRuntime('info', 'terminal finished', {
      sessionId: session.sessionId,
      terminalId,
      command,
      exitCode: exit.exitCode,
      signal: exit.signal,
    });

    const cancelled = cancelledBySignal || signal.aborted;
    const status = cancelled || exit.exitCode !== 0 ? 'failed' : 'completed';
    const durationMs = Date.now() - toolStart;
    const outputText =
      outputResp.output ||
      (status === 'completed' ? '(no output)' : `exit code ${exit.exitCode ?? 'unknown'}`);
    const modelOutput = truncateTerminalOutput(outputText, terminalOutputByteLimit());
    const outputForModel = modelOutput.truncated
      ? `${modelOutput.text}[Terminal output truncated: showing the last ${modelOutput.keptBytes} of ${modelOutput.originalBytes} bytes; full output saved to ${logPath}]`
      : `${outputText}`;
    const displayText = `${outputText}\n\n${formatMs(durationMs)}`;

    await emit({
      sessionUpdate: 'tool_call_update',
      toolCallId: call.id,
      status,
      content: [
        { type: 'terminal', terminalId },
        {
          type: 'content',
          content: { type: 'text', text: displayText },
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
      toolName: 'bash',
      output: { type: 'text', value: outputForModel },
    };
  } catch (error) {
    await releaseTerminal();
    const message = error instanceof Error ? error.message : String(error);
    const durationMs = Date.now() - toolStart;
    await emit({
      sessionUpdate: 'tool_call_update',
      toolCallId: call.id,
      status: 'failed',
      content: [
        {
          type: 'content',
          content: {
            type: 'text',
            text: `${message}\n\n${formatMs(durationMs)}`,
          },
        },
      ],
      rawOutput: { error: message, durationMs },
    });
    return {
      toolCallId: call.id,
      toolName: 'bash',
      output: { type: 'text', value: message },
    };
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}
