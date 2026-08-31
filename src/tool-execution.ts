import * as acp from '@agentclientprotocol/sdk';
import type { LlmToolCall } from './deepseek.js';
import { isRecord } from './is-record.js';
import type { ResolvedMedia } from './media.js';
import type { StoredSession } from './storage.js';
import { executeBashToolCall } from './tool-bash.js';
import { executeReadMedia } from './tool-read-media.js';

export interface ToolExecutionResult {
  toolCallId: string;
  toolName: string;
  output: { type: 'text'; value: string };
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
  /**
   * Whether the session may use tools at all (set by the `/tools` slash
   * command). When false, executeLlmToolCall refuses every call with a
   * failed result and never touches the terminal or media stack.
   */
  toolsEnabled: boolean;
  clientCapabilities: acp.ClientCapabilities;
  emit: (update: acp.SessionUpdate) => Promise<void>;
  logRuntime: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    details?: Record<string, unknown>,
  ) => Promise<void>;
}

/**
 * One shape and one log line for every refusal before a tool runs
 * (tools-disabled, malformed arguments, unknown tool, invalid command,
 * cancelled-before-run — see the ownership map in agent.ts): emit the failed
 * `tool_call` + `tool_call_update` pair, then return the failed result so
 * the LLM message history stays paired and the turn can recover.
 */
export async function emitFailedToolResult(
  emit: (update: acp.SessionUpdate) => Promise<void>,
  params: {
    toolCallId: string;
    toolName: string;
    title: string;
    kind: acp.ToolKind;
    rawInput: unknown;
    message: string;
    /** Extra fields merged into the update's rawOutput next to `error`. */
    extraRawOutput?: Record<string, unknown>;
  },
): Promise<ToolExecutionResult> {
  await emit({
    sessionUpdate: 'tool_call',
    toolCallId: params.toolCallId,
    title: params.title,
    kind: params.kind,
    status: 'failed',
    rawInput: params.rawInput,
  });
  await emit({
    sessionUpdate: 'tool_call_update',
    toolCallId: params.toolCallId,
    status: 'failed',
    content: [
      {
        type: 'content',
        content: { type: 'text', text: params.message },
      },
    ],
    rawOutput: { error: params.message, ...params.extraRawOutput },
  });
  return {
    toolCallId: params.toolCallId,
    toolName: params.toolName,
    output: { type: 'text', value: params.message },
  };
}

/**
 * Dispatch front door for LLM tool calls: validate and route to the bash or
 * read_media handler (tool-bash.ts / tool-read-media.ts), refusing calls that
 * must not run with one shared failed-result shape.
 */
export async function executeLlmToolCall(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
  signal: AbortSignal,
): Promise<ToolExecutionResult> {
  const { session, emit, logRuntime } = context;

  // Defensive guard: with `/tools off` no tool schema is offered to the
  // model, but a resumed session's stale history (or a model that calls a
  // tool anyway) must never reach the terminal or media stack. Refuse with
  // a failed result so the LLM message history stays paired and the turn
  // can recover instead of erroring.
  if (context.toolsEnabled === false) {
    void logRuntime('warn', 'refused tool call: tools disabled for session', {
      sessionId: session.sessionId,
      toolCallId: call.id,
      toolName: call.name,
    });
    return emitFailedToolResult(emit, {
      toolCallId: call.id,
      toolName: call.name,
      title: `Tools disabled (${call.name})`,
      kind: 'other',
      rawInput: call.input,
      message: `Tool ${call.name} is disabled: all tools are turned off for this session (/tools on to re-enable).`,
    });
  }

  // A streamed tool call whose JSON arguments never parsed cannot be
  // dispatched; report that clearly instead of a confusing per-tool
  // "requires a string" error (the client used to assume bash).
  const malformedArguments = (call.input as { malformed_arguments?: unknown } | null)
    ?.malformed_arguments;
  if (typeof malformedArguments === 'string') {
    void logRuntime('warn', 'tool call had malformed JSON arguments', {
      sessionId: session.sessionId,
      toolCallId: call.id,
      toolName: call.name,
    });
    return emitFailedToolResult(emit, {
      toolCallId: call.id,
      toolName: call.name,
      title: `Malformed arguments for ${call.name}`,
      kind: 'other',
      rawInput: call.input,
      message: `Tool ${call.name} produced malformed JSON arguments: ${malformedArguments.slice(0, 200)}`,
    });
  }

  if (call.name === 'read_media') {
    return executeReadMedia(context, cx, call);
  }

  if (call.name !== 'bash') {
    return emitFailedToolResult(emit, {
      toolCallId: call.id,
      toolName: call.name,
      title: `Unknown tool ${call.name}`,
      kind: 'other',
      rawInput: call.input,
      message: `Unknown tool: ${call.name}`,
    });
  }

  const command = isRecord(call.input) ? call.input.command : undefined;
  if (typeof command !== 'string' || command.trim().length === 0) {
    return emitFailedToolResult(emit, {
      toolCallId: call.id,
      toolName: 'bash',
      title: 'Invalid bash command',
      kind: 'execute',
      rawInput: call.input,
      message: 'bash tool requires a non-empty string command',
    });
  }

  if (signal.aborted) {
    return emitFailedToolResult(emit, {
      toolCallId: call.id,
      toolName: 'bash',
      title: 'Cancelled bash command',
      kind: 'execute',
      rawInput: call.input,
      message: 'bash tool cancelled before execution',
      extraRawOutput: { cancelled: true },
    });
  }

  return executeBashToolCall(context, cx, call, command, signal);
}
