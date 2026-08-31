import * as acp from '@agentclientprotocol/sdk';
import type { LlmToolCall } from '../providers/llm-client.js';
import { isRecord } from '../util/is-record.js';
import { resolveMedia } from './media.js';
import type { ToolExecutionResult, ToolExecutorContext } from './execution.js';

/**
 * The read_media tool handler (see the ownership map in agent.ts): load a
 * local image/audio file into the conversation. The payload is returned via
 * attachedMedia and injected as parts of the synthetic user message following
 * the tool result — the OpenAI-compatible tool role only accepts text content.
 *
 * Only `import type` from tool-execution.js — the context/result types are
 * erased at runtime, so there is no import cycle with the dispatcher.
 */

export async function executeReadMedia(
  context: ToolExecutorContext,
  cx: acp.AgentContext,
  call: LlmToolCall,
): Promise<ToolExecutionResult> {
  void cx;
  const { session, mediaModalities, emit } = context;
  const rawPath = isRecord(call.input) ? call.input.path : undefined;
  const displayPath = typeof rawPath === 'string' ? rawPath : String(rawPath ?? '');

  await emit({
    sessionUpdate: 'tool_call',
    toolCallId: call.id,
    title: `read_media ${displayPath}`,
    kind: 'read',
    status: 'pending',
    rawInput: call.input,
  });

  try {
    if (typeof rawPath !== 'string' || rawPath.trim().length === 0) {
      throw new Error('read_media requires a non-empty string path');
    }
    const allowed: Array<'image' | 'audio'> = [];
    if (mediaModalities.image) allowed.push('image');
    if (mediaModalities.audio) allowed.push('audio');
    const media = await resolveMedia(session.cwd, rawPath, allowed);

    const summary = `loaded ${media.path} (${media.mimeType}, ${(media.decodedBytes / 1024).toFixed(1)} KB); media attached to the conversation`;
    await emit({
      sessionUpdate: 'tool_call_update',
      toolCallId: call.id,
      status: 'completed',
      content: [
        {
          type: 'content',
          content: { type: 'text', text: summary },
        },
      ],
      rawOutput: { path: media.path, mimeType: media.mimeType, bytes: media.decodedBytes },
    });
    return {
      toolCallId: call.id,
      toolName: 'read_media',
      output: { type: 'text', value: summary },
      attachedMedia: media,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
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
      toolName: 'read_media',
      output: { type: 'text', value: `read_media failed: ${message}` },
    };
  }
}
