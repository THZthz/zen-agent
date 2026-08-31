import { describe, expect, it } from 'vitest';
import * as acp from '@agentclientprotocol/sdk';
import { DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT, truncateTerminalOutput } from './sandbox.js';
import { emitFailedToolResult } from './tool-execution.js';

describe('truncateTerminalOutput', () => {
  it('returns short text unchanged', () => {
    expect(truncateTerminalOutput('hello', 10)).toEqual({
      text: 'hello',
      truncated: false,
      originalBytes: 5,
      keptBytes: 5,
    });
  });

  it('keeps the tail when the text exceeds the byte budget', () => {
    const result = truncateTerminalOutput('abcdefghijklmnopqrstuvwxyz', 10);
    expect(result.truncated).toBe(true);
    expect(result.text).toBe('qrstuvwxyz');
    expect(result.originalBytes).toBe(26);
    expect(result.keptBytes).toBe(10);
  });

  it('keeps the end of the output (where errors and exit codes appear)', () => {
    const result = truncateTerminalOutput('head\n' + 'x'.repeat(200) + '\nexit code 1', 20);
    expect(result.text.endsWith('exit code 1')).toBe(true);
  });

  it('never splits a UTF-8 multi-byte sequence', () => {
    const input = 'a'.repeat(20) + '🎉'.repeat(10); // each 🎉 is 4 bytes
    const result = truncateTerminalOutput(input, 12);
    expect(result.truncated).toBe(true);
    expect(result.text).toBe('🎉🎉🎉');
    expect(Buffer.byteLength(result.text, 'utf8')).toBeLessThanOrEqual(12);
    expect(result.text).not.toContain('\uFFFD'); // no replacement chars
  });

  it('handles a cut landing inside a multi-byte sequence by keeping fewer bytes', () => {
    const input = 'a'.repeat(10) + '🎉'.repeat(10); // 10 + 40 = 50 bytes
    const result = truncateTerminalOutput(input, 15);
    // Byte 35 is the 7th emoji's lead; kept = 15 bytes exactly.
    // Cut at 35: emojis start at byte 10, each 4 bytes: emoji #7 = bytes 34-37.
    // Byte 35 is a continuation -> skip to 36 (also continuation) -> 37? no:
    //   #7: 34(lead),35,36,37(cont); #8 starts at 38. start=35 -> advance to 38.
    expect(Buffer.byteLength(result.text, 'utf8')).toBeLessThanOrEqual(15);
    expect(result.text).toBe('🎉🎉🎉'); // 12 bytes, the last 3 emoji
    expect(result.text).not.toContain('\uFFFD');
  });

  it('exposes the default byte limit constant', () => {
    expect(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT).toBe(50_000);
  });
});

describe('emitFailedToolResult', () => {
  const emitInto = (emitted: acp.SessionUpdate[]) => async (update: acp.SessionUpdate) => {
    emitted.push(update);
  };

  it('emits one failed tool_call/tool_call_update pair and returns the failed result', async () => {
    const emitted: acp.SessionUpdate[] = [];
    const result = await emitFailedToolResult(emitInto(emitted), {
      toolCallId: 'c1',
      toolName: 'bash',
      title: 'Invalid bash command',
      kind: 'execute',
      rawInput: { command: '' },
      message: 'bash tool requires a non-empty string command',
    });

    expect(emitted).toHaveLength(2);
    expect(emitted[0]).toMatchObject({
      sessionUpdate: 'tool_call',
      toolCallId: 'c1',
      title: 'Invalid bash command',
      kind: 'execute',
      status: 'failed',
      rawInput: { command: '' },
    });
    expect(emitted[1]).toMatchObject({
      sessionUpdate: 'tool_call_update',
      toolCallId: 'c1',
      status: 'failed',
      content: [
        {
          type: 'content',
          content: { type: 'text', text: 'bash tool requires a non-empty string command' },
        },
      ],
      rawOutput: { error: 'bash tool requires a non-empty string command' },
    });
    expect(result).toEqual({
      toolCallId: 'c1',
      toolName: 'bash',
      output: { type: 'text', value: 'bash tool requires a non-empty string command' },
    });
  });

  it('merges extraRawOutput next to error (e.g. the cancelled flag)', async () => {
    const emitted: acp.SessionUpdate[] = [];
    await emitFailedToolResult(emitInto(emitted), {
      toolCallId: 'c2',
      toolName: 'bash',
      title: 'Cancelled bash command',
      kind: 'execute',
      rawInput: {},
      message: 'bash tool cancelled before execution',
      extraRawOutput: { cancelled: true },
    });
    expect((emitted[1] as { rawOutput?: Record<string, unknown> }).rawOutput).toEqual({
      error: 'bash tool cancelled before execution',
      cancelled: true,
    });
  });
});
