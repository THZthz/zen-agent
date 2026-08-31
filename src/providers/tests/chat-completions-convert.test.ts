import { describe, expect, it } from 'vitest';
import type { Model as PiModel } from '@earendil-works/pi-ai';
import {
  audioFormat,
  mapFinishReason,
  patchPayload,
  reasoningReplayField,
  storedReasoningSignature,
  toLlmUsage,
  toPiTools,
  toPiUserContent,
  toolResultText,
  type StoredReasoningPart,
} from '../convert.js';

const model: PiModel<'openai-completions'> = {
  id: 'test-model',
  name: 'test-model',
  api: 'openai-completions',
  provider: 'test',
  baseUrl: 'https://api.test',
  reasoning: true,
  input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1_000_000,
  maxTokens: 384_000,
};

describe('chat-completions-convert', () => {
  it('mapFinishReason maps Pi stop reasons onto the persisted vocabulary', () => {
    expect(mapFinishReason('toolUse')).toBe('tool-calls');
    expect(mapFinishReason('stop')).toBe('stop');
    expect(mapFinishReason('length')).toBe('length');
    expect(mapFinishReason('mystery')).toBe('other');
  });

  it('toPiTools defaults to the bash schema and drops the empty list', () => {
    expect(toPiTools([])).toBeUndefined();
    const bash = toPiTools(undefined);
    expect(bash).toHaveLength(1);
    expect(bash?.[0]?.name).toBe('bash');
    const custom = toPiTools([
      { function: { name: 'echo', description: 'Echo', parameters: { type: 'object' } } },
    ]);
    expect(custom).toEqual([{ name: 'echo', description: 'Echo', parameters: { type: 'object' } }]);
  });

  it('toPiUserContent keeps text and transports audio as an image-shaped block', () => {
    expect(toPiUserContent('hi')).toBe('hi');
    expect(
      toPiUserContent([
        { type: 'text', text: 'see' },
        { type: 'audio', mimeType: 'audio/wav', data: 'aGk=' },
      ]),
    ).toEqual([
      { type: 'text', text: 'see' },
      { type: 'image', mimeType: 'audio/wav', data: 'aGk=' },
    ]);
  });

  it('toolResultText unwraps strings, {value} objects and JSON-falls-back', () => {
    expect(toolResultText('plain')).toBe('plain');
    expect(toolResultText({ type: 'text', value: 'wrapped' })).toBe('wrapped');
    expect(toolResultText({ out: 1 })).toBe('{"out":1}');
  });

  it('storedReasoningSignature reads forward-compatible metadata keys', () => {
    expect(storedReasoningSignature({ type: 'reasoning', text: 't' })).toBeUndefined();
    expect(
      storedReasoningSignature({
        type: 'reasoning',
        text: 't',
        reasoningSignature: 'sig',
      } as StoredReasoningPart),
    ).toBe('sig');
    expect(
      storedReasoningSignature({
        type: 'reasoning',
        text: 't',
        thinkingSignature: 'legacy',
      } as StoredReasoningPart),
    ).toBe('legacy');
    expect(
      storedReasoningSignature({
        type: 'reasoning',
        text: 't',
        signature: 7,
      } as StoredReasoningPart),
    ).toBeUndefined();
  });

  it('reasoningReplayField picks the field the provider expects reasoning replay in', () => {
    expect(
      reasoningReplayField({
        ...model,
        compat: { requiresReasoningContentOnAssistantMessages: true },
      }),
    ).toBe('reasoning_content');
    expect(reasoningReplayField({ ...model, compat: { thinkingFormat: 'openrouter' } })).toBe(
      'reasoning',
    );
    expect(reasoningReplayField(model)).toBeUndefined();
  });

  it('toLlmUsage folds cache read/write into Zen historical input', () => {
    const usage = toLlmUsage(
      {
        input: 100,
        output: 20,
        cacheRead: 50,
        cacheWrite: 10,
        totalTokens: 130,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      { llmMs: 1000, thinkingMs: 200, answeringMs: 800 },
    );
    expect(usage).not.toBeNull();
    expect(usage?.inputTokens).toBe(160); // input + cacheRead + cacheWrite
    expect(usage?.cacheReadTokens).toBe(50);
    expect(usage?.cacheMissTokens).toBe(110); // input + cacheWrite
    expect(usage?.outputTokens).toBe(20);
    expect(usage?.totalTokens).toBe(130);
  });

  it('patchPayload restores named-user messages and OpenAI input_audio parts', () => {
    const payload = {
      messages: [
        { role: 'system', content: 'sys' },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'hear' },
            { type: 'image_url', image_url: { url: 'data:audio/wav;base64,aGk=' } },
          ],
        },
      ],
      tools: [{ name: 'bash' }],
    };
    const patched = patchPayload(
      payload,
      [
        {
          role: 'user',
          name: 'environment',
          content: [
            { type: 'text', text: 'hear' },
            { type: 'audio', mimeType: 'audio/wav', data: 'aGk=' },
          ],
        },
      ],
      false,
    ) as typeof payload;

    const user = patched.messages[1] as Record<string, unknown>;
    expect(user.name).toBe('environment');
    expect(user.content).toEqual([
      { type: 'text', text: 'hear' },
      { type: 'input_audio', input_audio: { data: 'aGk=', format: 'wav' } },
    ]);
  });

  it('audioFormat recognizes the supported wire formats only', () => {
    expect(audioFormat('audio/wav')).toBe('wav');
    expect(audioFormat('AUDIO/MP3')).toBe('mp3');
    expect(audioFormat('audio/ogg')).toBeNull();
  });
});
