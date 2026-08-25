import { describe, expect, it, vi } from 'vitest';
import { StreamThrottle } from './stream-throttle.js';

describe('StreamThrottle', () => {
  it('emits queued chunks in order and drains', async () => {
    const emitted: Array<{ kind: string; text: string }> = [];
    const t = new StreamThrottle(async (kind, text) => {
      emitted.push({ kind, text });
    }, 1);
    t.push('thought', 'abc');
    t.push('message', 'def');
    await t.drain();
    expect(emitted).toEqual([
      { kind: 'thought', text: 'abc' },
      { kind: 'message', text: 'def' },
    ]);
  });

  it('splits chunks larger than maxCharsPerTick', async () => {
    const emitted: string[] = [];
    const t = new StreamThrottle(
      async (_kind, text) => {
        emitted.push(text);
      },
      1,
      4,
    );
    t.push('message', 'abcdefgh');
    await t.drain();
    expect(emitted.join('')).toBe('abcdefgh');
    expect(emitted.length).toBe(2);
  });

  it('drain rejects when emit fails instead of crashing or hanging', async () => {
    const emit = vi
      .fn<(kind: 'message' | 'thought', text: string) => Promise<void>>()
      .mockRejectedValueOnce(new Error('connection closed'));
    const t = new StreamThrottle(emit, 1);

    t.push('message', 'hello world');
    await expect(t.drain()).rejects.toThrow('connection closed');

    // The failure is sticky: later pushes/drains fail too rather than hang.
    t.push('message', 'more');
    await expect(t.drain()).rejects.toThrow('connection closed');
    // The failing chunk was the only one ever emitted; schedule() stays
    // inert after a failure (no new ticks).
    expect(emit).toHaveBeenCalledTimes(1);
  });

  it('recovers scheduling for chunks pushed after a successful tick', async () => {
    const emitted: string[] = [];
    const t = new StreamThrottle(
      async (_kind, text) => {
        emitted.push(text);
      },
      1,
      3,
    );
    t.push('message', 'aaa');
    await t.drain();
    t.push('message', 'bbb');
    await t.drain();
    expect(emitted.join('')).toBe('aaabbb');
  });
});
