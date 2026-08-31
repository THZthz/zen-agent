import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { waitForChatRateLimit } from '../rate-limit.js';

const originalEnv = { ...process.env };

describe('waitForChatRateLimit', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.useRealTimers();
  });

  it('returns immediately when ZEN_AGENT_CHAT_RPM is unset', async () => {
    delete process.env.ZEN_AGENT_CHAT_RPM;
    await expect(waitForChatRateLimit()).resolves.toBeUndefined();
  });

  it('spaces requests to one per 60000/rpm ms', async () => {
    process.env.ZEN_AGENT_CHAT_RPM = '120'; // one request per 500ms

    await waitForChatRateLimit(); // immediate — no prior reservation
    const second = waitForChatRateLimit();
    const third = waitForChatRateLimit();

    await vi.advanceTimersByTimeAsync(500);
    await expect(second).resolves.toBeUndefined();
    await vi.advanceTimersByTimeAsync(500);
    await expect(third).resolves.toBeUndefined();
  });

  it('rejects with the abort reason while waiting', async () => {
    process.env.ZEN_AGENT_CHAT_RPM = '120';
    const controller = new AbortController();

    // Flush reservations left by previous tests, then occupy the immediate slot.
    await vi.advanceTimersByTimeAsync(60_000);
    await waitForChatRateLimit();
    const waiting = waitForChatRateLimit(controller.signal);

    controller.abort(new Error('user cancelled'));
    await expect(waiting).rejects.toThrow('user cancelled');
  });
});
