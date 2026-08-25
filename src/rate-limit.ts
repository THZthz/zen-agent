import { envPositiveInt } from './env.js';

/**
 * Client-side spacing of chat requests, ported from Reasonix's
 * waitForChatRateLimit. Throttles to at most one request per 60000/rpm ms so
 * bursts do not trip the provider's 429. Disabled unless ZEN_AGENT_CHAT_RPM
 * is set (0 disables it explicitly).
 */

// Process-global on purpose: the throttle exists to keep THIS process from
// bursting the provider's request-per-minute budget, regardless of how many
// sessions share it. Known tradeoffs:
// - a request aborted while waiting still consumes its reserved slot (the
//   reservation happens up front so queued bursts spread out); harmless,
//   since an aborted caller frees its real budget slot anyway.
// - providers are not keyed separately; with ZEN_AGENT_CHAT_RPM set and two
//   providers in use, both draw from one shared spacing budget.
let nextChatRequestAt = 0;

export async function waitForChatRateLimit(signal?: AbortSignal): Promise<void> {
  const rpm = parseChatRpm();
  if (rpm <= 0) return;
  const minIntervalMs = Math.ceil(60_000 / rpm);

  const now = Date.now();
  const waitMs = Math.max(0, nextChatRequestAt - now);
  // Reserve the slot even when this request is already late, so a burst of
  // queued requests spreads out instead of all firing at once.
  nextChatRequestAt = Math.max(now, nextChatRequestAt) + minIntervalMs;

  if (waitMs <= 0) return;
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, waitMs);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(signal.reason instanceof Error ? signal.reason : new Error('aborted'));
      },
      { once: true },
    );
  });
}

/** Test/startup hook: drop any reserved slot so the next request fires immediately. */
export function resetChatRateLimit(): void {
  nextChatRequestAt = 0;
}

/** Requests per minute cap; unset/invalid/0 disables the throttle. */
function parseChatRpm(): number {
  return envPositiveInt('ZEN_AGENT_CHAT_RPM', 0);
}
