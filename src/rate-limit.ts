/**
 * Client-side spacing of chat requests, ported from Reasonix's
 * waitForChatRateLimit. Throttles to at most one request per 60000/rpm ms so
 * bursts do not trip the provider's 429. Disabled unless ZEN_AGENT_CHAT_RPM
 * is set (0 disables it explicitly).
 */

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
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason instanceof Error ? signal.reason : new Error("aborted"));
      },
      { once: true },
    );
  });
}

function parseChatRpm(): number {
  const raw = process.env.ZEN_AGENT_CHAT_RPM;
  if (!raw) return 0;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}
