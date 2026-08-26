/**
 * Retrying HTTP calls to the LLM API.
 *
 * Rules:
 * - Only `retryableStatuses` (408/429/5xx by default) are retried. A
 *   non-retryable status is returned immediately; when attempts run out the
 *   last response is returned so the caller can surface the real status.
 * - `Retry-After` is honored when present; otherwise exponential backoff with
 *   75-125% jitter so simultaneous 429s do not re-collide.
 * - Explicit aborts are NEVER retried — the user asked to stop.
 * - Only the initial request is retried. A mid-stream failure must not be
 *   re-sent: it would re-bill the user for desynced output.
 */

export interface RetryOptions {
  /** Maximum total attempts including the first. Default 4. */
  maxAttempts?: number;
  /** Initial backoff in ms, doubled each retry. Default 500. */
  initialBackoffMs?: number;
  /** Upper bound on any single backoff delay. Default 10000 (10s). */
  maxBackoffMs?: number;
  /** HTTP statuses to treat as retryable. Default [408, 429, 500, 502, 503, 504]. */
  retryableStatuses?: readonly number[];
  /** Abort signal; we do NOT retry once aborted. */
  signal?: AbortSignal;
  /** Telemetry hook — called before each wait. */
  onRetry?: (info: { attempt: number; reason: string; waitMs: number }) => void;
}

const DEFAULT_RETRYABLE_STATUSES = [408, 429, 500, 502, 503, 504] as const;

export async function fetchWithRetry(
  fetchFn: typeof fetch,
  url: string,
  init: RequestInit,
  opts: RetryOptions = {},
): Promise<Response> {
  const maxAttempts = opts.maxAttempts ?? 4;
  const initial = opts.initialBackoffMs ?? 500;
  const cap = opts.maxBackoffMs ?? 10_000;
  const retryable = new Set(opts.retryableStatuses ?? DEFAULT_RETRYABLE_STATUSES);

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (opts.signal?.aborted) throw abortError(opts.signal);

    try {
      const resp = await fetchFn(url, init);

      // Success or non-retryable failure: return as-is.
      if (resp.ok || !retryable.has(resp.status)) return resp;

      // Retryable but out of attempts: return the last response (body intact)
      // so the caller can surface the status to the user.
      if (attempt === maxAttempts - 1) return resp;

      // Drain the body so the connection can be reused on the next attempt.
      await resp.text().catch(() => undefined);

      const waitMs = computeWait(attempt, initial, cap, resp.headers.get('Retry-After'));
      opts.onRetry?.({ attempt: attempt + 1, reason: `http ${resp.status}`, waitMs });
      await sleep(waitMs, opts.signal);
    } catch (err) {
      // Respect explicit aborts — do not retry.
      if (isAbortError(err) || opts.signal?.aborted) throw err;
      if (attempt === maxAttempts - 1) throw err;

      const waitMs = computeWait(attempt, initial, cap, null);
      opts.onRetry?.({ attempt: attempt + 1, reason: `network: ${messageOf(err)}`, waitMs });
      await sleep(waitMs, opts.signal);
    }
  }

  throw new Error('fetchWithRetry: loop exited unexpectedly');
}

function computeWait(
  attempt: number,
  initial: number,
  cap: number,
  retryAfter: string | null,
): number {
  if (retryAfter) {
    const seconds = Number.parseFloat(retryAfter);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.min(seconds * 1000, cap);
    }
    // HTTP-date form ("Wed, 21 Oct 2026 07:28:00 GMT"): wait until then.
    const at = Date.parse(retryAfter);
    if (!Number.isNaN(at)) {
      const secondsUntil = Math.max(0, (at - Date.now()) / 1000);
      return Math.min(secondsUntil * 1000, cap);
    }
  }
  const exp = initial * 2 ** attempt;
  // Jitter range [75%, 125%] to spread retries out when many clients hit 429 together.
  const jitter = exp * (0.75 + Math.random() * 0.5);
  return Math.min(Math.max(jitter, 0), cap);
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError(signal!));
    };
    const timer = setTimeout(() => {
      // Normal resolution must not leak the abort listener.
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

/** Preserve the abort reason (e.g. a timeout error) instead of masking it with "aborted". */
function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error('aborted');
}

function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  return (err as { name?: unknown }).name === 'AbortError';
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.message;
  try {
    return String(err);
  } catch {
    return 'unknown error';
  }
}
