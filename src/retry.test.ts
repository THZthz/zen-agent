import { describe, expect, it, vi } from "vitest";
import { fetchWithRetry } from "./retry.js";

function jsonResponse(status: number, headers: Record<string, string> = {}): Response {
  return new Response("body", { status, headers });
}

describe("fetchWithRetry", () => {
  it("returns a non-retryable 4xx immediately without retrying", async () => {
    const fetchFn = vi.fn(async () => jsonResponse(401));
    const resp = await fetchWithRetry(fetchFn, "https://example.test", {}, { maxAttempts: 3 });
    expect(resp.status).toBe(401);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("retries 429 and succeeds on the next attempt", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(429))
      .mockResolvedValueOnce(jsonResponse(200));
    const resp = await fetchWithRetry(fetchFn, "https://example.test", {}, {
      initialBackoffMs: 1,
      maxBackoffMs: 2,
    });
    expect(resp.status).toBe(200);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("honors the Retry-After header", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(429, { "Retry-After": "0.01" }))
      .mockResolvedValueOnce(jsonResponse(200));
    const retries: Array<{ attempt: number; reason: string; waitMs: number }> = [];
    await fetchWithRetry(fetchFn, "https://example.test", {}, {
      initialBackoffMs: 10_000,
      maxBackoffMs: 60_000,
      onRetry: (info) => retries.push(info),
    });
    expect(retries).toHaveLength(1);
    expect(retries[0]!.waitMs).toBe(10); // 0.01s * 1000, capped
    expect(retries[0]!.reason).toBe("http 429");
  });

  it("jitters exponential backoff within [75%, 125%]", async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(503))
      .mockResolvedValueOnce(jsonResponse(503))
      .mockResolvedValueOnce(jsonResponse(200));
    const retries: number[] = [];
    await fetchWithRetry(fetchFn, "https://example.test", {}, {
      initialBackoffMs: 100,
      maxBackoffMs: 10_000,
      onRetry: (info) => retries.push(info.waitMs),
    });
    // attempt 1: 100 * 2^0 = 100 → [75, 125]; attempt 2: 100 * 2^1 = 200 → [150, 250]
    expect(retries).toHaveLength(2);
    expect(retries[0]!).toBeGreaterThanOrEqual(75);
    expect(retries[0]!).toBeLessThanOrEqual(125);
    expect(retries[1]!).toBeGreaterThanOrEqual(150);
    expect(retries[1]!).toBeLessThanOrEqual(250);
  });

  it("returns the last response when retryable attempts run out", async () => {
    const fetchFn = vi.fn(async () => jsonResponse(503));
    const resp = await fetchWithRetry(fetchFn, "https://example.test", {}, {
      maxAttempts: 2,
      initialBackoffMs: 1,
      maxBackoffMs: 2,
    });
    expect(resp.status).toBe(503);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("never retries once the signal is aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetchFn = vi.fn(async () => jsonResponse(200));
    await expect(
      fetchWithRetry(fetchFn, "https://example.test", {}, { signal: controller.signal }),
    ).rejects.toThrow("aborted");
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("retries network errors and throws when they persist", async () => {
    const flaky = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(jsonResponse(200));
    const resp = await fetchWithRetry(flaky, "https://example.test", {}, {
      initialBackoffMs: 1,
      maxBackoffMs: 2,
    });
    expect(resp.status).toBe(200);

    const alwaysDown = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    await expect(
      fetchWithRetry(alwaysDown, "https://example.test", {}, {
        maxAttempts: 2,
        initialBackoffMs: 1,
        maxBackoffMs: 2,
      }),
    ).rejects.toThrow("fetch failed");
  });
});
