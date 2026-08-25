/**
 * The recurring "override via environment variable, else default" pattern in
 * one place, so invalid values consistently fall back instead of throwing or
 * NaN-ing their way through the agent.
 */

/**
 * Non-negative float override (prices per 1M tokens, ...). Falls back when
 * unset, empty, non-numeric or negative.
 */
export function envNonNegativeFloat(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

/**
 * Positive integer override (timeouts, byte limits, RPM caps, ...). Falls
 * back when unset, empty, non-numeric or <= 0 — a non-positive value means
 * "invalid", not "disabled"; callers that want a disable switch use a
 * fallback of 0 and treat 0 as off themselves.
 */
export function envPositiveInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
