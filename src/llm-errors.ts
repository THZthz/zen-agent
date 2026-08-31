import { fetchBalanceSnapshot } from './provider.js';
import { requireProviderDefinition } from './provider-registry.js';
import type { ProviderId } from './storage.js';

/**
 * Classify an LLM API failure into user-facing guidance. Errors thrown
 * by runChatCompletions carry the message shape `<label> API error <status>: <body>`;
 * anything else is returned unchanged.
 */

const API_ERROR_RE = /^([A-Za-z]+) API error (\d{3}): ([\s\S]*)$/;

export async function formatLlmError(
  err: unknown,
  opts: { provider: ProviderId },
): Promise<string> {
  if (!(err instanceof Error)) return String(err);
  const match = API_ERROR_RE.exec(err.message ?? '');
  if (!match) return err.message;

  const status = Number.parseInt(match[2]!, 10);
  const body = match[3] ?? '';
  const def = requireProviderDefinition(opts.provider);

  // Context overflow comes back as a 400 whose body names the limit — give
  // the user the fix (fresh session) instead of the raw body.
  if (/maximum context length/i.test(body)) {
    const requested = /requested\s+([\d,]+)\s+tokens/.exec(body)?.[1];
    return requested
      ? `Context window exceeded — the request needs ${requested} tokens but the model's context limit is lower. Start a new session to reset the conversation context.`
      : "Context window exceeded — the request is larger than the model's context limit. Start a new session to reset the conversation context.";
  }

  switch (status) {
    case 401:
      return `API key rejected (401). Check ${def.apiKeyEnv ?? 'your API key'}.`;
    case 402:
      return def.id === 'deepseek'
        ? 'Insufficient DeepSeek balance (402). Top up your account at https://platform.deepseek.com.'
        : err.message;
    case 400:
      return `Bad request (400): ${extractErrorMessage(body)}`;
    case 422:
      return `Request rejected (422): ${extractErrorMessage(body)}`;
    case 429:
      return 'Rate limit reached (429). The request was already retried automatically — wait a few seconds and try again.';
    default:
      if (status >= 500) {
        return format5xx(status, opts.provider);
      }
      return err.message;
  }
}

/** 5xx: probe the provider with a cheap balance call so the user knows whether to check their network or wait. */
async function format5xx(status: number, provider: ProviderId): Promise<string> {
  const def = requireProviderDefinition(provider);
  const canProbe = def.balance && (!def.apiKeyEnv || process.env[def.apiKeyEnv]);
  if (!canProbe) {
    return `${def.label} returned ${status} — a temporary server error. Retry in a few seconds or try a different model.`;
  }
  const reachable = await probeProviderReachable(provider);
  return reachable
    ? `${def.label} returned ${status} but the API is reachable — a transient server error. Wait a moment and retry.`
    : `Cannot reach ${def.label} (${status}) — check your network connection or the provider's service status.`;
}

async function probeProviderReachable(provider: ProviderId): Promise<boolean> {
  try {
    await fetchBalanceSnapshot(provider);
    return true;
  } catch {
    return false;
  }
}

/** OpenAI-compatible error bodies are JSON `{error: {message}}`; fall back to the raw body. */
function extractErrorMessage(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) return '(empty response body)';
  try {
    const parsed = JSON.parse(trimmed) as { error?: { message?: unknown }; message?: unknown };
    if (parsed?.error && typeof parsed.error.message === 'string') return parsed.error.message;
    if (typeof parsed?.message === 'string') return parsed.message;
  } catch {
    /* not JSON — fall through */
  }
  return trimmed;
}
