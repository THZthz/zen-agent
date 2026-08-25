import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  clientModelsPath,
  DEFAULT_OPENROUTER_MODEL,
  type ThinkingEffort,
} from "./storage.js";
import {
  buildLlmUsage,
  runChatCompletions,
  type LlmStepOptions,
  type LlmStepResult,
  type LlmUsage,
} from "./llm-client.js";

export { DEFAULT_OPENROUTER_MODEL } from "./storage.js";

function getOpenRouterApiKey(): string {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY environment variable is required");
  }
  return apiKey;
}

function getOpenRouterBaseUrl(): string {
  return (
    process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1"
  ).replace(/\/+$/, "");
}

/**
 * Default OpenRouter provider-routing sort key (https://openrouter.ai/docs/guides/routing/provider-selection).
 * `price` routes to the cheapest provider for the model; `throughput` and
 * `latency` are the other supported values. Empty string opts out of sending
 * the `provider` block entirely.
 */
const DEFAULT_OPENROUTER_PROVIDER_SORT = "price";

/** Provider sort key from OPENROUTER_PROVIDER_SORT; null disables the block. */
function getOpenRouterProviderSort(): string | null {
  const raw = process.env.OPENROUTER_PROVIDER_SORT;
  if (raw === undefined) {
    return DEFAULT_OPENROUTER_PROVIDER_SORT;
  }
  const sort = raw.trim();
  return sort.length > 0 ? sort : null;
}

/**
 * Per-model metadata (USD per 1M tokens, context window).
 *
 * The live `GET /models` endpoint is the source of truth (fetched once per
 * process and cached); the fallbacks below only apply when that fetch fails
 * (offline start, invalid key, ...) or for models that left the catalog.
 * The values are indicative snapshots, not guaranteed current.
 */
export interface OpenRouterModelInfo {
  /** USD per 1M input tokens. */
  inputPerM: number;
  /** USD per 1M output tokens. */
  outputPerM: number;
  contextLength: number;
}

/** Full catalog entry: model info plus selector-relevant fields. */
interface CatalogEntry extends OpenRouterModelInfo {
  id: string;
  name: string | null;
  /** Whether the model supports tool calling (the agent requires it). */
  supportsTools: boolean;
  /**
   * Accepted input modalities from `architecture.input_modalities` (e.g.
   * ["text", "image"]), or null when unknown; "text" is implicit.
   */
  inputModalities: string[] | null;
}

const MODEL_FALLBACKS: Record<string, OpenRouterModelInfo> = {
  // `openrouter/free` routes to free models, billed at $0.
  "openrouter/free": { inputPerM: 0, outputPerM: 0, contextLength: 128_000 },
};

/** Conservative default for models unknown to both the live catalog and the fallback table. */
const UNKNOWN_MODEL_FALLBACK: OpenRouterModelInfo = {
  inputPerM: 1,
  outputPerM: 2,
  contextLength: 200_000,
};

/** Live catalog fetch timeout: config options must not block session creation forever. */
const MODELS_FETCH_TIMEOUT_MS = 5_000;
/** Bumped whenever the persisted catalog file shape changes. */
const MODELS_CACHE_VERSION = 2;

function parsePrice(raw: string | undefined): number {
  const parsed = Number.parseFloat(raw ?? "");
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

let modelsCache: { key: string; promise: Promise<Map<string, CatalogEntry>> } | null =
  null;

/** Test hook: drop the cached /models response (env/port changes between tests). */
export function resetOpenRouterModelsCache(): void {
  modelsCache = null;
}

/** Missing `supported_parameters` metadata = assume the model supports tools. */
function supportsTools(supportedParameters: unknown): boolean {
  return !Array.isArray(supportedParameters) || supportedParameters.includes("tools");
}

/**
 * Fetch OpenRouter's model catalog once per (base URL, key) and cache it in
 * memory for the process lifetime. Throws on missing key/HTTP errors; callers
 * fall back to the persisted file or the static table.
 */
async function fetchOpenRouterModels(): Promise<Map<string, CatalogEntry>> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const key = `${baseUrl}|${apiKey}`;
  if (modelsCache?.key === key) {
    return modelsCache.promise;
  }
  const promise: Promise<Map<string, CatalogEntry>> = (async () => {
    const response = await fetch(`${baseUrl}/models`, {
      headers: { authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) {
      throw new Error(
        `OpenRouter models API error ${response.status}: ${(await response.text().catch(() => "")).slice(0, 500)}`,
      );
    }
    const data = (await response.json()) as {
      data?: Array<{
        id?: string;
        name?: string | null;
        context_length?: number;
        pricing?: { prompt?: string; completion?: string; request?: string };
        supported_parameters?: string[];
        architecture?: { input_modalities?: unknown };
      }>;
    };
    const models = new Map<string, CatalogEntry>();
    for (const model of data.data ?? []) {
      if (!model.id) {
        continue;
      }
      models.set(model.id, {
        id: model.id,
        name: model.name ?? null,
        inputPerM: parsePrice(model.pricing?.prompt),
        outputPerM: parsePrice(model.pricing?.completion),
        contextLength:
          typeof model.context_length === "number" && model.context_length > 0
            ? model.context_length
            : 200_000,
        supportsTools: supportsTools(model.supported_parameters),
        inputModalities: parseInputModalities(model.architecture?.input_modalities),
      });
    }
    return models;
  })();
  modelsCache = { key, promise };
  // A rejected fetch must not be cached for the process lifetime (an offline
  // start would otherwise pin the fallback tables forever): drop the entry so
  // the next call retries. The catch also prevents an unhandled rejection if
  // no caller is attached yet.
  promise.catch(() => {
    if (modelsCache?.promise === promise) {
      modelsCache = null;
    }
  });
  return promise;
}

/**
 * Parse `architecture.input_modalities` into a string list; null when absent
 * or malformed (callers treat unknown as text-only).
 */
export function parseInputModalities(raw: unknown): string[] | null {
  if (!Array.isArray(raw)) {
    return null;
  }
  const modalities = raw.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
  return modalities.length > 0 ? modalities : null;
}

/**
 * Input modalities accepted by an OpenRouter model ("image"/"audio"
 * membership matters); null when unknown (no catalog / unknown slug), empty
 * for known text-only fallback models like `openrouter/free`.
 */
export async function getOpenRouterModelModalities(
  model: string,
): Promise<string[] | null> {
  try {
    const entry = (await fetchOpenRouterModels()).get(model);
    if (entry) {
      return entry.inputModalities;
    }
  } catch {
    // Offline start, bad key, ... - unknown modalities.
  }
  return MODEL_FALLBACKS[model] ? [] : null;
}

/**
 * Effective pricing/context info for an OpenRouter model id. Prefers the
 * live catalog; falls back to the static table, then to generic defaults.
 */
export async function getOpenRouterModelInfo(model: string): Promise<OpenRouterModelInfo> {
  try {
    const entry = (await fetchOpenRouterModels()).get(model);
    if (entry) {
      return {
        inputPerM: entry.inputPerM,
        outputPerM: entry.outputPerM,
        contextLength: entry.contextLength,
      };
    }
  } catch {
    // Offline start, bad key, ... — use the static fallbacks.
  }
  return MODEL_FALLBACKS[model] ?? UNKNOWN_MODEL_FALLBACK;
}

/** Shape of the persisted catalog file (clientModelsPath). */
interface ModelsCacheFile {
  version: number;
  fetchedAt: string;
  baseUrl: string;
  models: CatalogEntry[];
}

/**
 * Process-global write-once guard: avoid rewriting the catalog file on every
 * session in one process. Keyed on a single cwd — switching between projects
 * in one agent process writes once per project switch, which is fine.
 */
let modelsPersistedCwd: string | null = null;

/** Best-effort: persist the fetched catalog so offline restarts can use it. */
async function writeModelsFile(cwd: string, catalog: Map<string, CatalogEntry>): Promise<void> {
  if (modelsPersistedCwd === cwd) {
    return;
  }
  try {
    const payload: ModelsCacheFile = {
      version: MODELS_CACHE_VERSION,
      fetchedAt: new Date().toISOString(),
      baseUrl: getOpenRouterBaseUrl(),
      models: [...catalog.values()],
    };
    // Atomic replace: a crash mid-write must never leave a truncated cache
    // file behind (readModelsFile would discard it anyway, but only after
    // logging nothing and silently losing the offline fallback).
    const target = clientModelsPath(cwd);
    await mkdir(dirname(target), { recursive: true });
    const tmp = `${target}.${process.pid}.tmp`;
    try {
      await writeFile(tmp, `${JSON.stringify(payload)}\n`, "utf8");
      await rename(tmp, target);
    } catch (error) {
      await unlink(tmp).catch(() => {});
      throw error;
    }
    modelsPersistedCwd = cwd;
  } catch {
    // Caching must never break session creation.
  }
}

/** Read the persisted catalog; null when absent, malformed or outdated. */
async function readModelsFile(cwd: string): Promise<Map<string, CatalogEntry> | null> {
  try {
    const raw = await readFile(clientModelsPath(cwd), "utf8");
    const parsed = JSON.parse(raw) as ModelsCacheFile;
    if (parsed.version !== MODELS_CACHE_VERSION || !Array.isArray(parsed.models)) {
      return null;
    }
    const catalog = new Map<string, CatalogEntry>();
    for (const entry of parsed.models) {
      if (entry && typeof entry.id === "string" && entry.id.length > 0) {
        catalog.set(entry.id, entry);
      }
    }
    return catalog;
  } catch {
    return null;
  }
}

export interface OpenRouterModelOption {
  value: string;
  name: string;
  description: string;
}

/**
 * Model choices for the session selector. Prefers the live catalog (fetched
 * automatically and persisted to `.sessions/client/models.openrouter.json`),
 * falls back to the persisted file for offline starts, and returns null when
 * neither exists (the caller then uses the static list).
 */
export async function getOpenRouterModelOptions(
  cwd: string,
): Promise<OpenRouterModelOption[] | null> {
  let catalog: Map<string, CatalogEntry> | null = null;
  try {
    catalog = await fetchOpenRouterModels();
    await writeModelsFile(cwd, catalog);
  } catch {
    catalog = await readModelsFile(cwd);
  }
  if (!catalog || catalog.size === 0) {
    return null;
  }
  return buildModelOptions(catalog);
}

/** Tool-capable models only; `openrouter/free` first, then alphabetical. */
function buildModelOptions(catalog: Map<string, CatalogEntry>): OpenRouterModelOption[] {
  const options: OpenRouterModelOption[] = [];
  for (const entry of catalog.values()) {
    if (!entry.supportsTools) {
      continue;
    }
    options.push({
      value: entry.id,
      name: entry.name ?? entry.id,
      description: `${entry.name ?? entry.id} · ${formatContext(entry.contextLength)}`,
    });
  }
  options.sort((a, b) => a.value.localeCompare(b.value, "en"));
  const freeIndex = options.findIndex((option) => option.value === "openrouter/free");
  if (freeIndex === -1) {
    options.unshift({
      value: "openrouter/free",
      name: "OpenRouter Free",
      description: "OpenRouter's free-tier routing model",
    });
  } else if (freeIndex > 0) {
    options.unshift(...options.splice(freeIndex, 1));
  }
  return options;
}

function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1)}M ctx`;
  }
  if (tokens >= 1000) {
    return `${Math.round(tokens / 1000)}K ctx`;
  }
  return `${tokens} ctx`;
}

/** OpenRouter's streaming usage object (normalized OpenAI shape). */
interface OpenRouterUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  /** Provider passthrough — DeepSeek routes report cache hits this way. */
  prompt_cache_hit_tokens?: number;
  /** Anthropic-style passthrough of cached input tokens. */
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

function toNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/**
 * Parse OpenRouter's usage chunk. Cache-hit and reasoning-token fields are
 * only present when the upstream provider passes them through; everything
 * else falls back to 0 (stats then show cache hit 0% / no reasoning tokens,
 * which is accurate for providers that do not report them).
 */
export function parseOpenRouterUsage(
  usage: OpenRouterUsage | undefined,
  timing: { llmMs: number; thinkingMs: number; answeringMs: number },
): LlmUsage | null {
  if (!usage) {
    return null;
  }
  const inputTokens = toNumber(usage.prompt_tokens);
  const outputTokens = toNumber(usage.completion_tokens);
  const cacheReadTokens =
    usage.prompt_cache_hit_tokens !== undefined
      ? toNumber(usage.prompt_cache_hit_tokens)
      : toNumber(usage.prompt_tokens_details?.cached_tokens);
  return buildLlmUsage(
    {
      inputTokens,
      outputTokens,
      totalTokens: toNumber(usage.total_tokens),
      cacheReadTokens,
      cacheMissTokens: Math.max(0, inputTokens - cacheReadTokens),
      reasoningTokens: toNumber(usage.completion_tokens_details?.reasoning_tokens),
    },
    timing,
  );
}

/** OpenRouter key info from `GET /auth/key`, in USD credits. */
export interface OpenRouterBalance {
  isAvailable: boolean;
  currency: string;
  /** Remaining credits: the key's limit minus its usage. */
  remainingUsd: number;
  usageUsd: number;
  limitUsd: number;
  isFreeTier: boolean;
}

/**
 * Fetch the current key usage/limit from OpenRouter's `GET /auth/key`.
 * Used to cross-check the locally estimated token cost against actual
 * credit consumption (see ZenAgent.verifyTurnCost).
 */
export async function fetchOpenRouterBalance(): Promise<OpenRouterBalance> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const response = await fetch(`${baseUrl}/auth/key`, {
    headers: { authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    throw new Error(
      `OpenRouter auth/key API error ${response.status}: ${(await response.text().catch(() => "")).slice(0, 500)}`,
    );
  }
  const data = (await response.json()) as {
    data?: { label?: string; usage?: number; limit?: number; is_free_tier?: boolean };
  };
  const usageUsd = toNumber(data.data?.usage);
  const limitUsd = toNumber(data.data?.limit);
  return {
    isAvailable: true,
    currency: "USD",
    remainingUsd: Math.max(0, limitUsd - usageUsd),
    usageUsd,
    limitUsd,
    isFreeTier: data.data?.is_free_tier ?? false,
  };
}

/**
 * OpenRouter's chat completions step (OpenAI-compatible). Shares the SSE
 * client with DeepSeek; OpenRouter-specific bits:
 *
 * - reasoning streams as `delta.reasoning` (OpenRouter's normalized field;
 *   `delta.reasoning_content` is also accepted for DeepSeek passthrough
 *   routes), and stored reasoning is sent back as `reasoning` in history
 * - `stream_options.include_usage` requests the final usage chunk (OpenRouter
 *   does not send usage otherwise)
 * - `provider.sort` defaults to `price` so requests route to the cheapest
 *   provider for the model (OPENROUTER_PROVIDER_SORT overrides; empty disables)
 * - `reasoning_effort` uses the OpenAI vocabulary (low/medium/high), so
 *   DeepSeek's `max` maps to `high`
 * - optional `HTTP-Referer` / `X-Title` headers identify the app
 *   (OPENROUTER_SITE_URL / OPENROUTER_APP_NAME)
 */
export async function runOpenRouterStep(options: LlmStepOptions): Promise<LlmStepResult> {
  const apiKey = getOpenRouterApiKey();
  const baseUrl = getOpenRouterBaseUrl();
  const modelName =
    options.model ?? process.env.OPENROUTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;

  const extraHeaders: Record<string, string> = {};
  const siteUrl = process.env.OPENROUTER_SITE_URL;
  if (siteUrl) {
    extraHeaders["HTTP-Referer"] = siteUrl;
  }
  const appName = process.env.OPENROUTER_APP_NAME;
  if (appName) {
    extraHeaders["X-Title"] = appName;
  }
  const providerSort = getOpenRouterProviderSort();

  return runChatCompletions({
    baseUrl,
    apiKey,
    label: "OpenRouter",
    model: modelName,
    messages: options.messages,
    tools: options.tools,
    system: options.system,
    signal: options.signal,
    onTextDelta: options.onTextDelta,
    onReasoningDelta: options.onReasoningDelta,
    logRuntime: options.logRuntime,
    thinkingEffort: options.thinkingEffort,
    reasoningMessageField: "reasoning",
    reasoningDeltaFields: ["reasoning", "reasoning_content"],
    effortBody: (effort: ThinkingEffort) =>
      effort === "off"
        ? undefined
        : { reasoning_effort: effort === "max" ? "high" : effort },
    extraBody: {
      stream_options: { include_usage: true },
      ...(providerSort ? { provider: { sort: providerSort } } : {}),
    },
    extraHeaders,
    parseUsage: (raw, timing) => parseOpenRouterUsage(raw as OpenRouterUsage, timing),
  });
}
