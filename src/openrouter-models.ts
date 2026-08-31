import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { clientModelsPath } from './storage.js';

export function getOpenRouterApiKey(): string {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY environment variable is required');
  }
  return apiKey;
}

export function getOpenRouterBaseUrl(): string {
  return (process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/+$/, '');
}

/**
 * Default OpenRouter provider-routing sort key (https://openrouter.ai/docs/guides/routing/provider-selection).
 * `price` routes to the cheapest provider for the model; `throughput` and
 * `latency` are the other supported values. Empty string opts out of sending
 * the `provider` block entirely.
 */
const DEFAULT_OPENROUTER_PROVIDER_SORT = 'price';

/** Provider sort key from OPENROUTER_PROVIDER_SORT; null disables the block. */
export function getOpenRouterProviderSort(): string | null {
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

/**
 * Reasoning-effort configuration for an OpenRouter model, from the
 * catalog's `reasoning` object (e.g. `{"supported_efforts": ["high",
 * "medium", "low", "minimal"], "default_effort": "medium", "mandatory":
 * true}`). `supportedEfforts` is listed in descending effort order; null
 * means the model accepts every gateway effort value.
 */
export interface OpenRouterReasoning {
  /** Allowed `reasoning_effort` values for this model, highest first. */
  supportedEfforts: readonly string[] | null;
  /** The model's default effort when the field is omitted; null when unknown. */
  defaultEffort: string | null;
  /**
   * Whether the model's reasoning is mandatory (cannot be disabled): the
   * `none` effort is absent from `supportedEfforts` and `mandatory` is true.
   * Null when the catalog does not say.
   */
  mandatory: boolean | null;
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
  /** Reasoning-effort allowlist from the catalog; null = all values accepted. */
  reasoning: OpenRouterReasoning;
}

const MODEL_FALLBACKS: Record<string, OpenRouterModelInfo> = {
  // `openrouter/free` routes to free models, billed at $0.
  'openrouter/free': { inputPerM: 0, outputPerM: 0, contextLength: 128_000 },
};

/** Conservative default for models unknown to both the live catalog and the fallback table. */
const UNKNOWN_MODEL_FALLBACK: OpenRouterModelInfo = {
  inputPerM: 1,
  outputPerM: 2,
  contextLength: 200_000,
};

/** Live catalog fetch timeout: config options must not block session creation forever. */
const MODELS_FETCH_TIMEOUT_MS = 5_000;
/** Bumped whenever the persisted catalog shape or field semantics change. */
const MODELS_CACHE_VERSION = 4;

/** Convert OpenRouter's wire USD/token price to the catalog's USD/1M-token unit. */
function parsePricePerM(raw: string | undefined): number {
  const parsed = Number.parseFloat(raw ?? '');
  return Number.isFinite(parsed) && parsed >= 0 ? parsed * 1_000_000 : 0;
}

let modelsCache: { key: string; promise: Promise<Map<string, CatalogEntry>> } | null = null;

/** Test hook: drop the cached /models response (env/port changes between tests). */
export function resetOpenRouterModelsCache(): void {
  modelsCache = null;
}

/** Missing `supported_parameters` metadata = assume the model supports tools. */
function supportsTools(supportedParameters: unknown): boolean {
  return !Array.isArray(supportedParameters) || supportedParameters.includes('tools');
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
        `OpenRouter models API error ${response.status}: ${(await response.text().catch(() => '')).slice(0, 500)}`,
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
        reasoning?: unknown;
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
        inputPerM: parsePricePerM(model.pricing?.prompt),
        outputPerM: parsePricePerM(model.pricing?.completion),
        contextLength:
          typeof model.context_length === 'number' && model.context_length > 0
            ? model.context_length
            : 200_000,
        supportsTools: supportsTools(model.supported_parameters),
        inputModalities: parseInputModalities(model.architecture?.input_modalities),
        reasoning: parseReasoning(model.reasoning),
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
    (entry): entry is string => typeof entry === 'string' && entry.length > 0,
  );
  return modalities.length > 0 ? modalities : null;
}

/**
 * Parse the catalog's `reasoning` object into an effort allowlist +
 * default. `supported_efforts` is ordered highest-first by the gateway;
 * null when absent/malformed (callers then treat every gateway effort value
 * as accepted).
 */
export function parseReasoning(raw: unknown): OpenRouterReasoning {
  if (typeof raw !== 'object' || raw === null) {
    return { supportedEfforts: null, defaultEffort: null, mandatory: null };
  }
  const reasoning = raw as {
    supported_efforts?: unknown;
    default_effort?: unknown;
    mandatory?: unknown;
  };
  const supportedEfforts = Array.isArray(reasoning.supported_efforts)
    ? reasoning.supported_efforts.filter(
        (entry): entry is string => typeof entry === 'string' && entry.length > 0,
      )
    : [];
  return {
    supportedEfforts: supportedEfforts.length > 0 ? supportedEfforts : null,
    defaultEffort:
      typeof reasoning.default_effort === 'string' && reasoning.default_effort.length > 0
        ? reasoning.default_effort
        : null,
    mandatory: typeof reasoning.mandatory === 'boolean' ? reasoning.mandatory : null,
  };
}

/**
 * Input modalities accepted by an OpenRouter model ("image"/"audio"
 * membership matters); null when unknown (no catalog / unknown slug), empty
 * for known text-only fallback models like `openrouter/free`.
 */
export async function getOpenRouterModelModalities(model: string): Promise<string[] | null> {
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

/**
 * Reasoning-effort config for an OpenRouter model id, from the live catalog
 * (null allowlist when the model is unknown or the catalog is unavailable —
 * the gateway then accepts every effort value). When `cwd` is given and the
 * live fetch fails, the persisted catalog file is consulted so offline
 * starts keep the per-model allowlist (same fallback as the model selector).
 */
export async function getOpenRouterReasoning(
  model: string,
  cwd?: string,
): Promise<OpenRouterReasoning> {
  try {
    const entry = (await fetchOpenRouterModels()).get(model);
    if (entry) {
      return entry.reasoning;
    }
  } catch {
    // Offline start, bad key, ... - fall back to the persisted file below.
  }
  if (cwd) {
    const cached = await readModelsFile(cwd);
    const entry = cached?.get(model);
    if (entry) {
      return entry.reasoning;
    }
  }
  return { supportedEfforts: null, defaultEffort: null, mandatory: null };
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
      await writeFile(tmp, `${JSON.stringify(payload)}\n`, 'utf8');
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
    const raw = await readFile(clientModelsPath(cwd), 'utf8');
    const parsed = JSON.parse(raw) as ModelsCacheFile;
    if (parsed.version !== MODELS_CACHE_VERSION || !Array.isArray(parsed.models)) {
      return null;
    }
    const catalog = new Map<string, CatalogEntry>();
    for (const entry of parsed.models) {
      if (entry && typeof entry.id === 'string' && entry.id.length > 0) {
        // Older cache files predate the reasoning block; unknown allowlist
        // (null) is the safe default: every gateway effort value accepted.
        if (!entry.reasoning) {
          entry.reasoning = { supportedEfforts: null, defaultEffort: null, mandatory: null };
        }
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
  options.sort((a, b) => a.value.localeCompare(b.value, 'en'));
  const freeIndex = options.findIndex((option) => option.value === 'openrouter/free');
  if (freeIndex === -1) {
    options.unshift({
      value: 'openrouter/free',
      name: 'OpenRouter Free',
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
