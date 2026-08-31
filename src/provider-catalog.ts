import { mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { indexDirectory, writeFileAtomic } from './session-index.js';
import { readFile } from 'node:fs/promises';

/**
 * Generic OpenAI-compatible model discovery: GET {baseUrl}/models, parsed
 * into a Zen-side catalog (context, pricing, modalities, reasoning, tool
 * support) and persisted per provider for offline starts. OpenRouter's extra
 * fields (pricing, context_length, architecture.input_modalities, reasoning,
 * supported_parameters) are optional extras; generic endpoints degrade to
 * conservative defaults.
 */

export interface CatalogReasoning {
  /** Allowed `reasoning_effort` values for this model, highest first; null = all accepted. */
  supportedEfforts: readonly string[] | null;
  /** The model's default effort when the field is omitted; null when unknown. */
  defaultEffort: string | null;
  /** Whether reasoning is mandatory (cannot be disabled); null when unknown. */
  mandatory: boolean | null;
}

/** Per-model metadata from a `/models` catalog entry. */
export interface CatalogEntry {
  id: string;
  name: string | null;
  /** USD per 1M input tokens. */
  inputPerM: number;
  /** USD per 1M output tokens. */
  outputPerM: number;
  /** USD per 1M cache-read input tokens; 0 when the provider does not break it out. */
  cacheReadPerM: number;
  /** Maximum output tokens; undefined when unknown. */
  maxOutputTokens?: number;
  contextLength: number;
  /** Whether the model supports tool calling (the agent requires it). */
  supportsTools: boolean;
  /**
   * Accepted input modalities from `architecture.input_modalities` (e.g.
   * ["text", "image", "audio"]), or null when unknown; "text" is implicit.
   */
  inputModalities: string[] | null;
  reasoning: CatalogReasoning;
}

/** Conservative default for models unknown to both the catalog and the fallback table. */
export const UNKNOWN_MODEL_CONTEXT = 200_000;

/** Catalog fetch timeout: config options must not block session creation forever. */
export const MODELS_FETCH_TIMEOUT_MS = 5_000;

/** Bumped whenever the persisted catalog shape or field semantics change. */
export const CATALOG_CACHE_VERSION = 5;

/** Convert an OpenRouter-style wire USD/token price to USD/1M-token units. */
export function parsePricePerM(raw: string | undefined): number {
  const parsed = Number.parseFloat(raw ?? '');
  if (!Number.isFinite(parsed) || parsed < 0) {
    return 0;
  }
  // Round to 6 decimals so per-1M prices like 0.0000002*1M come out clean
  // (0.2) instead of 0.19999999999999998.
  return Number((parsed * 1_000_000).toFixed(6));
}

/** Missing `supported_parameters` metadata = assume the model supports tools. */
function supportsTools(supportedParameters: unknown): boolean {
  return !Array.isArray(supportedParameters) || supportedParameters.includes('tools');
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
 * Parse the catalog's `reasoning` object into an effort allowlist + default.
 * `supported_efforts` is ordered highest-first by OpenRouter; null when
 * absent/malformed (callers then treat every gateway effort value as accepted).
 */
export function parseReasoning(raw: unknown): CatalogReasoning {
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
 * Curated corrections/fallbacks for models whose provider catalog omits or
 * misreports spec fields (verified against the vendor docs; tvly/web search
 * at implementation time):
 *
 * - z.ai reports 1M (1,048,576) context for GLM-5.3 / GLM-5.3-Flash;
 *   OpenRouter's `/models` lists 1,310,720 (its own FAQ says 1,048,576) and
 *   z.ai's direct endpoint returns no metadata at all.
 * - GLM-5.3-Flash is natively multimodal (text/image/video per docs.z.ai;
 *   audio is NOT in the official input list, so it stays disabled).
 * - Both always reason (thinking cannot be disabled): low/high/max efforts.
 *
 * Spec fields always win over the upstream entry; pricing fields only fill
 * in when the upstream entry has none, so live price updates are not frozen.
 */
export interface ModelSpecOverride {
  contextLength?: number;
  inputModalities?: string[];
  maxOutputTokens?: number;
  reasoning?: CatalogReasoning;
  inputPerM?: number;
  outputPerM?: number;
  cacheReadPerM?: number;
}

export const KNOWN_MODEL_SPECS: Readonly<Record<string, ModelSpecOverride>> = {
  // z.ai direct (api.z.ai OpenAI-compatible ids)
  'glm-5.3': {
    contextLength: 1_048_576,
    inputModalities: ['text'],
    maxOutputTokens: 131_072,
    reasoning: {
      supportedEfforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
      mandatory: true,
    },
    inputPerM: 1.4,
    outputPerM: 4.4,
    cacheReadPerM: 0.26,
  },
  'glm-5.3-flash': {
    contextLength: 1_048_576,
    inputModalities: ['text', 'image', 'video'],
    maxOutputTokens: 131_072,
    reasoning: {
      supportedEfforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
      mandatory: true,
    },
    inputPerM: 0.075,
    outputPerM: 0.25,
    cacheReadPerM: 0.015,
  },
  // OpenRouter slugs for the same models (prices only fill in when the
  // gateway omits them; live OpenRouter pricing wins).
  'z-ai/glm-5.3': {
    contextLength: 1_048_576,
    inputModalities: ['text'],
    maxOutputTokens: 131_072,
    reasoning: {
      supportedEfforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
      mandatory: true,
    },
    inputPerM: 1.4,
    outputPerM: 4.4,
    cacheReadPerM: 0.26,
  },
  'z-ai/glm-5.3-flash': {
    contextLength: 1_048_576,
    inputModalities: ['text', 'image', 'video'],
    maxOutputTokens: 131_072,
    reasoning: {
      supportedEfforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
      mandatory: true,
    },
    inputPerM: 0.075,
    outputPerM: 0.25,
    cacheReadPerM: 0.015,
  },
};

/** Merge the curated spec overrides onto a parsed entry (or synthesize one). */
export function applyModelSpec(
  entry: CatalogEntry | null,
  modelId: string = entry?.id ?? '',
): CatalogEntry | null {
  const spec = modelId ? KNOWN_MODEL_SPECS[modelId] : undefined;
  if (!spec) {
    return entry;
  }
  const base = entry ?? {
    id: modelId,
    name: null,
    inputPerM: 0,
    outputPerM: 0,
    cacheReadPerM: 0,
    contextLength: UNKNOWN_MODEL_CONTEXT,
    supportsTools: true,
    inputModalities: null,
    reasoning: { supportedEfforts: null, defaultEffort: null, mandatory: null },
  };
  const hasPricing = base.inputPerM > 0 || base.outputPerM > 0;
  return {
    ...base,
    contextLength: spec.contextLength ?? base.contextLength,
    inputModalities: spec.inputModalities ?? base.inputModalities,
    maxOutputTokens: spec.maxOutputTokens ?? base.maxOutputTokens,
    reasoning: spec.reasoning ?? base.reasoning,
    // Live prices win when present; curated prices only fill missing data.
    inputPerM: hasPricing ? base.inputPerM : (spec.inputPerM ?? base.inputPerM),
    outputPerM: hasPricing ? base.outputPerM : (spec.outputPerM ?? base.outputPerM),
    cacheReadPerM:
      base.cacheReadPerM > 0 ? base.cacheReadPerM : (spec.cacheReadPerM ?? base.cacheReadPerM),
  };
}

/** Curated spec for a model id (used for offline/unknown lookups). */
export function modelSpecFor(modelId: string): ModelSpecOverride | undefined {
  return KNOWN_MODEL_SPECS[modelId];
}

/** Parse one `/models` catalog entry; null when the entry has no id. */
export function parseCatalogEntry(raw: unknown): CatalogEntry | null {
  if (typeof raw !== 'object' || raw === null) {
    return null;
  }
  const entry = raw as Record<string, unknown>;
  if (typeof entry.id !== 'string' || entry.id.length === 0) {
    return null;
  }
  const pricing = (
    typeof entry.pricing === 'object' && entry.pricing !== null
      ? (entry.pricing as Record<string, unknown>)
      : {}
  ) as Record<string, unknown>;
  const parsed: CatalogEntry = {
    id: entry.id,
    name: typeof entry.name === 'string' ? entry.name : null,
    inputPerM: parsePricePerM(typeof pricing.prompt === 'string' ? pricing.prompt : undefined),
    outputPerM: parsePricePerM(
      typeof pricing.completion === 'string' ? pricing.completion : undefined,
    ),
    cacheReadPerM: parsePricePerM(
      typeof pricing.input_cache_read === 'string' ? pricing.input_cache_read : undefined,
    ),
    maxOutputTokens:
      typeof entry.max_output_tokens === 'number' && entry.max_output_tokens > 0
        ? entry.max_output_tokens
        : undefined,
    contextLength:
      typeof entry.context_length === 'number' && entry.context_length > 0
        ? entry.context_length
        : UNKNOWN_MODEL_CONTEXT,
    supportsTools: supportsTools(entry.supported_parameters),
    inputModalities: parseInputModalities(
      typeof entry.architecture === 'object' && entry.architecture !== null
        ? (entry.architecture as Record<string, unknown>).input_modalities
        : undefined,
    ),
    reasoning: parseReasoning(entry.reasoning),
  };
  return applyModelSpec(parsed);
}

/** Fetch and parse a provider's `/models` catalog (best-effort, 5s timeout). */
export async function fetchCatalog(
  baseUrl: string,
  apiKey: string | undefined,
  signal: AbortSignal,
): Promise<Map<string, CatalogEntry>> {
  const timeout = AbortSignal.timeout(MODELS_FETCH_TIMEOUT_MS);
  const response = await fetch(`${baseUrl}/models`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : undefined,
    signal: AbortSignal.any([signal, timeout]),
  });
  if (!response.ok) {
    throw new Error(
      `models API error ${response.status}: ${(await response.text().catch(() => '')).slice(0, 500)}`,
    );
  }
  const data = (await response.json()) as { data?: unknown[] };
  const catalog = new Map<string, CatalogEntry>();
  for (const raw of data.data ?? []) {
    const entry = parseCatalogEntry(raw);
    if (entry) {
      catalog.set(entry.id, entry);
    }
  }
  return catalog;
}

/** Shape of the persisted catalog file. */
interface CatalogFile {
  version: number;
  fetchedAt: string;
  provider: string;
  models: CatalogEntry[];
}

/** Global per-provider catalog file: <dataHome>/zen-agent/models/<providerId>.catalog.json. */
export function catalogFilePath(providerId: string): string {
  return join(indexDirectory(), 'models', `${providerId}.catalog.json`);
}

/** Read the persisted catalog; null when absent, malformed or outdated. */
export async function readCatalogFile(
  providerId: string,
): Promise<Map<string, CatalogEntry> | null> {
  try {
    const raw = await readFile(catalogFilePath(providerId), 'utf8');
    const parsed = JSON.parse(raw) as CatalogFile;
    if (parsed.version !== CATALOG_CACHE_VERSION || !Array.isArray(parsed.models)) {
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

/** Best-effort: persist the fetched catalog so offline restarts can use it. */
export async function writeCatalogFile(
  providerId: string,
  catalog: Map<string, CatalogEntry>,
): Promise<void> {
  try {
    const payload: CatalogFile = {
      version: CATALOG_CACHE_VERSION,
      fetchedAt: new Date().toISOString(),
      provider: providerId,
      models: [...catalog.values()],
    };
    const target = catalogFilePath(providerId);
    await mkdir(dirname(target), { recursive: true });
    await writeFileAtomic(target, `${JSON.stringify(payload)}\n`);
  } catch {
    // Caching must never break session creation.
  }
}

/** In-memory per-process catalog cache (mirrors pi's dynamic model store). */
const catalogCache = new Map<string, Map<string, CatalogEntry> | null>();

export function getCatalog(providerId: string): Map<string, CatalogEntry> | null {
  return catalogCache.get(providerId) ?? null;
}

export function setCatalog(providerId: string, catalog: Map<string, CatalogEntry> | null): void {
  catalogCache.set(providerId, catalog);
}

/** Test hook: drop the in-memory catalog cache. */
export function resetCatalogCache(): void {
  catalogCache.clear();
}
