import {
  createModels,
  createProvider,
  envApiKeyAuth,
  type Model,
  type ModelsStore,
  type ModelsStoreEntry,
  type MutableModels,
  type ThinkingLevelMap,
} from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { mkdir, readFile, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { indexDirectory, writeFileAtomic } from './session-index.js';
import {
  getCatalog,
  readCatalogFile,
  setCatalog,
  writeCatalogFile,
  fetchCatalog,
  type CatalogEntry,
} from './provider-catalog.js';
import {
  getProviderDefinitions,
  getProviderEnvKey,
  requireProviderDefinition,
  type ProviderDefinition,
  type StaticModelOption,
} from './provider-registry.js';
import type { ThinkingEffort } from './storage.js';

/**
 * pi-ai integration: a `Models` collection where every registered provider
 * (built-in or user-defined) is built with pi's `createProvider`. Dynamic
 * providers discover their models through `fetchModels` (GET /models), pi
 * restores/persists the catalog through a `ModelsStore`, and Zen reads
 * metadata back out of pi `Model` objects (context, cost, modalities, effort
 * map) for selectors, pricing and streaming.
 */

const CATALOG_STORE_VERSION = 5;

function piModelsPath(providerId: string): string {
  return join(indexDirectory(), 'models', `${providerId}.pi.json`);
}

/** pi ModelsStore backed by a global per-provider file (offline restore). */
class FileModelsStore implements ModelsStore {
  async read(providerId: string): Promise<ModelsStoreEntry | undefined> {
    try {
      const raw = await readFile(piModelsPath(providerId), 'utf8');
      const parsed = JSON.parse(raw) as { version: number } & ModelsStoreEntry;
      if (parsed.version !== CATALOG_STORE_VERSION || !Array.isArray(parsed.models)) {
        return undefined;
      }
      return parsed;
    } catch {
      return undefined;
    }
  }

  async write(providerId: string, entry: ModelsStoreEntry): Promise<void> {
    const target = piModelsPath(providerId);
    const payload = { version: CATALOG_STORE_VERSION, ...entry };
    await mkdir(dirname(target), { recursive: true });
    await writeFileAtomic(target, `${JSON.stringify(payload)}\n`);
  }

  async delete(providerId: string): Promise<void> {
    await unlink(piModelsPath(providerId)).catch(() => {});
  }
}

let piModels: MutableModels | null = null;
let piModelsKey: string | null = null;
const refreshedProviders = new Set<string>();

/** Test hook: drop the pi collection, refresh state and catalog caches. */
export function resetPiModels(): void {
  piModels = null;
  piModelsKey = null;
  refreshedProviders.clear();
}

/** The process-global pi Models collection, built lazily from the registry. */
export function getPiModels(): MutableModels {
  const key = getProviderEnvKey();
  if (!piModels || piModelsKey !== key) {
    const collection = createModels({ modelsStore: new FileModelsStore() });
    for (const def of getProviderDefinitions()) {
      collection.setProvider(
        createProvider({
          id: def.id,
          name: def.name,
          baseUrl: def.baseUrl,
          auth: {
            apiKey: def.apiKeyEnv
              ? envApiKeyAuth(def.name, [def.apiKeyEnv])
              : { name: def.name, resolve: async () => ({ auth: {} }) },
          },
          models: staticPiModels(def),
          ...(def.discovery.enabled ? { fetchModels: (ctx) => fetchPiModels(def, ctx) } : {}),
          api: openAICompletionsApi(),
        }),
      );
    }
    piModels = collection;
    piModelsKey = key;
    // A rebuilt collection starts with no discovery state: re-fetch catalogs.
    refreshedProviders.clear();
  }
  return piModels;
}

function defaultContext(def: ProviderDefinition): number {
  return def.discovery.enabled ? 200_000 : 1_000_000;
}

function fallbackCost(def: ProviderDefinition): {
  inputPerM: number;
  outputPerM: number;
  cacheHitPerM: number;
} {
  if (def.pricing.kind === 'catalog') {
    return {
      inputPerM: def.pricing.fallback.inputPerM,
      outputPerM: def.pricing.fallback.outputPerM,
      cacheHitPerM: def.pricing.fallback.inputPerM,
    };
  }
  if (def.pricing.kind === 'fixed') {
    return {
      inputPerM: def.pricing.cacheMissPerM,
      outputPerM: def.pricing.outputPerM,
      cacheHitPerM: def.pricing.cacheHitPerM,
    };
  }
  return { inputPerM: 0, outputPerM: 0, cacheHitPerM: 0 };
}

function entryCost(
  def: ProviderDefinition,
  entry: CatalogEntry | null,
): { inputPerM: number; outputPerM: number; cacheHitPerM: number } {
  if (entry && (entry.inputPerM > 0 || entry.outputPerM > 0)) {
    return {
      inputPerM: entry.inputPerM,
      outputPerM: entry.outputPerM,
      cacheHitPerM: entry.inputPerM,
    };
  }
  return fallbackCost(def);
}

/**
 * Map a session effort to a provider value by ladder distance (ties resolve
 * toward the HIGHER effort). Mirrors OpenRouter's own remap behavior for
 * models whose `supported_efforts` lacks the requested tier.
 */
function mapOpenRouterEffort(
  effort: ThinkingEffort,
  ladder: readonly string[],
  supportedEfforts: readonly string[] | null,
  mandatory: boolean,
  offValue: string,
): string | null {
  if (supportedEfforts === null) {
    // Unknown model/catalog: every gateway value is accepted, `off` omits
    // the field (the provider picks its default).
    return effort === 'off' ? null : effort;
  }
  if (effort === 'off') {
    if (supportedEfforts.includes(offValue)) {
      return offValue;
    }
    if (mandatory && supportedEfforts.length > 0) {
      return supportedEfforts[supportedEfforts.length - 1]!;
    }
    return null;
  }
  if (supportedEfforts.includes(effort)) {
    return effort;
  }
  const requested = ladder.indexOf(effort);
  if (requested === -1) {
    return null;
  }
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of supportedEfforts) {
    const index = ladder.indexOf(candidate);
    if (index === -1) {
      continue;
    }
    const distance = Math.abs(index - requested);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
}

/** Build a pi `thinkingLevelMap` from the definition + catalog entry. */
export function buildEffortMap(
  def: ProviderDefinition,
  entry: CatalogEntry | null,
): ThinkingLevelMap | undefined {
  if (def.effort.kind === 'static-map') {
    return def.effort.map;
  }
  if (def.effort.kind === 'passthrough') {
    return undefined;
  }
  // allowlist (OpenRouter-style): remap every session level, honor `off`.
  const supported = entry?.reasoning.supportedEfforts ?? null;
  const mandatory = entry?.reasoning.mandatory === true;
  const ladder = def.effort.ladder;
  const map: ThinkingLevelMap = {};
  for (const effort of ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const) {
    const mapped = mapOpenRouterEffort(effort, ladder, supported, mandatory, def.effort.offValue);
    if (mapped !== null) {
      map[effort] = mapped;
    }
  }
  map.off = mapOpenRouterEffort('off', ladder, supported, mandatory, def.effort.offValue);
  return map;
}

/** Convert a catalog entry (or null for an unknown slug) to a pi model. */
export function catalogToPiModel(
  def: ProviderDefinition,
  entry: CatalogEntry | null,
  modelId: string = entry?.id ?? '',
  modelName: string = entry?.name ?? modelId,
): Model<'openai-completions'> {
  const cost = entryCost(def, entry);
  const image = entry?.inputModalities?.includes('image') ?? false;
  return {
    id: modelId,
    name: modelName,
    api: 'openai-completions',
    provider: def.id,
    baseUrl: def.baseUrl,
    reasoning: true,
    input: ['text', ...(image ? ['image'] : [])] as ('text' | 'image')[],
    cost: {
      input: cost.inputPerM,
      output: cost.outputPerM,
      cacheRead: cost.cacheHitPerM,
      cacheWrite: cost.cacheHitPerM,
    },
    contextWindow: entry?.contextLength ?? defaultContext(def),
    maxTokens: 384_000,
    compat: def.compat,
    thinkingLevelMap: buildEffortMap(def, entry),
    ...(def.extraHeaders ? { headers: def.extraHeaders } : {}),
  };
}

function staticModelToPiModel(
  def: ProviderDefinition,
  opt: StaticModelOption,
): Model<'openai-completions'> {
  const fallback = fallbackCost(def);
  const inputPerM = opt.cost?.inputPerM ?? fallback.inputPerM;
  const outputPerM = opt.cost?.outputPerM ?? fallback.outputPerM;
  return {
    id: opt.value,
    name: opt.name,
    api: 'openai-completions',
    provider: def.id,
    baseUrl: def.baseUrl,
    reasoning: true,
    input: ['text'],
    cost: {
      input: inputPerM,
      output: outputPerM,
      cacheRead: opt.cost ? inputPerM : fallback.cacheHitPerM,
      cacheWrite: opt.cost ? inputPerM : fallback.cacheHitPerM,
    },
    contextWindow: opt.contextLength ?? defaultContext(def),
    maxTokens: 384_000,
    compat: def.compat,
    thinkingLevelMap: buildEffortMap(def, null),
    ...(def.extraHeaders ? { headers: def.extraHeaders } : {}),
  };
}

function staticPiModels(def: ProviderDefinition): Model<'openai-completions'>[] {
  return def.staticModels.map((opt) => staticModelToPiModel(def, opt));
}

/** pi's dynamic-model hook: fetch /models, cache Zen metadata, return pi models. */
async function fetchPiModels(
  def: ProviderDefinition,
  ctx: { signal: AbortSignal },
): Promise<Model<'openai-completions'>[]> {
  const apiKey = def.apiKeyEnv ? process.env[def.apiKeyEnv] : undefined;
  const catalog = await fetchCatalog(def.baseUrl, apiKey, ctx.signal);
  setCatalog(def.id, catalog);
  await writeCatalogFile(def.id, catalog);
  return [...catalog.values()].map((entry) => catalogToPiModel(def, entry));
}

/**
 * Make sure a discovery provider's catalog has been fetched (or restored from
 * disk) at least once this process. Best-effort: failures keep the static
 * fallbacks / persisted catalog, exactly like the old OpenRouter cache.
 */
export async function ensureProviderRefreshed(providerId: string): Promise<void> {
  const def = requireProviderDefinition(providerId);
  if (!def.discovery.enabled || refreshedProviders.has(providerId)) {
    return;
  }
  refreshedProviders.add(providerId);
  try {
    await getPiModels().refresh({
      providers: [providerId],
      signal: AbortSignal.timeout(5_000),
    });
  } catch {
    // Static fallbacks / persisted catalog remain available.
  }
  if (!getCatalog(providerId)) {
    setCatalog(providerId, await readCatalogFile(providerId));
  }
}

/**
 * Sync pi model lookup; unknown slugs on discovery providers are synthesized
 * from the catalog (or conservative defaults) so any slug stays usable.
 */
export function getPiModel(providerId: string, modelId: string): Model<'openai-completions'> {
  const def = requireProviderDefinition(providerId);
  const found = getPiModels().getModel(providerId, modelId);
  if (found) {
    return found as Model<'openai-completions'>;
  }
  const entry = getCatalog(providerId)?.get(modelId) ?? null;
  return catalogToPiModel(def, entry, modelId);
}

/** Selector option for a discovered/static model. */
export interface ModelOption {
  value: string;
  name: string;
  description: string;
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

/**
 * Model choices for the session selector. Discovery providers return the
 * live catalog (tool-capable models; when the catalog is unavailable the
 * persisted/static list is used); static providers return their fixed list.
 * Returns null when no catalog exists yet (caller falls back to static).
 */
export async function getModelOptions(providerId: string): Promise<ModelOption[] | null> {
  const def = requireProviderDefinition(providerId);
  if (!def.discovery.enabled) {
    return def.staticModels.map((opt) => ({
      value: opt.value,
      name: opt.name,
      description: opt.description,
    }));
  }
  await ensureProviderRefreshed(providerId);
  const all = getPiModels().getModels(providerId);
  if (all.length === 0) {
    return null;
  }
  const catalog = getCatalog(providerId);
  const toolCapable =
    catalog === null
      ? null
      : new Set(
          [...catalog.values()].filter((entry) => entry.supportsTools).map((entry) => entry.id),
        );
  // Static baseline models (e.g. openrouter/free) are always offered, even
  // when a fetched catalog does not list them.
  const staticIds = new Set(def.staticModels.map((opt) => opt.value));
  const options: ModelOption[] = all
    .filter((model) => staticIds.has(model.id) || toolCapable === null || toolCapable.has(model.id))
    .map((model) => ({
      value: model.id,
      name: model.name,
      description: `${model.name} · ${formatContext(model.contextWindow)}`,
    }));
  options.sort((a, b) => a.value.localeCompare(b.value, 'en'));
  for (let i = def.pinnedModelIds.length - 1; i >= 0; i--) {
    const pinned = def.pinnedModelIds[i]!;
    const index = options.findIndex((option) => option.value === pinned);
    if (index > i) {
      options.unshift(...options.splice(index, 1));
    }
  }
  return options;
}

const FULL_EFFORT_ORDER: readonly ThinkingEffort[] = [
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
];

/**
 * Thinking-effort values the selector offers for a provider/model: the
 * provider's static vocabulary, the model's catalog allowlist (plus off), or
 * the full ladder for unknown/generic models.
 */
export async function getSupportedThinkingEfforts(
  providerId: string,
  modelId: string,
): Promise<readonly ThinkingEffort[]> {
  const def = requireProviderDefinition(providerId);
  if (def.effort.kind === 'static-map') {
    return def.effort.options;
  }
  if (def.effort.kind === 'passthrough') {
    return FULL_EFFORT_ORDER;
  }
  await ensureProviderRefreshed(providerId);
  const entry = getCatalog(providerId)?.get(modelId) ?? null;
  const supported = entry?.reasoning.supportedEfforts ?? null;
  if (supported === null) {
    const model = getPiModel(providerId, modelId);
    const mapped = model.thinkingLevelMap;
    if (mapped) {
      return FULL_EFFORT_ORDER.filter(
        (effort) => effort === 'off' || (mapped[effort] !== undefined && mapped[effort] !== null),
      );
    }
    return FULL_EFFORT_ORDER;
  }
  return FULL_EFFORT_ORDER.filter((effort) => effort === 'off' || supported.includes(effort));
}
