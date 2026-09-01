import {
  createModels,
  createProvider,
  envApiKeyAuth,
  type Model,
  type ModelsStore,
  type ModelsStoreEntry,
  type MutableModels,
} from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { mkdir, readFile, unlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { indexDirectory, writeFileAtomic } from '../session/data-dir.js';
import {
  fetchCatalog,
  getCatalog,
  readCatalogFile,
  setCatalog,
  writeCatalogFile,
  type CatalogEntry,
} from './catalog.js';
import {
  getProviderDefinitions,
  getProviderEnvKey,
  requireProviderDefinition,
  THINKING_EFFORT_VALUES,
  type ProviderDefinition,
  type StaticModelOption,
} from './registry.js';
import type { ThinkingEffort } from '../session/storage.js';

/**
 * pi-ai integration: a `Models` collection where every configured provider
 * (there are no built-ins — all providers come from ZEN_AGENT_PROVIDERS /
 * ZEN_AGENT_PROVIDERS_FILE) is built with pi's `createProvider`. Providers
 * with `fetchModels: true` discover their models through `fetchModels`
 * (GET /models); pi restores/persists the catalog through a `ModelsStore`,
 * and Zen reads metadata back out of pi `Model` objects (context, cost,
 * modalities) for selectors, pricing and streaming.
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

const DEFAULT_CONTEXT_WINDOW = 200_000;

function modelCost(
  def: ProviderDefinition,
  entry: CatalogEntry | null,
  declared: StaticModelOption | null,
): { inputPerM: number; outputPerM: number; cacheReadPerM: number } {
  if (declared?.cost) {
    return {
      inputPerM: declared.cost.inputPerM,
      outputPerM: declared.cost.outputPerM,
      cacheReadPerM: declared.cost.inputPerM,
    };
  }
  if (entry && (entry.inputPerM > 0 || entry.outputPerM > 0)) {
    return {
      inputPerM: entry.inputPerM,
      outputPerM: entry.outputPerM,
      cacheReadPerM: entry.cacheReadPerM > 0 ? entry.cacheReadPerM : entry.inputPerM,
    };
  }
  return {
    inputPerM: def.pricing.fallback.inputPerM,
    outputPerM: def.pricing.fallback.outputPerM,
    cacheReadPerM: def.pricing.fallback.inputPerM,
  };
}

function piModel(
  def: ProviderDefinition,
  modelId: string,
  modelName: string,
  entry: CatalogEntry | null,
  declared: StaticModelOption | null,
): Model<'openai-completions'> {
  const cost = modelCost(def, entry, declared);
  const modalities = declared?.modalities ?? entry?.inputModalities ?? ['text'];
  return {
    id: modelId,
    name: modelName,
    api: 'openai-completions',
    provider: def.id,
    baseUrl: def.baseUrl,
    reasoning: true,
    input: ['text', ...(modalities.includes('image') ? ['image'] : [])] as ('text' | 'image')[],
    cost: {
      input: cost.inputPerM,
      output: cost.outputPerM,
      cacheRead: cost.cacheReadPerM,
      cacheWrite: cost.cacheReadPerM,
    },
    contextWindow: declared?.contextLength ?? entry?.contextLength ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: 384_000,
    compat: def.compat,
    ...(def.extraHeaders ? { headers: def.extraHeaders } : {}),
  };
}

function staticPiModels(def: ProviderDefinition): Model<'openai-completions'>[] {
  return def.staticModels.map((opt) => piModel(def, opt.value, opt.name, null, opt));
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
  return [...catalog.values()].map((entry) =>
    piModel(def, entry.id, entry.name ?? entry.id, entry, null),
  );
}

/**
 * Make sure a discovery provider's catalog has been fetched (or restored from
 * disk) at least once this process. Best-effort: failures keep the static
 * baseline / persisted catalog.
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
    // Static baseline / persisted catalog remain available.
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
  const declared = def.staticModels.find((opt) => opt.value === modelId) ?? null;
  const entry = getCatalog(providerId)?.get(modelId) ?? null;
  return piModel(def, modelId, declared?.name ?? entry?.name ?? modelId, entry, declared);
}

/** Selector option for a discovered/declared model. */
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
 * Model choices for the session selector. Static providers return their
 * declared list; discovery providers return the fetched catalog (when the
 * catalog is unavailable the declared/restored list is used). Returns null
 * when a discovery provider has neither catalog nor declared models.
 */
export async function getModelOptions(providerId: string): Promise<ModelOption[] | null> {
  const def = requireProviderDefinition(providerId);
  const declared = def.staticModels.map((opt) => ({
    value: opt.value,
    name: opt.name,
    description: opt.contextLength
      ? `${opt.description} · ${formatContext(opt.contextLength)}`
      : opt.description,
  }));
  if (!def.discovery.enabled) {
    return declared;
  }
  await ensureProviderRefreshed(providerId);
  const all = getPiModels().getModels(providerId);
  if (all.length === 0) {
    return declared.length > 0 ? declared : null;
  }
  const catalog = getCatalog(providerId);
  const toolCapable =
    catalog === null
      ? null
      : new Set(
          [...catalog.values()].filter((entry) => entry.supportsTools).map((entry) => entry.id),
        );
  const declaredIds = new Set(def.staticModels.map((opt) => opt.value));
  const options: ModelOption[] = all
    .filter(
      (model) => declaredIds.has(model.id) || toolCapable === null || toolCapable.has(model.id),
    )
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

/**
 * Thinking-effort values the session selector offers for a provider/model:
 * the model's declared `thinkingEfforts` when present (in declared order),
 * otherwise the full ladder.
 */
export async function getSupportedThinkingEfforts(
  providerId: string,
  modelId: string,
): Promise<readonly ThinkingEffort[]> {
  const def = requireProviderDefinition(providerId);
  const declared = def.staticModels.find((opt) => opt.value === modelId)?.thinkingEfforts;
  return declared && declared.length > 0 ? declared : THINKING_EFFORT_VALUES;
}
