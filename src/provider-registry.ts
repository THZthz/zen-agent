import { readFileSync } from 'node:fs';
import type { OpenAICompletionsCompat } from '@earendil-works/pi-ai';

/**
 * Provider registry: the single source of truth for which LLM providers
 * exist. There are NO built-in providers — every provider is defined by the
 * user via ZEN_AGENT_PROVIDERS (inline JSON) or ZEN_AGENT_PROVIDERS_FILE
 * (JSON file):
 *
 *   [{ "id": "deepseek", "name": "DeepSeek", "baseUrl": "https://api.deepseek.com",
 *      "apiKeyEnv": "DEEPSEEK_API_KEY", "defaultModel": "deepseek-v4-flash",
 *      "models": [{ "id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash",
 *                   "contextLength": 1000000 }] }]
 *
 * Models are declared in `models`; set `fetchModels: true` instead to
 * auto-discover them from GET {baseUrl}/models (declared `models` then act
 * as a static baseline that is always offered).
 */

/** Balance/credit snapshot for the active provider, in its billing currency. */
export interface BalanceSnapshot {
  isAvailable: boolean;
  currency: string;
  /** Remaining balance in `currency` units. */
  total: number;
  /** Provider-specific extras for the debug log. */
  details: Record<string, unknown>;
}

/** A model the provider offers (declared statically, or a fallback baseline). */
export interface StaticModelOption {
  value: string;
  name: string;
  description: string;
  /** Context window in tokens; defaults to 200K when absent. */
  contextLength?: number;
  /** Per-1M-token prices in the provider's currency; defaults to the pricing fallback. */
  cost?: { inputPerM: number; outputPerM: number };
  /** Declared input modalities ("image"/"audio"); "text" is implicit. */
  modalities?: string[];
}

/** Balance endpoint plus a parser for its JSON response. */
export interface ProviderBalance {
  /** Path under the provider base URL, e.g. "/user/balance". */
  path: string;
  parse: (json: unknown, label: string) => BalanceSnapshot;
}

/** Everything Zen + pi need to run a provider. */
export interface ProviderDefinition {
  id: string;
  /** Display name, e.g. "DeepSeek". */
  name: string;
  /** Name used in Zen's error messages, e.g. "DeepSeek API error 401: ...". */
  label: string;
  /** Base URL without trailing slash, e.g. "https://api.deepseek.com". */
  baseUrl: string;
  /** Env var holding the API key; absent for keyless local endpoints. */
  apiKeyEnv?: string;
  defaultModel: string;
  /** Billing currency, e.g. "CNY" or "USD". */
  currency: string;
  /** Whether models are auto-discovered from GET {baseUrl}/models (fetchModels). */
  discovery: { enabled: boolean };
  /** Declared model list (always offered, even when discovery is on). */
  staticModels: readonly StaticModelOption[];
  /** Models pinned to the top of the selector, in order. */
  pinnedModelIds: readonly string[];
  /** Per-1M-token price fallback used when a model declares none. */
  pricing: { fallback: { inputPerM: number; outputPerM: number } };
  balance?: ProviderBalance;
  /** Pi OpenAI-completions compatibility settings for this provider. */
  compat: OpenAICompletionsCompat;
  /** Extra request fields merged into the chat body. */
  extraBody?: Record<string, unknown>;
  /** Extra request headers. */
  extraHeaders?: Record<string, string>;
  /** Send the session id to the provider for sticky context caching. */
  sendSessionId?: boolean;
}

/** Minimal shape accepted from ZEN_AGENT_PROVIDERS / ZEN_AGENT_PROVIDERS_FILE. */
export interface UserProviderConfig {
  id: string;
  name?: string;
  /** OpenAI-compatible base URL, e.g. "https://api.deepseek.com". */
  baseUrl: string;
  /** Env var holding the API key, e.g. "DEEPSEEK_API_KEY". */
  apiKeyEnv?: string;
  /** Fallback model used when the session sends no model. */
  defaultModel?: string;
  /** Billing currency, default "USD". */
  currency?: string;
  /**
   * Set true to auto-discover models from GET {baseUrl}/models. When false
   * (default), `models` must be declared.
   */
  fetchModels?: boolean;
  /**
   * Static model list. Each entry can carry `name`, `description`,
   * `contextLength`, `cost` ({inputPerM, outputPerM}) and `modalities`
   * (["image"] / ["audio"]; "text" is implicit). With `fetchModels: true`
   * these are offered in addition to the discovered catalog.
   */
  models?: Array<
    | string
    | {
        id: string;
        name?: string;
        description?: string;
        contextLength?: number;
        cost?: { inputPerM: number; outputPerM: number };
        modalities?: string[];
      }
  >;
}

function requireString(value: unknown, field: string, context: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`Provider config error (${context}): "${field}" must be a non-empty string`);
  }
  return value.trim();
}

function parseModalities(value: unknown, context: string): string[] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (
    !Array.isArray(value) ||
    !value.every((entry) => typeof entry === 'string' && entry.length > 0)
  ) {
    throw new Error(`Provider config error (${context}): "modalities" must be an array of strings`);
  }
  return value as string[];
}

function parseUserProvider(raw: unknown, context: string): ProviderDefinition {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`Provider config error (${context}): each entry must be an object`);
  }
  const entry = raw as Record<string, unknown>;
  const id = requireString(entry.id, 'id', context);
  const baseUrl = requireString(entry.baseUrl, 'baseUrl', context).replace(/\/+$/, '');
  if (entry.apiKeyEnv !== undefined && typeof entry.apiKeyEnv !== 'string') {
    throw new Error(`Provider config error (${context}): "apiKeyEnv" must be a string`);
  }
  const apiKeyEnv = (entry.apiKeyEnv as string | undefined)?.trim() || undefined;
  const name =
    typeof entry.name === 'string' && entry.name.trim().length > 0 ? entry.name.trim() : id;
  const currency =
    typeof entry.currency === 'string' && entry.currency.trim().length > 0
      ? entry.currency.trim()
      : 'USD';
  const fetchModels = entry.fetchModels === true;
  let defaultModel =
    typeof entry.defaultModel === 'string' && entry.defaultModel.trim().length > 0
      ? entry.defaultModel.trim()
      : '';

  const staticModels: StaticModelOption[] = [];
  if (entry.models !== undefined) {
    if (!Array.isArray(entry.models)) {
      throw new Error(`Provider config error (${context}): "models" must be an array`);
    }
    for (const model of entry.models) {
      if (typeof model === 'string') {
        if (model.trim().length === 0) {
          throw new Error(`Provider config error (${context}): model ids must be non-empty`);
        }
        staticModels.push({ value: model.trim(), name: model.trim(), description: model.trim() });
      } else if (typeof model === 'object' && model !== null) {
        const modelId = requireString(
          (model as Record<string, unknown>).id,
          'models[].id',
          context,
        );
        const modelName =
          typeof (model as Record<string, unknown>).name === 'string' &&
          ((model as Record<string, unknown>).name as string).trim().length > 0
            ? ((model as Record<string, unknown>).name as string).trim()
            : modelId;
        const description =
          typeof (model as Record<string, unknown>).description === 'string'
            ? ((model as Record<string, unknown>).description as string).trim()
            : modelName;
        const rawContextLength = (model as Record<string, unknown>).contextLength;
        const contextLength =
          typeof rawContextLength === 'number' && rawContextLength > 0
            ? rawContextLength
            : undefined;
        const rawCost = (model as Record<string, unknown>).cost as
          { inputPerM?: unknown; outputPerM?: unknown } | undefined;
        const cost =
          rawCost && typeof rawCost.inputPerM === 'number' && typeof rawCost.outputPerM === 'number'
            ? { inputPerM: rawCost.inputPerM, outputPerM: rawCost.outputPerM }
            : undefined;
        const modalities = parseModalities(
          (model as Record<string, unknown>).modalities,
          `${context}.models[${modelId}]`,
        );
        staticModels.push({
          value: modelId,
          name: modelName,
          description,
          ...(contextLength !== undefined ? { contextLength } : {}),
          ...(cost !== undefined ? { cost } : {}),
          ...(modalities !== undefined ? { modalities } : {}),
        });
      } else {
        throw new Error(
          `Provider config error (${context}): "models" entries must be strings or objects`,
        );
      }
    }
  }

  if (!fetchModels && staticModels.length === 0) {
    throw new Error(
      `Provider config error (${context}): provider "${id}" declares no models — add a "models" list or set "fetchModels": true to discover them from the endpoint`,
    );
  }
  if (defaultModel.length === 0) {
    if (fetchModels) {
      throw new Error(
        `Provider config error (${context}): "defaultModel" is required when "fetchModels" is true`,
      );
    }
    defaultModel = staticModels[0]!.value;
  }

  return {
    id,
    name,
    label: name,
    baseUrl,
    ...(apiKeyEnv ? { apiKeyEnv } : {}),
    defaultModel,
    currency,
    discovery: { enabled: fetchModels },
    staticModels,
    pinnedModelIds: [],
    pricing: { fallback: { inputPerM: 1, outputPerM: 2 } },
    compat: {
      // Generic OpenAI-compatible endpoints default to the OpenAI wire format;
      // pi auto-detects known endpoints (DeepSeek, Together, ...) for the rest.
      supportsDeveloperRole: false,
      thinkingFormat: 'openai',
    },
  };
}

function parseProviderArray(raw: string, source: string): unknown[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `Provider config error (${source}): invalid JSON — ${(error as Error).message}`,
    );
  }
  if (!Array.isArray(parsed)) {
    throw new Error(`Provider config error (${source}): expected a JSON array`);
  }
  return parsed;
}

/** Providers from ZEN_AGENT_PROVIDERS (inline JSON) and/or a JSON file. */
function userDefinitions(): ProviderDefinition[] {
  const definitions: ProviderDefinition[] = [];
  const inline = process.env.ZEN_AGENT_PROVIDERS;
  if (inline && inline.trim().length > 0) {
    parseProviderArray(inline, 'ZEN_AGENT_PROVIDERS').forEach((entry, index) => {
      definitions.push(parseUserProvider(entry, `ZEN_AGENT_PROVIDERS[${index}]`));
    });
  }
  const filePath = process.env.ZEN_AGENT_PROVIDERS_FILE;
  if (filePath && filePath.trim().length > 0) {
    let raw: string;
    try {
      raw = readFileSync(filePath.trim(), 'utf8');
    } catch (error) {
      throw new Error(
        `Provider config error (ZEN_AGENT_PROVIDERS_FILE): cannot read ${filePath} — ${(error as Error).message}`,
      );
    }
    parseProviderArray(raw, 'ZEN_AGENT_PROVIDERS_FILE').forEach((entry, index) => {
      definitions.push(parseUserProvider(entry, `ZEN_AGENT_PROVIDERS_FILE[${index}]`));
    });
  }
  const seen = new Set<string>();
  for (const def of definitions) {
    if (seen.has(def.id)) {
      throw new Error(`Provider config error: duplicate provider id "${def.id}"`);
    }
    seen.add(def.id);
  }
  return definitions;
}

let cachedDefinitions: readonly ProviderDefinition[] | null = null;
let cachedKey: string | null = null;

/**
 * Env fingerprint for the registry. Definitions read env at build time, so a
 * change (different provider JSON / keys) rebuilds the registry.
 */
export function getProviderEnvKey(): string {
  return JSON.stringify([
    process.env.ZEN_AGENT_PROVIDERS ?? '',
    process.env.ZEN_AGENT_PROVIDERS_FILE ?? '',
    process.env.ZEN_AGENT_DEFAULT_PROVIDER ?? '',
  ]);
}

/** All registered provider definitions (user-defined only). */
export function getProviderDefinitions(): readonly ProviderDefinition[] {
  const key = getProviderEnvKey();
  if (cachedDefinitions === null || cachedKey !== key) {
    cachedDefinitions = userDefinitions();
    cachedKey = key;
  }
  return cachedDefinitions;
}

/** Test hook: drop cached definitions so env changes are picked up. */
export function resetProviderRegistry(): void {
  cachedDefinitions = null;
  cachedKey = null;
}

export function getProviderDefinition(providerId: string): ProviderDefinition | undefined {
  return getProviderDefinitions().find((def) => def.id === providerId);
}

export function requireProviderDefinition(providerId: string): ProviderDefinition {
  const def = getProviderDefinition(providerId);
  if (!def) {
    throw new Error(
      `Unknown LLM provider "${providerId}". Define it via ZEN_AGENT_PROVIDERS or ZEN_AGENT_PROVIDERS_FILE.`,
    );
  }
  return def;
}

export function isKnownProvider(providerId: string): boolean {
  return getProviderDefinition(providerId) !== undefined;
}

/**
 * Default provider for new sessions: ZEN_AGENT_DEFAULT_PROVIDER when set,
 * otherwise the first configured provider. Throws a clear error when no
 * providers are configured at all.
 */
export function getDefaultProviderId(): string {
  const override = process.env.ZEN_AGENT_DEFAULT_PROVIDER?.trim();
  if (override && override.length > 0) {
    if (!isKnownProvider(override)) {
      throw new Error(
        `Provider config error: ZEN_AGENT_DEFAULT_PROVIDER "${override}" is not a configured provider`,
      );
    }
    return override;
  }
  const defs = getProviderDefinitions();
  if (defs.length === 0) {
    throw new Error(
      'No LLM providers configured. Define at least one provider via ZEN_AGENT_PROVIDERS or ZEN_AGENT_PROVIDERS_FILE.',
    );
  }
  return defs[0]!.id;
}

/** Resolve the provider's API key; throws a clear error when required and absent. */
export function resolveApiKey(def: ProviderDefinition): string {
  if (!def.apiKeyEnv) {
    // Keyless local endpoints (Ollama, LM Studio, ...) ignore the header.
    return 'unused';
  }
  const apiKey = process.env[def.apiKeyEnv];
  if (!apiKey) {
    throw new Error(`${def.apiKeyEnv} environment variable is required`);
  }
  return apiKey;
}
