import { readFileSync } from 'node:fs';
import type { OpenAICompletionsCompat, ThinkingLevelMap } from '@earendil-works/pi-ai';
import type { ThinkingEffort } from './storage.js';
import { parseDeepSeekBalance, parseOpenRouterBalance } from './provider-balances.js';

/**
 * Provider registry: the single source of truth for which LLM providers
 * exist and how they talk to the wire. DeepSeek and OpenRouter are just two
 * built-in entries; users can add any OpenAI-compatible provider by endpoint
 * + API key (see ZEN_AGENT_PROVIDERS / ZEN_AGENT_PROVIDERS_FILE).
 *
 * Everything pi needs to run a step (baseUrl, auth, compat, effort map) and
 * everything Zen needs for accounting (currency, pricing, balance, model
 * discovery) is expressed as data here, so adding a provider never requires
 * touching the dispatch or config code.
 */

/** Default provider id used when the user has not chosen one. */
export const DEFAULT_PROVIDER_ID = 'deepseek';

/** `ZEN_AGENT_DEFAULT_PROVIDER` overrides the default session provider. */
export function getDefaultProviderId(): string {
  const override = process.env.ZEN_AGENT_DEFAULT_PROVIDER?.trim();
  return override && override.length > 0 ? override : DEFAULT_PROVIDER_ID;
}

/** Balance/credit snapshot for the active provider, in its billing currency. */
export interface BalanceSnapshot {
  isAvailable: boolean;
  currency: string;
  /** Remaining balance in `currency` units. */
  total: number;
  /** Provider-specific extras for the debug log. */
  details: Record<string, unknown>;
}

/** A model the provider offers without needing discovery (static list). */
export interface StaticModelOption {
  value: string;
  name: string;
  description: string;
  /** Context window in tokens; defaults to the provider's default when absent. */
  contextLength?: number;
  /** Static per-1M-token prices (in the provider's currency); defaults to the pricing fallback. */
  cost?: { inputPerM: number; outputPerM: number };
}

/** DeepSeek-style rate table (CNY per 1M tokens) with Beijing peak/off-peak. */
export interface DeepSeekRateTable {
  cacheHit: { peak: number; offPeak: number };
  cacheMiss: { peak: number; offPeak: number };
  output: { peak: number; offPeak: number };
}

/**
 * Where per-model token prices come from:
 * - `catalog`: the discovered `/models` entry's pricing (OpenRouter-style),
 *   falling back to `fallback` when the entry has no prices. Cache hits are
 *   billed at the input rate (OpenRouter's behavior).
 * - `table`: a static per-model table with peak/off-peak rates (DeepSeek).
 * - `fixed`: constant rates for every model (generic/unknown providers).
 */
export type ProviderPricing =
  | { kind: 'catalog'; fallback: { inputPerM: number; outputPerM: number } }
  | { kind: 'table'; rates: Record<string, DeepSeekRateTable>; defaultModel: string }
  | { kind: 'fixed'; cacheHitPerM: number; cacheMissPerM: number; outputPerM: number };

/** Balance endpoint plus a parser for its JSON response. */
export interface ProviderBalance {
  /** Path under the provider base URL, e.g. "/user/balance". */
  path: string;
  parse: (json: unknown, label: string) => BalanceSnapshot;
}

/**
 * How session thinking efforts map to the provider's wire vocabulary:
 * - `static-map`: a fixed table (DeepSeek: minimal→low, ...) plus the effort
 *   values the selector offers.
 * - `allowlist`: per-model `supported_efforts` from the catalog, remapped by
 *   ladder distance (OpenRouter); unknown models accept every value.
 * - `passthrough`: send the session value unchanged as `reasoning_effort`
 *   (generic OpenAI-compatible providers).
 */
export type ProviderEffort =
  | { kind: 'static-map'; map: ThinkingLevelMap; options: readonly ThinkingEffort[] }
  | { kind: 'allowlist'; ladder: readonly string[]; offValue: string }
  | { kind: 'passthrough' };

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
  /** Whether the provider auto-discovers models via GET {baseUrl}/models. */
  discovery: { enabled: boolean };
  /** Static model list (DeepSeek, or fallback entries like openrouter/free). */
  staticModels: readonly StaticModelOption[];
  /** Models pinned to the top of the selector, in order. */
  pinnedModelIds: readonly string[];
  pricing: ProviderPricing;
  balance?: ProviderBalance;
  effort: ProviderEffort;
  /** Pi OpenAI-completions compatibility settings for this provider. */
  compat: OpenAICompletionsCompat;
  /** Extra request fields merged into the chat body (e.g. OpenRouter routing). */
  extraBody?: Record<string, unknown>;
  /** Extra request headers (e.g. OpenRouter's HTTP-Referer / X-Title). */
  extraHeaders?: Record<string, string>;
  /** Send the session id to the provider for sticky context caching. */
  sendSessionId?: boolean;
}

/** OpenRouter's gateway-wide `reasoning_effort` ladder, highest first. */
export const OPENROUTER_EFFORT_LADDER: readonly string[] = [
  'max',
  'xhigh',
  'high',
  'medium',
  'low',
  'minimal',
  'none',
];

/**
 * DeepSeek's `reasoning_effort` vocabulary is `low` / `high` / `max` (no
 * `minimal`/`medium`/`xhigh`). Values outside the vocabulary map per
 * DeepSeek's official table: `minimal`→`low`, `medium`→`high`, `xhigh`→`high`
 * (only `max` maps to `max`). `off` is non-null so pi sends
 * `thinking: {type: "disabled"}` (DeepSeek defaults thinking ON, so omitting
 * the field alone would still think).
 */
export const DEEPSEEK_EFFORT_MAP: ThinkingLevelMap = {
  off: 'disabled',
  minimal: 'low',
  low: 'low',
  medium: 'high',
  high: 'high',
  xhigh: 'high',
  max: 'max',
};

/** DeepSeek selector offers exactly off/low/high/max (its real vocabulary). */
export const DEEPSEEK_EFFORT_OPTIONS: readonly ThinkingEffort[] = ['off', 'low', 'high', 'max'];

/** Official DeepSeek pricing (CNY per 1M tokens) for the V4 models. */
export const DEEPSEEK_RATE_TABLE: Record<string, DeepSeekRateTable> = {
  'deepseek-v4-flash': {
    cacheHit: { peak: 0.1, offPeak: 0.05 },
    cacheMiss: { peak: 3.0, offPeak: 1.5 },
    output: { peak: 9.0, offPeak: 4.5 },
  },
  'deepseek-v4-pro': {
    cacheHit: { peak: 0.3, offPeak: 0.15 },
    cacheMiss: { peak: 9.0, offPeak: 4.5 },
    output: { peak: 27.0, offPeak: 13.5 },
  },
};

function trimTrailingSlash(url: string): string {
  return url.replace(/\/+$/, '');
}

function openRouterProviderSort(): Record<string, unknown> | null {
  const raw = process.env.OPENROUTER_PROVIDER_SORT;
  const sort = raw === undefined ? 'price' : raw.trim();
  return sort.length > 0 ? { sort } : null;
}

function openRouterHeaders(): Record<string, string> | undefined {
  const headers: Record<string, string> = {};
  const siteUrl = process.env.OPENROUTER_SITE_URL;
  if (siteUrl) {
    headers['HTTP-Referer'] = siteUrl;
  }
  const appName = process.env.OPENROUTER_APP_NAME;
  if (appName) {
    headers['X-Title'] = appName;
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

/**
 * Built-in definitions. Env is read when the registry is (re)built, so tests
 * reset the registry to pick up new env values.
 */
function builtinDefinitions(): ProviderDefinition[] {
  return [
    {
      id: 'deepseek',
      name: 'DeepSeek',
      label: 'DeepSeek',
      baseUrl: trimTrailingSlash(process.env.DEEPSEEK_BASE_URL ?? 'https://api.deepseek.com'),
      apiKeyEnv: 'DEEPSEEK_API_KEY',
      defaultModel: process.env.DEEPSEEK_MODEL ?? 'deepseek-v4-flash',
      currency: 'CNY',
      discovery: { enabled: false },
      staticModels: [
        {
          value: 'deepseek-v4-flash',
          name: 'Deepseek V4 Flash',
          description: 'Fast model for everyday coding tasks',
          contextLength: Number(process.env.DEEPSEEK_CONTEXT_WINDOW ?? 1_000_000),
        },
        {
          value: 'deepseek-v4-pro',
          name: 'Deepseek V4 Pro',
          description: 'More powerful model for complex tasks',
          contextLength: Number(process.env.DEEPSEEK_CONTEXT_WINDOW ?? 1_000_000),
        },
      ],
      pinnedModelIds: [],
      pricing: { kind: 'table', rates: DEEPSEEK_RATE_TABLE, defaultModel: 'deepseek-v4-flash' },
      balance: { path: '/user/balance', parse: parseDeepSeekBalance },
      effort: { kind: 'static-map', map: DEEPSEEK_EFFORT_MAP, options: DEEPSEEK_EFFORT_OPTIONS },
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        maxTokensField: 'max_tokens',
        requiresReasoningContentOnAssistantMessages: true,
        thinkingFormat: 'deepseek',
      },
    },
    {
      id: 'openrouter',
      name: 'OpenRouter',
      label: 'OpenRouter',
      baseUrl: trimTrailingSlash(process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1'),
      apiKeyEnv: 'OPENROUTER_API_KEY',
      defaultModel: process.env.OPENROUTER_MODEL ?? 'openrouter/free',
      currency: 'USD',
      discovery: { enabled: true },
      staticModels: [
        {
          value: 'openrouter/free',
          name: 'OpenRouter Free',
          description: "OpenRouter's free-tier routing model",
          contextLength: 128_000,
          cost: { inputPerM: 0, outputPerM: 0 },
        },
      ],
      pinnedModelIds: ['openrouter/free'],
      pricing: { kind: 'catalog', fallback: { inputPerM: 1, outputPerM: 2 } },
      balance: { path: '/auth/key', parse: parseOpenRouterBalance },
      effort: { kind: 'allowlist', ladder: OPENROUTER_EFFORT_LADDER, offValue: 'none' },
      compat: { supportsDeveloperRole: false, thinkingFormat: 'openrouter' },
      extraBody: (() => {
        const body: Record<string, unknown> = {};
        const sort = openRouterProviderSort();
        if (sort) {
          body.provider = sort;
        }
        return body;
      })(),
      extraHeaders: openRouterHeaders(),
      sendSessionId: true,
    },
  ];
}

/** Minimal shape accepted from ZEN_AGENT_PROVIDERS / ZEN_AGENT_PROVIDERS_FILE. */
export interface UserProviderConfig {
  id: string;
  name?: string;
  /** OpenAI-compatible base URL, e.g. "https://api.groq.com/openai/v1". */
  baseUrl: string;
  /** Env var holding the API key, e.g. "GROQ_API_KEY". */
  apiKeyEnv?: string;
  /** Fallback model used when discovery is off or the user sends no model. */
  defaultModel?: string;
  /** Billing currency, default "USD". */
  currency?: string;
  /**
   * Static model list. Omitted (or empty) enables auto-discovery via
   * GET {baseUrl}/models from just the endpoint + API key.
   */
  models?: Array<
    string | { id: string; name?: string; description?: string; contextLength?: number }
  >;
}

function requireString(value: unknown, field: string, context: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`Provider config error (${context}): "${field}" must be a non-empty string`);
  }
  return value.trim();
}

function parseUserProvider(raw: unknown, context: string): ProviderDefinition {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`Provider config error (${context}): each entry must be an object`);
  }
  const entry = raw as Record<string, unknown>;
  const id = requireString(entry.id, 'id', context);
  const baseUrl = trimTrailingSlash(requireString(entry.baseUrl, 'baseUrl', context));
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
        staticModels.push({
          value: model.trim(),
          name: model.trim(),
          description: model.trim(),
        });
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
        staticModels.push({ value: modelId, name: modelName, description, contextLength });
      } else {
        throw new Error(
          `Provider config error (${context}): "models" entries must be strings or objects`,
        );
      }
    }
  }

  const discoveryEnabled = staticModels.length === 0;
  if (defaultModel.length === 0) {
    if (discoveryEnabled) {
      throw new Error(
        `Provider config error (${context}): "defaultModel" is required when models are auto-discovered`,
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
    discovery: { enabled: discoveryEnabled },
    staticModels,
    pinnedModelIds: [],
    pricing: { kind: 'catalog', fallback: { inputPerM: 1, outputPerM: 2 } },
    effort: { kind: 'passthrough' },
    compat: {
      // Generic OpenAI-compatible endpoints default to the OpenAI wire format;
      // pi auto-detects known endpoints (DeepSeek, Together, ...) for the rest.
      supportsDeveloperRole: false,
      thinkingFormat: 'openai',
    },
  };
}

/** Providers from ZEN_AGENT_PROVIDERS (inline JSON) and/or a JSON file. */
function userDefinitions(): ProviderDefinition[] {
  const definitions: ProviderDefinition[] = [];
  const inline = process.env.ZEN_AGENT_PROVIDERS;
  if (inline && inline.trim().length > 0) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(inline);
    } catch (error) {
      throw new Error(
        `Provider config error (ZEN_AGENT_PROVIDERS): invalid JSON — ${(error as Error).message}`,
      );
    }
    if (!Array.isArray(parsed)) {
      throw new Error('Provider config error (ZEN_AGENT_PROVIDERS): expected a JSON array');
    }
    parsed.forEach((entry, index) => {
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
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new Error(
        `Provider config error (ZEN_AGENT_PROVIDERS_FILE): invalid JSON — ${(error as Error).message}`,
      );
    }
    if (!Array.isArray(parsed)) {
      throw new Error('Provider config error (ZEN_AGENT_PROVIDERS_FILE): expected a JSON array');
    }
    parsed.forEach((entry, index) => {
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
 * change (different base URL / key / model / provider sort / user providers)
 * rebuilds the registry — the same behavior the old OpenRouter cache had
 * (keyed by baseUrl|apiKey), now across every provider knob.
 */
export function getProviderEnvKey(): string {
  const names = [
    'ZEN_AGENT_DEFAULT_PROVIDER',
    'ZEN_AGENT_PROVIDERS',
    'ZEN_AGENT_PROVIDERS_FILE',
    'DEEPSEEK_BASE_URL',
    'DEEPSEEK_API_KEY',
    'DEEPSEEK_MODEL',
    'DEEPSEEK_CONTEXT_WINDOW',
    'OPENROUTER_BASE_URL',
    'OPENROUTER_API_KEY',
    'OPENROUTER_MODEL',
    'OPENROUTER_SITE_URL',
    'OPENROUTER_APP_NAME',
    'OPENROUTER_PROVIDER_SORT',
  ];
  return JSON.stringify(names.map((name) => process.env[name] ?? ''));
}

/** All registered provider definitions (built-ins + user-defined). */
export function getProviderDefinitions(): readonly ProviderDefinition[] {
  const key = getProviderEnvKey();
  if (cachedDefinitions === null || cachedKey !== key) {
    const builtins = builtinDefinitions();
    const users = userDefinitions();
    const seen = new Set<string>(builtins.map((def) => def.id));
    for (const def of users) {
      if (seen.has(def.id)) {
        throw new Error(
          `Provider config error: provider id "${def.id}" collides with a built-in provider`,
        );
      }
      seen.add(def.id);
    }
    cachedDefinitions = [...builtins, ...users];
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
      `Unknown LLM provider "${providerId}". Configure it via ZEN_AGENT_PROVIDERS or ZEN_AGENT_PROVIDERS_FILE.`,
    );
  }
  return def;
}

export function isKnownProvider(providerId: string): boolean {
  return getProviderDefinition(providerId) !== undefined;
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
