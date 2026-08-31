import * as acp from '@agentclientprotocol/sdk';
import { hostname } from 'node:os';
import { Sonyflake } from 'sonyflake';
import { type ProviderId, type StoredSession, type ThinkingEffort } from './storage.js';
import { envPositiveInt } from './env.js';
import {
  getDefaultModel,
  getModelOptions,
  getProviderCurrency,
  getThinkingEfforts,
} from './provider.js';
import { getDefaultProviderId, getProviderDefinitions } from './provider-registry.js';

/** Safety valve for graceful cancel: hard-abort after this long. Unset/0 = wait forever. */
export const GRACEFUL_CANCEL_TIMEOUT_MS = envPositiveInt('ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS', 0);

export const MAX_TURN_STEPS = envPositiveInt('ZEN_AGENT_MAX_TURN_STEPS', 25);

/**
 * Per-session provider selector, built from the provider registry. Every
 * registered provider (built-in DeepSeek/OpenRouter plus any user-defined
 * ZEN_AGENT_PROVIDERS entries) is offered; sessions pick one here (locked
 * once the user sent the first message, like model and thinking effort).
 * New sessions default to ZEN_AGENT_DEFAULT_PROVIDER or deepseek.
 */
export function providerConfigOption(): acp.SessionConfigOption {
  return {
    id: 'provider',
    name: 'Provider',
    description: 'LLM provider used for this session (locked after the first message)',
    category: 'model',
    type: 'select',
    currentValue: getDefaultProviderId(),
    options: getProviderDefinitions().map((def) => ({
      value: def.id,
      name: def.name,
      description: `${def.name} · ${def.baseUrl}`,
    })),
  };
}

/**
 * Model selector for a provider. Discovery providers get the live model
 * catalog (auto-fetched through pi, cached on disk, static list as
 * fallback); static providers get their fixed model list.
 */
export async function modelConfigOption(provider: ProviderId): Promise<acp.SessionConfigOption> {
  const options =
    (await getModelOptions(provider)) ??
    (getProviderDefinitions().find((def) => def.id === provider)?.staticModels ?? []).map(
      (opt) => ({ value: opt.value, name: opt.name, description: opt.description }),
    );
  const description = `Model used for this session on ${getProviderCurrency(provider)}-billed provider "${provider}"`;
  return {
    id: 'model',
    name: 'Model',
    description,
    category: 'model',
    type: 'select',
    currentValue: getDefaultModel(provider),
    options,
  };
}

/** Human labels for every session thinking-effort level. */
const THINKING_EFFORT_OPTIONS: Record<ThinkingEffort, { name: string; description: string }> = {
  off: { name: 'Off', description: 'Disable extended thinking' },
  minimal: { name: 'Minimal', description: 'Use minimal reasoning effort' },
  low: { name: 'Low', description: 'Use low reasoning effort' },
  medium: { name: 'Medium', description: 'Use medium reasoning effort' },
  high: { name: 'High', description: 'Use high reasoning effort' },
  xhigh: { name: 'X-High', description: 'Use extra-high reasoning effort' },
  max: { name: 'Max', description: 'Use maximum reasoning effort' },
};

/** Every valid session effort value (set_config_option accepts all of them). */
export const THINKING_EFFORT_VALUES: readonly ThinkingEffort[] = [
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
];

function thinkingEffortOption(value: ThinkingEffort) {
  const meta = THINKING_EFFORT_OPTIONS[value];
  return { value, name: meta.name, description: meta.description };
}

/**
 * Thinking-effort selector for a session. DeepSeek offers its own vocabulary
 * (`off`/`low`/`high`/`max`); OpenRouter offers `off` plus the selected
 * model's `supported_efforts` allowlist from the catalog (every gateway
 * value when the model/catalog is unknown), sorted ascending. Every tier is
 * offered as itself — `minimal` is a real gateway level that some models
 * list in their allowlist (and even default to), so it is never folded into
 * `low`.
 */
export async function thinkingConfigOption(session: StoredSession) {
  const efforts = await getThinkingEfforts(session.config.provider, session.config.model);
  return {
    id: 'thinking_effort',
    name: 'Thinking Effort',
    description: 'Reasoning effort used by the model',
    category: 'thought_level',
    type: 'select',
    currentValue: session.config.thinkingEffort,
    options: efforts.map(thinkingEffortOption),
  };
}

export const AVAILABLE_COMMANDS: acp.AvailableCommand[] = [
  {
    name: 'prompt',
    description: 'Set a custom system prompt / instructions for this session',
    input: {
      hint: 'custom system prompt or instructions',
    },
  },
  {
    name: 'sandbox',
    description: 'Run every bash tool call inside a bubblewrap sandbox for this session',
    input: {
      hint: 'on | off | (empty for status)',
    },
  },
  {
    name: 'tools',
    description: 'Enable or disable all tools (bash, read_media) for this session',
    input: {
      hint: 'on | off | (empty for status)',
    },
  },
];

/**
 * Sonyflake's default machine-id space is 16 bits (max 0xFFFF); the old fixed
 * constant 0x0d000721 overflowed it and made the constructor throw. Deriving
 * the id from hostname:pid keeps ids unique across containers on one host and
 * across processes in one container. FNV-1a keeps this dependency-free.
 */
function sonyflakeMachineId(): number {
  let hash = 0x811c9dc5;
  const input = `${hostname()}:${process.pid}`;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0) & 0xffff;
}

const sonyflake = new Sonyflake({ machineId: sonyflakeMachineId() });

const BASE62_CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';

function toBase62(bigintNum: bigint): string {
  if (bigintNum === 0n) return '0';
  let result = '';
  let num = bigintNum;
  while (num > 0n) {
    result = BASE62_CHARS[Number(num % 62n)] + result;
    num = num / 62n;
  }
  return result;
}

export function newMessageId(): string {
  const id = sonyflake.nextId(); // Returns a BigInt or stringified BigInt
  const bigintId = BigInt(id);
  return `msg_${toBase62(bigintId)}`; // Example: msg_7zK4X9p2Q
}

/**
 * Local-time startup timestamp in the same shape Zed's terminal artifacts
 * use, e.g. 2026-08-21-23-06-04, so client debug logs sort chronologically
 * and are human-readable next to terminal logs.
 */
export function formatStartupTimestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join('-');
}
