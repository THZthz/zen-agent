import * as acp from '@agentclientprotocol/sdk';
import { hostname } from 'node:os';
import { Sonyflake } from 'sonyflake';
import {
  DEFAULT_DEEPSEEK_MODEL,
  DEFAULT_PROVIDER,
  type ProviderId,
  type StoredSession,
  type ThinkingEffort,
} from './storage.js';
import { envPositiveInt } from './env.js';
import { getDefaultModel } from './provider.js';
import { getOpenRouterModelOptions, getOpenRouterReasoning } from './openrouter-models.js';
import { OPENROUTER_EFFORT_VALUES } from './openrouter.js';

/** Safety valve for graceful cancel: hard-abort after this long. Unset/0 = wait forever. */
export const GRACEFUL_CANCEL_TIMEOUT_MS = envPositiveInt('ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS', 0);

export const MAX_TURN_STEPS = envPositiveInt('ZEN_AGENT_MAX_TURN_STEPS', 25);

/**
 * Per-session provider selector. DeepSeek and OpenRouter can be used side by
 * side: each session picks its provider here (locked once the user sent the
 * first message, like model and thinking effort). New sessions default to
 * deepseek.
 */
export const PROVIDER_CONFIG_OPTION = {
  id: 'provider',
  name: 'Provider',
  description: 'LLM provider used for this session (locked after the first message)',
  category: 'model',
  type: 'select',
  currentValue: DEFAULT_PROVIDER,
  options: [
    {
      value: 'deepseek',
      name: 'DeepSeek',
      description: "DeepSeek's own API, billed in CNY",
    },
    {
      value: 'openrouter',
      name: 'OpenRouter',
      description: 'OpenRouter model aggregator, billed in USD',
    },
  ],
};

const DEEPSEEK_MODEL_CONFIG_OPTION = {
  id: 'model',
  name: 'Model',
  description: 'Deepseek model used for this session',
  category: 'model',
  type: 'select',
  currentValue: DEFAULT_DEEPSEEK_MODEL,
  options: [
    {
      value: 'deepseek-v4-flash',
      name: 'Deepseek V4 Flash',
      description: 'Fast model for everyday coding tasks',
    },
    {
      value: 'deepseek-v4-pro',
      name: 'Deepseek V4 Pro',
      description: 'More powerful model for complex tasks',
    },
  ],
};

/**
 * OpenRouter models offered in the session selector. Any OpenRouter model
 * slug can be used beyond this list via OPENROUTER_MODEL or
 * session/set_config_option; `openrouter/free` routes to OpenRouter's
 * free-tier models.
 */
const OPENROUTER_MODEL_OPTIONS: Array<{
  value: string;
  name: string;
  description: string;
}> = [
  {
    value: 'openrouter/free',
    name: 'OpenRouter Free',
    description: "OpenRouter's free-tier routing model",
  },
];

/**
 * Model selector for a provider. OpenRouter sessions get the live model
 * catalog (auto-fetched, cached on disk, static list as fallback); DeepSeek
 * sessions get the two fixed models.
 */
export async function modelConfigOption(provider: ProviderId, cwd: string) {
  if (provider === 'openrouter') {
    const options = (await getOpenRouterModelOptions(cwd)) ?? OPENROUTER_MODEL_OPTIONS;
    return {
      id: 'model',
      name: 'Model',
      description: 'OpenRouter model used for this session',
      category: 'model',
      type: 'select',
      currentValue: getDefaultModel('openrouter'),
      options,
    };
  }
  return {
    ...DEEPSEEK_MODEL_CONFIG_OPTION,
    currentValue: DEFAULT_DEEPSEEK_MODEL,
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

/**
 * Effort selector order: ascending (off < minimal < low < medium < high <
 * xhigh < max). Every session value is offered; OpenRouter models that do
 * not support a tier simply omit it from the selector (unknown models get
 * the full gateway ladder).
 */
const THINKING_EFFORT_ORDER: readonly ThinkingEffort[] = [
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
  let efforts: readonly ThinkingEffort[] = ['off', 'low', 'high', 'max'];
  if (session.config.provider === 'openrouter') {
    const reasoning = await getOpenRouterReasoning(session.config.model, session.cwd);
    const supported = reasoning.supportedEfforts ?? OPENROUTER_EFFORT_VALUES;
    efforts = THINKING_EFFORT_ORDER.filter(
      (effort) => effort === 'off' || supported.includes(effort),
    );
  }
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
