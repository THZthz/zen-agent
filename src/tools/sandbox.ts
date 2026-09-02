import { existsSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { envPositiveInt } from '../util/env.js';

export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** Default cap on terminal output text sent to the model, in bytes. */
export const DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 50_000;

export interface TruncatedTerminalOutput {
  /** The kept text (the tail when truncated, otherwise the original). */
  text: string;
  truncated: boolean;
  /** UTF-8 byte length of the original text. */
  originalBytes: number;
  /** UTF-8 byte length of the kept text. */
  keptBytes: number;
}

/**
 * Keep only the tail of `text` so it fits within `maxBytes` UTF-8 bytes.
 * The cut is adjusted forward to a UTF-8 lead byte so the kept text never
 * contains a split multi-byte sequence (and never exceeds `maxBytes`).
 *
 * Used for the model-visible tool result only: the full output stays at
 * /tmp/zen-agent/<id>.log and in the session database (terminal_calls), so
 * the model can read more with bash whenever it needs to.
 */
export function truncateTerminalOutput(text: string, maxBytes: number): TruncatedTerminalOutput {
  const bytes = Buffer.from(text, 'utf8');
  const originalBytes = bytes.length;
  if (originalBytes <= maxBytes) {
    return { text, truncated: false, originalBytes, keptBytes: originalBytes };
  }
  let start = bytes.length - maxBytes;
  // Continuation bytes are 0b10xxxxxx; skip forward to the next lead byte
  // so we never split a multi-byte character.
  while (start < bytes.length && (bytes[start]! & 0b1100_0000) === 0b1000_0000) {
    start++;
  }
  const kept = bytes.subarray(start).toString('utf8');
  return {
    text: kept,
    truncated: true,
    originalBytes,
    keptBytes: Buffer.byteLength(kept, 'utf8'),
  };
}

/**
 * Model-visible terminal output byte budget. Override with
 * `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` (bytes; must be > 0).
 */
export function terminalOutputByteLimit(): number {
  return envPositiveInt('ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT', DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT);
}

/**
 * Commands the bash tool may not run; their real binaries are shadowed
 * inside the bwrap namespace by `sandboxBlockShim`, which refuses to run
 * and points the agent at the substitute. The host binaries are untouched,
 * so every other script on the machine keeps using the real `rm`, `grep`
 * and `find`.
 */
const BLOCKED_COMMANDS: Record<string, string> = {
  rm: 'trash',
  grep: 'rg',
  find: 'fdfind',
};

/** Standard locations of the blocked binaries (checked at runtime). */
const BLOCKED_BINARY_DIRS = ['/usr/bin', '/bin'];

/**
 * Path to the shim mounted over the blocked binaries. Override with
 * `ZEN_AGENT_SANDBOX_BLOCK_SHIM`; the default resolves relative to this
 * file, so it works both from `src/` (tsx) and `dist/` (compiled).
 */
function sandboxBlockShimPath(): string {
  const override = process.env.ZEN_AGENT_SANDBOX_BLOCK_SHIM;
  if (override !== undefined && override.trim() !== '') {
    return override.trim();
  }
  return fileURLToPath(new URL('../../bin/zen-agent-sandbox-block.sh', import.meta.url));
}

/**
 * `--ro-bind` arguments that replace each blocked binary with the shim
 * inside the sandbox. `/bin` is a symlink to `/usr/bin` on many distros,
 * so destinations are deduplicated by their resolved path. Only existing
 * binaries are shadowed (bwrap refuses to bind onto a missing file).
 */
function blockedBinaryBinds(): string[] {
  const shim = sandboxBlockShimPath();
  const seen = new Set<string>();
  const binds: string[] = [];
  for (const command of Object.keys(BLOCKED_COMMANDS)) {
    for (const dir of BLOCKED_BINARY_DIRS) {
      const dest = join(dir, command);
      if (!existsSync(dest)) continue;
      const resolved = realpathSync(dest);
      if (seen.has(resolved)) continue;
      seen.add(resolved);
      binds.push(`--ro-bind ${shim} ${dest}`);
    }
  }
  return binds;
}

/**
 * Default bubblewrap sandbox used for bash tool calls when
 * `ZEN_AGENT_SANDBOX=1`:
 *   --ro-bind / /   the whole rootfs becomes READ-ONLY: the agent can only
 *                   write to paths explicitly listed in `writablePaths`
 *   --dev /dev      fresh devtmpfs (host /dev is unusable in a userns)
 *   --bind path     each writable path is bind-mounted read-write
 *   --ro-bind shim  `rm`, `grep` and `find` are replaced by a shim that
 *                   refuses to run (use trash/rg/fdfind instead)
 *
 * The sandboxed process runs as the invoking uid in a new user+mount
 * namespace, so it cannot remount anything read-write or escape the
 * namespace. Shadowing the binaries only affects processes inside the
 * namespace: scripts on the host keep using the real `rm`/`grep`/`find`.
 * Override the entire bwrap command with `ZEN_AGENT_SANDBOX_CMD` if a
 * different policy is needed. The session's `/writable` paths are added as
 * `--bind <path> <path>` (see `bashSandboxPrefix`); a missing path fails
 * the whole bwrap invocation loudly (`Can't find source path`) rather than
 * silently dropping the path.
 */
function defaultBashSandbox(writablePaths: string[]): string {
  const bindArgs = writablePaths.map((p) => `--bind ${p} ${p}`);
  return [
    `bwrap --die-with-parent --ro-bind / / --dev /dev`,
    `--bind /dev/pts /dev/pts --tmpfs /dev/shm`,
    ...bindArgs,
    ...blockedBinaryBinds(),
  ].join(' ');
}

export function bashSandboxPrefix(enabled: boolean, writablePaths: string[]): string {
  if (!enabled) {
    return '';
  }
  const custom = process.env.ZEN_AGENT_SANDBOX_CMD;
  if (custom !== undefined && custom.trim() !== '') {
    return `${custom.trim()} `;
  }
  return `${defaultBashSandbox(writablePaths)} `;
}

/**
 * Runs a bash tool call through Zed's ACP terminal API.
 *
 * Zed exposes client-side terminals (`acp::methods.client.terminal.*`) that
 * run in Zed's own PTY on the host; we create one per call, stream its
 * output to Zed as a `tool_call_update` (which Zed renders as a live
 * terminal card), and wait for exit before collecting output. The abort
 * listener below kills the terminal ONLY on a hard abort — a graceful
 * cancel (user follow-up / Stop) lets the command finish.
 *
 * Because these terminals run in Zed (on the host), sandboxing the agent
 * process with bwrap does NOT constrain the bash tool. Every bash call is
 * therefore wrapped in its own bubblewrap invocation (see
 * `bashSandboxPrefix`) whenever sandboxing is enabled — either per session
 * via the `/sandbox` slash command or globally via `ZEN_AGENT_SANDBOX=1`.
 * The sandbox bind-mounts /mnt read-only: the tool can read /mnt but every
 * write to it fails with EROFS. It also shadows `rm`, `grep` and `find`
 * with refusing shims (see `blockedBinaryBinds`), so the agent can only
 * use `trash`, `rg` and `fdfind` for those operations; host scripts are
 * unaffected because the shadowing lives in the bwrap mount namespace.
 */
