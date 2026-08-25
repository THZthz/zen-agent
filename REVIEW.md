Review complete. I read every source file (all 24 modules under `src/`, plus configs, bin scripts, package files), ran the test suite (182 tests pass) and `tsc --noEmit` (clean), and verified the most serious finding with a live reproduction. Here is what I found.

# Zen-Agent Code Review

## High severity — likely bugs

- [x] 1. `StreamThrottle` turns a failed emit into a process crash or a hung turn
`src/stream-throttle.ts` — `schedule()` starts the tick with `void this.tick()`. Inside `tick()`, `await this.emit(...)` is not wrapped in try/catch:
- If the emit rejects (e.g., the JSON-RPC connection to Zed drops mid-stream — exactly what `cx.notify` does in `agent.ts`'s callback), the rejection is **unhandled**. On Node ≥ 15 the default behavior terminates the whole agent process. I reproduced this: a throwing emit crashes the runtime with `triggerUncaughtException`.
- Even if the process survived, `this.running` stays `true` forever (the reset lines are after the throw) and the queue is never cleared, so `drain()` — awaited in `runTurn` — would spin/hang indefinitely.

Fix: wrap the tick body in try/catch, always reset `running`/`timer`, decide whether to drop the queue (connection gone) or propagate the error to `drain()`.

- [x] 2. OpenRouter model catalog caches a rejected promise forever
`src/openrouter.ts` — `fetchOpenRouterModels()` stores `{ key, promise }` and returns the cached promise whenever the key matches. If the first fetch fails (offline start, transient 5xx, bad key), the **rejected promise stays cached for the entire process lifetime**. Every later call — pricing (`getOpenRouterModelInfo`), modalities, model selector — silently falls back to static tables even long after the network recovers. Only `resetOpenRouterModelsCache()` (a test hook) clears it. Fix: on rejection, null out `modelsCache` so the next call retries.

- [x] 3. Negative modalities memoized per session
`src/agent.ts` `mediaModalities()` does `active.mediaModalities ??= await getModelModalities(...)`. Combined with issue 2 (and plain transient failures), a single failed lookup at the first message pins `{ image: false, audio: false }` for the whole session: `read_media` is never offered and attached images/audio degrade to placeholder notes, permanently. Consider retrying on failure instead of caching a negative result.

- [x] 4. Race: a new `prompt` mutates history while the aborted turn is still draining
`ZenAgent.prompt()` calls `abortActiveSession()` (which does not await the old `runTurn`'s completion) and then immediately pushes the new user message and saves. The old turn's loop still runs until it reaches its next `signal.aborted` check and can push assistant/tool messages *after* the new user message — interleaved, misordered history (partially masked later by `healMessages` dropping the unpaired leftovers, i.e., losing real content). Similarly, two concurrent `writeSession` calls do whole-file read-modify-write with no serialization; last write wins. An in-flight per-session turn promise (awaited before starting a new turn) would close this.

- [x] 5. Chat timeout budget consumed by client-side rate limiting
`src/llm-client.ts`: the timeout `AbortController` timer is armed *before* `await waitForChatRateLimit(signal)`. With `ZEN_AGENT_CHAT_RPM` low and several queued requests, a request can be killed with "request timed out" before it was ever sent. Arm the timer after the rate-limit wait.

## Medium severity

- [x] 6. **Non-atomic persistence** — `storage.ts` `writeSession()`/`writeIndex()` write directly with `writeFile`. A crash mid-write corrupts `state.json`, which holds the full history including base64 media — an unrecoverable session. Write-to-temp + rename is the standard fix, and matters more here because saves happen multiple times per turn.

- [x] 7. **Unbounded `state.json` growth / redundant storage** — every bash call persists the full output **three times**: `rawOutput.output`, `_meta.terminal_output.data` (same text again), and the on-disk log; media payloads are stored both in `events` and in `llmMessages`. And `save()` rewrites the entire JSON document on every step. Long multimodal sessions make each save multi-MB and turn I/O effectively quadratic.

- [x] 8. **`deleteStoredSession` leaks artifacts** — it removes only `state.json`; `terminals/*.log|sh` and `llm.jsonl` stay on disk forever, and the index forgets the cwd so nothing can find them later. Remove `sessionRootDirectory` recursively.

- [x] 9. **Any prompt starting with `/` is hijacked** — `parseSlashCommand()` treats e.g. `/etc/hosts permissions?` as a slash command and replies "Unknown slash command" instead of sending the message to the model. Require a known command/skill match, or fall through to a normal prompt otherwise.

- [x] 10. **`readResourceLink` has no size cap** (`src/prompt-content.ts`) — `file://` resource links are read fully as utf8 into the conversation, unlike media paths which are capped by `ZEN_AGENT_MAX_MEDIA_BYTES`. A large or binary linked file blows the context (binary as mojibake). Cap it and detect non-text content.

- [x] 11. **SSE CRLF handling breaks at chunk boundaries** — `buffer += decode(value).replace(/\r\n/g, "\n")` is applied per chunk; a `\r\n` split across two network chunks survives normalization and the event never splits on `\n\n`. Normalize after assembling, or split events on a regex tolerant of `\r`.

- [x] 12. **`readStoredSession` trusts the file shape** — `JSON.parse(raw) as StoredSession` with no validation; a truncated/corrupt file (see issue 6) throws a bare TypeError from deep inside instead of a clean "session corrupt" error. Legacy sessions missing `config.sandbox` also silently carry `undefined` typed as `boolean`.

## Maintainability findings

- [x] 13. **Dead code: the whole tokenizer** — `src/tokenizer.ts` (633 lines) plus `data/deepseek-tokenizer.json.gz` (~MBs shipped in the repo) is imported by nothing in production code (no `countTokens`/`encode` callers outside tests; there isn't even a `tokenizer.test.ts`). It also carries stale branding: `ZEN_AGENT_*` is the convention everywhere else, yet this file uses `REASONIX_TOKENIZER_PATH` and resolves a `reasonix/package.json` dependency that isn't in `package.json`. Delete it or wire it up.

- [x] 14. **Misleading currency naming** — `costYuan` / `formatYuan` / `roundYuan` hold **USD** for OpenRouter sessions (`turn-stats.ts`, `storage.ts` SessionUsage, agent logging). Anyone reading `usage.costYuan` will assume CNY. Rename to currency-neutral (`cost`, `formatCost`) now while the persisted-shape compat story is already being managed.

- [x] 15. **Module-global state with hidden coupling**:
    - `rate-limit.ts`: `nextChatRequestAt` is process-global, shared across all sessions/providers, and a request aborted while waiting still consumes its reserved slot.
    - `verifyTurnCost()` keeps one `lastObservedBalance` per process; with two concurrent DeepSeek sessions the balance delta of one session is compared against another's estimated cost (currency guard only separates providers).
    - `openrouter.ts` `modelsPersistedCwd` is a process-global keyed on one cwd.
    These are fine today (single-process ACP agent) but are traps for anyone adding concurrency; at minimum they deserve comments where they exist (some have none).

- [x] 16. **Lint hygiene** — no ESLint/configured lint at all; `tsconfig.json` doesn't enable `noUnusedLocals`. Consequences visible today: `agent.ts` imports `randomBytes`, `mkdir`, `readFile` unused, and has three separate import statements from `./llm-client.js`. Cheap wins.

- [ ] 17. **`healMessages` silent data loss is only console.warn'd** — dropped assistant/tool messages go to stdout, not the session debug log (`logRuntime`). When healing kicks in after a crash you'll want it in `llm.jsonl`/`log.jsonl` correlated with the session, not on a stdout Zed may swallow.

- [ ] 18. **`listStoredSessions(cwd)` reads and JSON-parses every `state.json` in full** (including base64 media) just to produce id/title/date. Store title/updatedAt in the index or sidecar and listing stays O(entries) cheap.

- [ ] 19. **Shutdown is abrupt** — `index.ts` exits on SIGTERM/SIGHUP without letting `runTurn` finish, killing/releasing client terminals, or flushing pending `void`-ed log writes. At least abort controllers and give the connection a moment to close.

- [ ] 20. **Duplication worth consolidating**: env-number parsing (`parseEnvNumber` in deepseek.ts, `parseChatRpm`, `maxMediaBytes`, `parseGracefulCancelTimeoutMs`, `parseMaxTurnSteps`, `terminalOutputByteLimit`, `parseChatTimeoutMs` — seven hand-rolled variants of "int from env with fallback"); usage parsing between `parseDeepSeekUsage`/`parseOpenRouterUsage` (near-identical); provider option objects (`DEEPSEEK_MODEL_CONFIG_OPTION` vs `modelConfigOption` openrouter branch).

## Minor / nits

- [ ] 21. `SYSTEM_PROMPT` typo: "experiened software engineer".
- [ ] 22. `retry.ts`: `Retry-After` HTTP-date format isn't handled (only delta-seconds); the abort listener added in `sleep()` is never removed on normal resolve.
- [ ] 23. `toOpenAiMessages` only returns stored reasoning when the assistant message has tool calls — intentional-looking but subtle; deserves the comment treatment since it interacts with the thinking-mode backfill logic right below.
- [ ] 24. Malformed streamed tool-call JSON falls back to `{ command: partial.arguments }` (`llm-client.ts`) — bash-specific assumption baked into the generic client; for `read_media` it produces a confusing "requires a non-empty string path" error instead of "malformed arguments".
- [ ] 25. `tokenizer.ts` aside, `LruCache.set` evicts even when `limit <= 0` re-inserts… fine in practice, but `new LruCache(0)` degenerates silently.
