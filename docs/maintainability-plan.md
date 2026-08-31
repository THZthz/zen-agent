# Maintainability Remediation Plan

Follow-up to the code review of 2026-08-31. The review's correctness findings are
already fixed and committed (`5670e76 Fix session and provider lifecycle defects`).
This plan covers the five maintainability concerns the review raised, so they can
be executed one concern per commit with no behavior change unless stated.

Guiding rules:

- One concern per commit, `npm test` / `npm run typecheck` / `npm run build` /
  `npm run format:check` green before each commit.
- Prefer deleting an abstraction over rewiring it when production never uses it.
- No file-size targets for their own sake; extract only units with a single clear
  ownership (accounting, command handling, terminal lifecycle, wire conversion).

## Status overview

| #   | Concern                                               | Status            |
| --- | ----------------------------------------------------- | ----------------- |
| 1   | Persisted histories only array-checked (`storage.ts`) | Open              |
| 2   | `StreamThrottle` had no close/discard                 | Done in `5670e76` |
| 3   | Duplicate retry policy (`retry.ts` vs Pi)             | Open              |
| 4   | `replay.ts` mutates persisted `_meta`                 | Open              |
| 5   | Distributed lifecycle ownership / oversized modules   | Open (phased)     |

---

## 1. Element-level validation of persisted histories

**Problem.** `normalizeStoredSession` (`src/storage.ts`) accepts any array for
`events`, `llmMessages`, `turnStats` and `cacheDiagnostics` and casts it. A `null`
element, or a message with a missing `role`, survives loading and crashes later:
`prepareReplayEvents` dereferences `event.sessionUpdate`, environment-message
checks dereference `message.role`. The promised "one bad field degrades to its
default" contract holds for scalars but not for collection elements.

**Approach.** Keep loading permissive; drop structurally unusable elements instead
of failing the whole session (a load failure blocks resume entirely, which is
worse than losing one malformed entry). Add two pure normalizers in `storage.ts`:

- `normalizeEvents(raw: unknown): SessionUpdate[]` — keep only objects whose
  `sessionUpdate` is a non-empty string. Optionally cross-check against the known
  discriminator set and drop unknown kinds.
- `normalizeLlmMessages(raw: unknown): LlmMessage[]` — per element:
  - must be an object with `role` of `user` | `assistant` | `tool`;
  - `user`: `content` is a string, or an array of parts whose `type` is
    `text` | `image` | `audio` (drop invalid parts; drop the message if nothing
    survives and it has no usable content); `name`, when present, must be a
    string or is dropped;
  - `assistant`: `content` must be an array of `text` | `reasoning` |
    `tool-call` parts (`tool-call` requires a string `toolCallId`); drop invalid
    parts, drop the message if empty;
  - `tool`: `content` must be an array of `tool-result` parts requiring a string
    `toolCallId`; otherwise drop the message.
  - Apply the same element pass to `turnStats` (drop non-object entries) and
    `cacheDiagnostics` (ditto).

Dropped assistant tool-calls can orphan tool messages; that is already tolerated
at request time by `healMessages` (`src/heal.ts`), so no pairing repair is needed
here.

**Reporting.** Normalizers stay pure; have each return the cleaned list plus a
dropped-count, and let `readStoredSession` surface the total through the existing
runtime logger at the call site in `agent-session.ts` (load/resume already log
there), so silent data loss is visible in `log.jsonl` without threading a logger
into storage.

**Files.** `src/storage.ts`; tests in `src/storage-validate.test.ts`.

**Tests to add.**

- `events: [null, {sessionUpdate: 'user_message_chunk', ...}]` loads, keeps the
  valid event, drops `null`.
- `llmMessages: [null, {role: 'bogus'}, valid user string message]` keeps only
  the valid one.
- Assistant message with one valid and one invalid part keeps the valid part.
- Tool message whose `content` is not an array is dropped.
- Healthy session round-trips byte-identically through `writeSession` →
  `readStoredSession` (guard against accidental normalization churn, important
  for the provider cache prefix).
- Dropped-entry count reaches the runtime log line on load.

**Risks.** Over-eager dropping could alter history; the byte-identical
round-trip test is the guard. Keep validators syntactic (shape), never semantic.

**Acceptance.** No `Array.isArray` bare cast remains for these four fields;
malformed-element fixtures load without throwing; full suite green.

---

## 2. `StreamThrottle` close/discard — already resolved

`discard()` was added in `5670e76` together with `agent-turn.ts` calling it when
an LLM step rejects, plus a test asserting queued chunks are dropped and later
pushes stay inert. No further work.

Residual verification only: confirm no other `StreamThrottle` consumer is created
outside `agent-turn.ts` (none today); if one appears, it must own `discard()` on
its failure path.

---

## 3. Single retry owner — retire `src/retry.ts`

**Problem.** `RetryOptions` advertises `initialBackoffMs`, `retryableStatuses`,
`signal` and `onRetry`, but the production path (`runChatCompletions` →
`streamOpenAiCompletions`) forwards only `maxAttempts` → `maxRetries` and
`maxBackoffMs` → `maxRetryDelayMs`. `fetchWithRetry` is exercised solely by its
own tests. Two retry policies, one of them dead, and coverage that implies more
behavior than exists.

**Decision.** Pi owns retries. Delete the dead abstraction rather than routing
requests through `fetchWithRetry` (the alternative — custom `fetch` injection
plus `maxRetries: 0` — would re-plumb streaming error semantics for no gain).

**Steps.**

1. Replace `retry?: RetryOptions` in `ChatCompletionsOptions`
   (`src/chat-completions.ts`) with the two fields actually honored, named as
   they map onto Pi: `maxAttempts?: number` (default 4) and
   `maxRetryDelayMs?: number`. Update the doc comments to state that the
   provider library owns retryable-status selection, `Retry-After` handling and
   mid-stream non-retry.
2. Grep confirms nothing outside tests constructs `retry:` today, so no caller
   migration is needed; `deepseek.ts` / `openrouter.ts` need no change.
3. Delete `src/retry.ts` and `src/retry.test.ts`.
4. If a pure mapping is still wanted for testability (`maxAttempts` →
   `Math.max(0, attempts - 1)`), extract it as a two-line exported helper next
   to the options and unit-test that; otherwise no new tests.

**Files.** `src/chat-completions.ts`; delete `src/retry.ts`, `src/retry.test.ts`.

**Risks.** None known — the removed surface was unreachable. Typecheck proves it.

**Acceptance.** `rg 'RetryOptions|fetchWithRetry' src` returns nothing; options
type no longer promises unsupported knobs; suite green.

---

## 4. Stop mutating persisted events in replay

**Problem.** `remapTerminalContent` (`src/replay.ts`) takes
`const meta = readMeta(event) ?? {}` and then assigns
`meta.terminal_output` / `meta.terminal_exit`. When the event already carries a
`_meta` object without those keys, `meta` aliases the persisted event's object,
so derived replay payloads are written into `session.events` and get persisted on
the next save — state grows and derived data becomes indistinguishable from
recorded data.

**Approach.** Shallow-clone before writing:
`const meta = { ...(readMeta(event) ?? {}) }`. The only writes are new keys with
freshly built objects, so a shallow copy is sufficient; existing nested values
are never modified. Then audit the rest of the module for the same class of bug:

- `prepareReplayEvents` only reads `rawOutput` and builds fresh objects — fine.
- `coalesceReplayEvents` / `enrichReplayEvent` return spreads — fine, but the
  test below pins it.

**Files.** `src/replay.ts`; regression test near the existing replay tests
(`src/transcript-dedupe.test.ts` or `src/agent.test.ts`, wherever
`prepareReplayEvents` fixtures live).

**Tests to add.**

- Input event with `_meta: { terminal_info: ... }` (no `terminal_output`) passes
  through `prepareReplayEvents`; `expect(input).toEqual(deepCopyOfInput)`
  afterwards, and the returned event carries `terminal_output` while the input
  still does not.
- Same assertion for `coalesceReplayEvents` input immutability while at it.

**Risks.** None; strictly narrower behavior.

**Acceptance.** Immutability test red on the old code, green after; suite green.

---

## 5. Consolidate lifecycle ownership; shrink the four large modules

**Problem.** `agent-turn.ts` (~680 lines), `agent-prompt.ts` (~540),
`tool-execution.ts` (~550), `chat-completions.ts` (~530). The review's point was
not length per se: cancellation, accounting, persistence and cleanup are
distributed, so every defect fix had to touch several of these files at once.
The target is explicit ownership boundaries, extracted in low-risk phases.

**Ownership map to reach (and then document in `src/agent.ts`):**

| Concern                            | Owner                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| Serialization / cancellation       | `withSessionOperation` queue + `AbortController` in core/prompt/session mixins |
| Turn accounting & reporting        | new `turn-reporting` unit                                                      |
| Persistence boundaries             | `save()` call sites remain with the turn/prompt owners                         |
| Terminal lifecycle & kill-on-abort | new bash-tool unit                                                             |
| Stream cleanup                     | `StreamThrottle.discard()` at the step owner                                   |
| Wire conversion                    | new pure converter module                                                      |

**Phase A — `turn-reporting` (lowest risk).** Move
`accumulateTurnUsage`, `recordCacheDiagnostic`, `mergeTurnStats`, `reportUsage`,
`emitTurnStats`, `logStepStats`, `logTurnStats`, `costCurrency` out of
`agent-turn.ts` into a new mixin (`src/turn-reporting.ts`) over `ZenAgentCore`,
or plain helpers taking `(active, cx)`. The exactly-once `finalize` guard stays
in `agent-turn.ts` — it is the turn owner's job — and only delegates emission.
Exit criterion: `agent-turn.ts` contains the loop, the finalize guard and the
hard-abort/tool-pairing logic, nothing about formatting or diagnostics.

**Phase B — `agent-commands`.** Move `handleSlashCommand` and the
`/prompt`, `/sandbox`, `/tools`, skill-invocation handlers from
`agent-prompt.ts` into `src/agent-commands.ts` (mixin or methods taking
`ActiveSession`). `agent-prompt.ts` keeps the prompt entry point, block-to-parts
preprocessing and the parse/dispatch gate (`isKnownSlashCommand` stays with it or
moves with the handlers — pick one home). The controller-ownership rule added in
`5670e76` (`return await this.handleSlashCommand(...)`) stays in
`agent-prompt.ts`, since that file owns the persistence boundary.

**Phase C — tool layer.** From `tool-execution.ts` extract:

- `src/tool-bash.ts`: terminal create/wait/output/kill/release, the abort
  listener, pre-execution abort check and result formatting for bash;
- `src/tool-read-media.ts`: the read_media handler;
- a shared `emitFailedToolResult(...)` helper for the five duplicated
  emit-pairs (tools-disabled, malformed arguments, unknown tool, invalid
  command, cancelled-before-run). Centralizing these is the concrete fix for
  "distributed lifecycle": every refusal path gets one shape and one log line.

  `tool-execution.ts` keeps the context types and the dispatch front door
  (`executeLlmToolCall`). Its `isRecord` moves next to the dispatch or to a tiny
  shared guard, resolving the duplicate in `chat-completions.ts` in passing.

**Phase D — `chat-completions` conversion split.** Move the pure layer —
`toolFromSchema`, `toPiTools`, `toPiUserContent`, `reasoningReplayField`,
`storedReasoningSignature`, `toPiContext`, `toLlmUsage`, `mapFinishReason`,
`patchPayload` — into `src/chat-completions-convert.ts`, exported for direct
unit tests. What remains in `chat-completions.ts`: model construction, timeout,
rate-limit wait, the streaming loop and result assembly.

**Execution notes.**

- Order A → D by increasing blast radius; each phase is its own commit and must
  keep every existing test passing unchanged — the tests are the behavior
  contract. Add tests only for newly pure units (converters, the failed-result
  helper).
- No mixin-signature churn in `src/agent.ts` beyond composing the new mixins.
- Soft check after each phase: `wc -l src/agent-*.ts src/tool-*.ts
src/chat-completions*.ts`; the goal is reviewable units, not a hard cap.
- Rollback is `git revert` of the single phase commit.

**Risks.** Mixin method visibility (`protected` across files) can force widening
to `protected` on the new mixins or switching those helpers to free functions
taking `ActiveSession`; prefer free functions where they need no agent state.
Watch for accidental import cycles (turn-reporting ↔ agent-turn) — keep
reporting dependent on the turn, never the reverse.

**Acceptance.** Each phase lands separately, suite green; afterwards the
ownership table above is written into the `src/agent.ts` header comment so the
next defect fix has one place to look.

---

## Suggested sequencing

1. Concern 4 (one-line fix + immutability test) — quick win, independent.
2. Concern 3 (delete dead retry surface) — independent, shrinks API.
3. Concern 1 (storage element validation) — self-contained in storage + tests.
4. Concern 5 phases A–D — one commit each, after the above settle.

Any concern may be dropped independently; none depends on another except the
soft guidance that 5 benefits from landing last on a stable base.
