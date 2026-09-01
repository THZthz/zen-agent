# Zen Agent — ACP v1 Coding Agent Specification

## 1. Overview

Zen Agent is a TypeScript coding agent speaking **Agent Client Protocol (ACP) v1** over **stdio**, launched by Zed (or any ACP client) on WSL2/Linux. It exposes exactly one tool — an unrestricted **`bash`** — executed through the client's ACP terminal, with approval policy **never**.

LLM providers are **user-defined** — there are no built-ins. Every provider is declared in `ZEN_AGENT_PROVIDERS` / `ZEN_AGENT_PROVIDERS_FILE` with an endpoint, API key env, and either an explicit `models` list or `fetchModels: true` to auto-discover models from `GET {baseUrl}/models`. Providers are chosen **per session**; all of them run through pi-ai's `createProvider`/`Models` registry.

## 2. Protocol Surface

### 2.1 Transport

- stdio only; one JSON-RPC message per line of UTF-8 JSON.
- stdout carries only valid ACP messages; logs go to stderr.

### 2.2 Agent-Implemented Methods

| Method                      | Description                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `initialize`                | Negotiate protocol version and capabilities.                                                     |
| `authenticate`              | No-op; returns `{}`.                                                                             |
| `session/new`               | Create a persistent session (row in the SQLite database) and freeze the environment message.     |
| `session/load`              | Load a stored session and replay its events.                                                     |
| `session/list`              | List stored sessions (by `cwd`, or the global index).                                            |
| `session/resume`            | Load without replaying history.                                                                  |
| `session/delete`            | Delete a stored session.                                                                         |
| `session/close`             | Hard-abort active work and drop the session from memory.                                         |
| `session/set_config_option` | Change `provider` / `model` / `thinking_effort` (locked after the first user message, see §3.1). |
| `session/prompt`            | Run a full agent turn.                                                                           |
| `session/cancel`            | Request a graceful stop of the active turn (§4.1).                                               |

`initialize` response: `protocolVersion: 1`; `agentCapabilities.loadSession: true` with `sessionCapabilities: { list, delete, resume, close }`; `agentCapabilities.promptCapabilities: { image: true, audio: true }` (Zed gates paste / drag & drop / @-mention of images on this; embeddedContext stays off); `agentInfo` (name `zen-agent`, title `Zen Agent`, version `0.1.0`); `authMethods: []`.

### 2.3 Client-Implemented Methods Used

| Method                       | When                                                                       |
| ---------------------------- | -------------------------------------------------------------------------- |
| `session/update`             | Text/thought chunks, tool calls, usage updates, available commands.        |
| `terminal/*`                 | `create`, `wait_for_exit`, `output`, `release`, `kill` for bash execution. |
| `session/request_permission` | Never (approval policy `never`).                                           |

## 3. Sessions & Storage

Everything persists in one SQLite database (built-in `node:sqlite`, WAL mode + busy timeout for concurrent agent processes). The file defaults to `zen-agent.db` next to the package and is overridden by `ZEN_AGENT_DB_FILE` (relative paths resolve against the agent process cwd):

```
sessions        one row per session; scalar columns (session_id, cwd,
                created_at, updated_at, title) + JSON columns (config, usage,
                events, llm_messages, turn_stats, cache_diagnostics)
llm_log         LLM request/response transcript, one JSON entry per row
runtime_log     per-process diagnostic log, grouped by startup_key
terminal_calls  one row per bash tool call (id, session_id, created_at,
                command, output)
```

Each bash tool call also writes `/tmp/zen-agent/<id>.sh` (input script) and `/tmp/zen-agent/<id>.log` (full `script -q -e` output), named by a short `msg_<base62>` id that matches the `terminal_calls` row id; the database row is the durable copy, the files stay around for direct inspection.

A session row holds: `sessionId`, `cwd`, `createdAt`/`updatedAt`, `title`, `events` (ACP `session/update` payloads for replay), `llmMessages` (full conversation), `config` (`provider`, `model`, `thinkingEffort`, `systemPrompt`, `sandbox`, `roBindPaths`, `toolsEnabled`), `usage` and `turnStats` (cumulative + per-turn statistics). The `sessions` table itself is the session index (id -> cwd, titles, `updatedAt`); no separate index exists. Deleted sessions remove their row plus their `llm_log` and `terminal_calls` entries.

Provider model catalogs are cached globally under `$XDG_DATA_HOME/zen-agent/models/<providerId>.pi.json` (pi `ModelsStore`) plus `<providerId>.catalog.json` (Zen metadata: modalities, tool support) so offline starts restore the last-known catalog.

- **`session/new`** validates an absolute `cwd`, creates the session and appends a frozen environment message (working directory, session time, git state) as a `user` message named `Environment` — byte-stable so provider prefix caches keep hitting. With `/tools off` the environment message is omitted (chat-only session). Returns `{ sessionId, configOptions }`. `mcpServers`/`additionalDirectories` are accepted and ignored.
- **`session/load`** replays persisted events through `prepareReplayEvents` (see §5.2) and returns the current `configOptions`; **`session/resume`** loads without replay.
- On load/resume, a fresh environment _continuation_ message is appended at the **end** of the conversation (the cached prefix is untouched); for `/tools off` sessions nothing is appended or backfilled, and any environment message left in the history by an older build is stripped (chat-only invariant).
- Sessions created before the `provider` field existed default to `ZEN_AGENT_DEFAULT_PROVIDER` (or the first configured provider).

### 3.1 Config Options

| Option            | Values                                                                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `provider`        | any configured provider id (no built-ins) — switching resets `model` to the provider default                                     |
| `model`           | declared models, or the live catalog when `fetchModels: true` (any slug via `set_config_option`)                                 |
| `thinking_effort` | full ladder (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`); `off` omits the field so the provider picks its default |

The provider selector is built from the registry; new sessions default to `ZEN_AGENT_DEFAULT_PROVIDER` (or the first configured provider).

`provider`, `model` and `thinking_effort` are **locked once the conversation contains a user message** (environment messages don't count): `session/set_config_option` then rejects the change with an error. The `/tools on|off` toggle and the `/prompt <new-prompt>` setter are locked the same way (their state is part of the cache prefix); `/tools`, `/tools status` and `/prompt` printing stay available.

## 4. Agent Turn Lifecycle (`session/prompt`)

1. Look up the session; convert `ContentBlock[]` via `promptBlocksToPromptContent`: text and resource links become text parts (as before), `image`/`audio` blocks become media parts (`ZEN_AGENT_MAX_MEDIA_BYTES`, default 10 MB decoded; oversize -> placeholder note). Transcript events keep the original blocks (Zed renders them); stored user messages keep plain-string content for pure-text prompts (cache-compatible) or part arrays otherwise. Media the active model cannot consume (per `getModelModalities`) degrades to placeholder text.
2. Slash commands are intercepted first (see §8). Otherwise the user message — named after `git config user.name`, fallback `User` — is appended to history.
3. Loop (max `ZEN_AGENT_MAX_TURN_STEPS`, default 25):
   a. Call `runLlmStep(provider, ...)` with the system prompt, full history, and the session's tool list: `bash` always, plus `read_media` when the model accepts image/audio input (stable per session - the list is part of the cached prefix). With `config.toolsEnabled` off (`/tools off`), the list is empty, the default system prompt drops the whole `<toolbox>` section — the bash-tool guidance (`SYSTEM_PROMPT_NO_TOOLS`) — and any tool call the model still emits is refused with a failed result.
   b. Stream text deltas as `agent_message_chunk` and reasoning deltas as `agent_thought_chunk` (batched through `StreamThrottle`).
   c. For each tool call: emit `tool_call` (pending) → `tool_call_update` (in_progress) → execute via terminal (§5) → `tool_call_update` (completed/failed with terminal content and `rawOutput`); append assistant + tool messages to history.
   d. No tool calls → finish with the mapped stop reason (`length` → `max_tokens`, `content-filter` → `refusal`, `error` → throw, else `end_turn`).
4. After each step with usage: accumulate stats, emit `usage_update`, log step stats (§7).

### 4.1 Cancellation

`session/cancel` (Zed sends it for Stop **and** force-send; the protocol carries no reason field) triggers a **graceful** stop: the in-flight LLM step or bash command runs to completion, then the turn returns `stopReason: "cancelled"` at the next boundary. Tool calls proposed by a cancelled step are discarded; completed tool results are persisted so the follow-up has context. Zed awaits the prompt response before delivering the next message, which is what makes force-send "insert after the current unit of work".

Hard abort (terminal killed, stream cut) happens only on `session/close`, `session/delete`, or after `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` (0 = wait forever).

## 5. `bash` Tool

### 5.1 Tool Definition

```json
{
  "type": "function",
  "function": {
    "name": "bash",
    "description": "Execute a bash command in current OS. The command is completely unrestricted. Your command will be wrapped inside `script -q -e -c \"bash <script file containing your command>\" \"<log path>\"`. If output is large, this tool will tell you to check the log file instead of showing all.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": { "type": "string", "description": "The bash command to execute." }
      },
      "required": ["command"],
      "additionalProperties": false
    }
  }
}
```

Byte-identical across providers.

### 5.2 Execution

- Requires client `terminal: true`. Runs `/bin/bash -lc <command>` in the session `cwd` via `terminal/create`, waits with `terminal/wait_for_exit`, fetches output with `terminal/output`, releases with `terminal/release`; on hard abort `terminal/kill` + `terminal/release`. No local `child_process` execution.
- The command script and full output are kept as `/tmp/zen-agent/<id>.sh` / `<id>.log` (output via `script -q -e` to preserve TTY behavior) and stored in the `terminal_calls` table. Only the last `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` (default 50 000) bytes go back to the model, with a pointer to the log file and the database row.
- Tool cards show `$ <command>` and append `⏱ <duration>` to the output.
- **Replay across restarts**: each `tool_call` carries `_meta.terminal_info = { terminal_id: "zen-<callId>", cwd }` (display-only terminal) and the final update streams `_meta.terminal_output = { terminal_id, data }` + `_meta.terminal_exit = { terminal_id, exit_code, signal }`. Zed re-creates the display terminal during `session/load`, so bash cards replay with collapsible terminal output. `prepareReplayEvents` rewrites stale real terminal ids and synthesizes the metadata for legacy sessions.

### 5.3 Sandboxing

- `bin/zen-agent-bwrap.sh` wraps the agent process: `--bind / / --ro-bind /mnt /mnt --dev /dev --bind /dev/pts /dev/pts --tmpfs /dev/shm` (`/mnt` = Windows drives, read-only; fresh devtmpfs; host PTYs). Runs in a new user+mount namespace.
- The bash tool runs on the host, so it needs its own sandbox: with `ZEN_AGENT_SANDBOX=1` (env policy) or `config.sandbox` (`/sandbox on`), each bash call is wrapped in its own `bwrap` with the same `/mnt` policy (`bashSandboxPrefix` in `src/tools/execution.ts`). The env policy always wins: `/sandbox off` is refused while `ZEN_AGENT_SANDBOX=1`.
- Inside the bash sandbox, `rm`/`grep`/`find` are shadowed (read-only mount) by `bin/zen-agent-sandbox-block.sh`, which refuses to run and suggests `trash`/`rg`/`fdfind`; the host is unaffected. `ZEN_AGENT_SANDBOX_CMD` overrides the whole bwrap command; `ZEN_AGENT_SANDBOX_BLOCK_SHIM` overrides the shim path.
- Per-session extra read-only mounts: `config.roBindPaths` (set with `/robind <path>[,<path>…]`, emptied with `/robind clear`, status with no argument) adds `--ro-bind <path> <path>` for each entry to the bash sandbox; an empty list means no extra binds.

### 5.2 `read_media` Tool (`src/tools/media.ts`)

Offered only on sessions whose model accepts image/audio input (OpenRouter `architecture.input_modalities`; unknown catalog entries count as text-only). The model calls it with a file path; the agent resolves it against the session cwd, maps extensions to MIME types (png/jpeg/webp/gif/bmp images, wav/mp3 audio), enforces the size limit, then:

- returns a short metadata line as the normal tool result (keeps assistant tool_calls paired - DeepSeek 400s on unpaired calls), and
- injects the base64 payload as parts of a synthetic **user** message (the OpenAI-compatible tool role only accepts string content). For cache safety the synthetic message is inserted **before** the assistant tool-call message (the tool result still pairs with it): GLM/Z.AI's context cache drops to a 0% hit rate when a request ends with an image/audio-bearing user message (verified against a live `z-ai/glm-5.3-flash` session), while the identical prefix ending in a tool result keeps hitting ~99%. The request after a `read_media` step therefore ends with the tool result, not the media message.

Failures (missing file, unsupported extension, modality not accepted by the model) produce a failed tool result without injection. The system prompt is not modified for media; the model learns about `read_media` from the tool schema alone.

## 6. LLM Providers

Every provider runs through pi-ai's `createProvider`/`Models` collection and the shared OpenAI-compatible chat-completions adapter. There are **no built-in providers** — users define every provider via `ZEN_AGENT_PROVIDERS` (inline JSON) or `ZEN_AGENT_PROVIDERS_FILE` (JSON file).

### 6.1 Provider registry (`src/providers/registry.ts`)

A `ProviderDefinition` is pure data: `id`, `name`, `label`, `baseUrl`, `apiKeyEnv`, `defaultModel`, `currency`, `discovery.enabled` (from `fetchModels`), `thinkingMode` (`openai`/`deepseek`, see §6.4), `staticModels` (declared models with `contextLength`/`cost`/`modalities`/`thinkingEfforts`), `pricing.fallback`, optional `balance`, pi `compat`, and generic `extraBody`/`extraHeaders`/`sendSessionId`. The registry rebuilds automatically when provider-relevant env changes.

Example (`ZEN_AGENT_PROVIDERS`):

```json
[
  {
    "id": "deepseek",
    "name": "DeepSeek",
    "baseUrl": "https://api.deepseek.com",
    "apiKeyEnv": "DEEPSEEK_API_KEY",
    "defaultModel": "deepseek-v4-flash",
    "models": [
      { "id": "deepseek-v4-flash", "contextLength": 1000000 },
      { "id": "deepseek-v4-pro", "contextLength": 1000000 }
    ]
  },
  {
    "id": "groq",
    "name": "Groq",
    "baseUrl": "https://api.groq.com/openai/v1",
    "apiKeyEnv": "GROQ_API_KEY",
    "defaultModel": "llama-3.3-70b-versatile",
    "fetchModels": true
  }
]
```

- `models` declares what the provider offers; each entry may carry `name`, `description`, `contextLength`, `cost` (`{inputPerM, outputPerM}` per 1M tokens in the provider's currency), `modalities` (`["image"]` / `["audio"]`; `text` implicit) and `thinkingEfforts` (see §6.4). This is the way to describe models the endpoint itself doesn't document.
- `fetchModels: true` auto-discovers models from `GET {baseUrl}/models`; declared `models` are still offered alongside the catalog. `defaultModel` is required in this mode.
- A provider with neither `models` nor `fetchModels: true` is a config error. `apiKeyEnv` is optional (keyless local endpoints). Duplicate ids are rejected with clear errors.

### 6.2 pi integration (`src/providers/pi.ts`)

Each definition becomes a pi provider via `createProvider({ id, name, baseUrl, auth: envApiKeyAuth(...), models: declaredModels, fetchModels, api: openAICompletionsApi() })` inside one `Models` collection. `fetchModels` runs the generic `/models` discovery and converts entries to pi `Model` objects carrying `cost`, `contextWindow` and `input` modalities (declared metadata wins over the catalog). pi's `ModelsStore` (a per-provider file under `$XDG_DATA_HOME/zen-agent/models/<providerId>.pi.json`) restores the catalog on offline starts; Zen.s `src/providers/catalog.ts` keeps a parallel `<providerId>.catalog.json` for modalities and tool support. Unknown slugs on discovery providers are synthesized from the catalog (or conservative defaults), so any slug stays usable.

### 6.3 Shared client (`src/chat-completions.ts`)

`runChatCompletions` takes the pi model (from the registry), the resolved API key, and Zen's session/step options, then consumes pi's `openai-completions` event stream. Pi owns endpoint construction, provider compatibility (auto-detected by base URL, overridden by the definition's `compat`), retry behavior, and SSE parsing; it emits live text/thinking deltas and normalized tool calls, finish reasons, and usage. Zen keeps its client-side rate limit, hard request timeout, healing and local usage/timing rollup.

### 6.4 Thinking effort

The session `thinking_effort` maps to the OpenAI `reasoning_effort` field. `thinkingMode` selects the wire format: `openai` (default) sends `reasoning_effort` and `off` omits the field; `deepseek` sends `thinking: {type: "enabled" | "disabled"}` so `off` actually turns thinking off (DeepSeek defaults it ON) while effort values pass through unchanged — DeepSeek's API auto-maps them to its own levels.
A model's declared `thinkingEfforts` (per-model, in declared order) restricts the selector and remaps unsupported values; without it the full ladder is accepted passthrough:

- value in the list → sent unchanged; `off` in the list omits the field (provider default applies).
- value not in the list → nearest declared value by ladder distance (ties resolve upward: `medium` between `low` and `high` → `high`, `xhigh` → `max`).
- `off` not in the list → mandatory reasoning: the LOWEST declared effort is sent (closest to disabled), e.g. z.ai GLM-5.3's `["low", "high", "max"]`.

### 6.5 Cost, context, modalities, balance

- **Pricing**: a model's declared `cost` wins; otherwise the discovered `/models` prices are used — including `pricing.input_cache_read` when the gateway breaks cache reads out — and models without prices fall back to the provider's `fallback` (USD 1/2 per 1M).
- **Context window**: a model's declared `contextLength` wins; otherwise `context_length` from `/models` (default 200K).
- **Modalities**: declared `modalities` are definitive; discovery providers additionally read the catalog entry's `architecture.input_modalities` (image/audio) — `null` while unknown so the session retries instead of caching a wrong text-only answer.
- **Balance**: providers may declare an optional balance endpoint + parser; without one, `fetchBalanceSnapshot` reports `isAvailable: false` and balance verification is skipped (best-effort data gathering never fails the turn).

## 7. Usage & Stats## 7. Usage & Stats

- Per-step and per-turn token/cost/timing stats accumulate into `session.usage` (cumulative) and `turnStats` (per turn), persisted in the session row.
- ACP `usage_update` after each LLM step: `used`/`size` (context window from the provider: DeepSeek env or OpenRouter model) and `cost: { amount, currency }` — `CNY` (DeepSeek) or `USD` (OpenRouter) — which Zed renders as the token-usage ring in the agent panel header.
- With `ZEN_AGENT_SHOW_STATS` (default on), a per-turn stats line is emitted as a display-only `agent_message_chunk` (never pushed to `llmMessages`, so it costs no context): `Turn 3 · 4 steps · think 3.2s · answer 8.5s · tools 14.2s` + `in 45.6K · out 3.4K · cache hit 87% · ¥0.043 (session ¥0.12)`.
- The experimental `session/prompt` `usage` field carries cumulative input/output/thought/cache tokens.
- After each stats line, `verifyTurnCost` fetches the provider's balance and logs the delta vs. the locally estimated cost to `log.jsonl` ("turn stats balance verify") — data gathering only, never blocks the turn.

## 8. Slash Commands & Skills

After `session/new`/`load`/`resume`, an `available_commands_update` notification advertises:

| Command        | Behavior                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | Set the session's entire system prompt (multi-line supported) or print the current one; returns `end_turn` without calling the model. The set form is locked after the first user message (the system prompt is part of the cache prefix); printing stays open.                                                                                                                                                     |
| `sandbox`      | Toggle `config.sandbox` (`on`/`off`/status), persisted in the session row; refused while `ZEN_AGENT_SANDBOX=1`.                                                                                                                                                                                                                                                                                                     |
| `robind`       | Replace `config.roBindPaths` with the comma-separated argument (`/robind <path>[,<path>…]`), clear it (`/robind clear`) or print status (no argument); persisted in the session row. Non-empty lists add `--ro-bind <path> <path>` per entry to the bash sandbox.                                                                                                                                                   |
| `tools`        | Toggle `config.toolsEnabled` (`on`/`off`/status), persisted in the session row; locked after the first user message (tool list + environment are part of the cache prefix), status stays open. Off makes the session chat-only: the environment snapshot is dropped from the history and nothing is injected on load/resume; on restores the snapshot. Off sends no tool schemas and refuses any emitted tool call. |
| `<skill-name>` | One per installed skill: reads `SKILL.md` from `<cwd>/.agents/skills/` or `~/.agents/skills/`, injects it (plus the user's argument) as a user message, and runs a normal turn. Always available, independent of `ZEN_AGENT_SHOW_SKILLS_CATALOG`.                                                                                                                                                                   |

Unknown commands reply `Unknown slash command` and return `end_turn`.

Skills: with `ZEN_AGENT_SHOW_SKILLS_CATALOG=1`, a compact catalog (name, description, load command) is frozen into the environment message at session creation. The model never loads skills on its own — invocation is by hand or slash command only.

## 9. Logs

- `llm_log` (db): every LLM request (system prompt, messages) and response (text, tool calls, finish reason, usage), one JSON entry per row.
- `runtime_log` (db): lifecycle events (session created/loaded, prompt received, terminal created/finished), `llm step stats`, `turn stats`, `turn stats balance verify`, skill invocations, graceful-cancel events — grouped per agent process by `startup_key` ("YYYY-MM-DD-HH-mm-ss_<uuid>").

## 10. Project Layout

```
zen-agent/
  bin/
    zen-agent-bwrap.sh         bwrap wrapper for the agent process
    zen-agent-sandbox-block.sh shim shadowing rm/grep/find in the bash sandbox
  src/
    index.ts                   stdio entry point (ACP wiring)
    agent/                     ACP orchestration: handlers, sessions, turns, commands, stats
      index.ts                 ZenAgent class (ACP handlers + mixin composition)
      core.ts                  shared state and plumbing
      config.ts                session config options + locking
      session.ts               session lifecycle mixin (new/load/resume/close/delete)
      turn.ts                  LLM turn loop + stream cleanup
      commands.ts              slash commands
      prompt.ts                prompt/slash-command entry point
      reporting.ts             turn accounting and reporting
      stats.ts                 stats formatting
      stream-throttle.ts       LLM delta batching for agent_message_chunk
      prompt-content.ts        ACP ContentBlock[] -> user-message parts (text + media)
      tests/                   lifecycle, concurrency, shutdown, prompt tests
    providers/                 provider facade, registry, LLM client, wire conversion
      index.ts                 per-session provider facade (step, pricing, balance)
      registry.ts              user-defined provider definitions + parsing
      pi.ts                    pi createProvider/Models collection + discovery
      catalog.ts               generic /models discovery + Zen metadata cache
      balances.ts              optional balance fetch (generic)
      llm-client.ts            shared LlmUsage/LlmStep types + cost + bash schema
      llm-errors.ts            LLM API failure classification
      chat-completions.ts      pi-ai OpenAI-completions adapter (stream loop)
      convert.ts               OpenAI wire conversion (pure layer)
      heal.ts                  message history healing before API calls
      rate-limit.ts            client-side chat request spacing
      cache-diagnostics.ts     per-turn cache hit/miss diagnostics
      tests/                   facade, registry, client, wire tests
    tools/                     bash/read_media tools, media handling, sandboxing
      bash.ts                  bash tool via client terminals
      execution.ts             tool execution + sandbox policy
      read-media.ts            read_media tool
      media.ts                 read_media path resolution/validation
      media-limit.ts           shared ZEN_AGENT_MAX_MEDIA_BYTES limit
      sandbox.ts               bwrap policy + rm/grep/find shim shadowing
      tests/                   execution, sandbox, media flow tests
    session/                   persistence, replay, skills, system prompt
      storage.ts               SQLite-backed session persistence (db.ts)
      db.ts                    the single SQLite database: path, schema, inserts
      data-dir.ts              user data dir + crash-safe file writes (catalog cache)
      replay.ts                session/load event replay + terminal metadata synthesis
      skills.ts                Agent Skills discovery and invocation
      system-prompt.ts         system prompt, environment message, user naming
      tests/                   storage, skills, system prompt tests
    util/                      shared helpers
      env.ts                   environment variable parsing
      logger.ts                log entry builder
      is-record.ts             record type guard
    test-server.ts             local HTTP server for provider tests
    test-setup.ts              vitest setup
  dist/                        build output
```

## 11. Dependencies

- `@agentclientprotocol/sdk` — ACP v1 JSON-RPC/stdio plumbing and types.
- `@earendil-works/pi-ai` — the LLM provider layer: `createProvider`/`Models` for provider registration, `/models` discovery and offline catalog restore, and the OpenAI-compatible chat-completions stream (compat auto-detection, retries, thinking formats, usage normalization).
- `sonyflake` — message/session id generation.
- dev: `typescript`, `tsx`, `vitest`, `@types/node`.

## 12. Testing

`npm test` (vitest):

- `provider.test.ts` / `provider-pi.test.ts` / `provider-catalog.test.ts` — facade dispatch, user-provider parsing (declared models, `fetchModels` validation), pi collection + model options, generic `/models` discovery + parse + persistence, SSE streaming through a user-defined provider against a local HTTP server.
- `agent.test.ts` / `agent.graceful.test.ts` — session lifecycle, config options + locking, graceful cancel, stats lines (with `runLlmStep` mocked).
- `skills.test.ts` / `skills-slash.test.ts` / `sandbox.test.ts` / `tools.test.ts` / `system-prompt.test.ts` / `tool-execution.test.ts` — skills, sandbox toggling, tools toggling + refusal, environment messages, terminal artifacts.
- `db.test.ts` / `storage-validate.test.ts` / `storage-delete.test.ts` — database path resolution + schema, session row CRUD/list/delete cascades, raw-row normalization backfill, corrupt-column errors, llm/runtime/terminal record inserts.
- `media.test.ts` / `prompt-content.test.ts` / `user-parts.test.ts` / `media-flow.test.ts` - media path resolution, prompt-block intake, OpenAI wire mapping (image_url data URIs, input_audio), and the end-to-end read_media turn flow (provider mocked).

## 13. Decisions

1. **Providers**: no built-ins — users define every provider in `ZEN_AGENT_PROVIDERS` with declared `models` and/or `fetchModels: true` auto-discovery, all through pi-ai's `createProvider`/`Models`. The wire/stream layer stays Zen-owned (`chat-completions.ts`) so healing, rate limiting, timeouts and usage rollup stay under test. Model metadata (context, cost, modalities) is declared by the user instead of being guessed from catalogs.
2. **ACP SDK**: official `@agentclientprotocol/sdk`.
3. **Sessions**: one shared SQLite database (`ZEN_AGENT_DB_FILE`, default next to the package) with load/list/resume/delete/close; terminal input/output also recorded there plus mirrored to `/tmp/zen-agent/`.
4. **MCP servers**: ignored; only the `bash` tool is exposed.
5. **Bash execution**: always through the client's ACP terminal; no local subprocess.
6. **Cancellation**: graceful only (the current unit of work completes); hard abort on close/delete or the timeout escape hatch.
