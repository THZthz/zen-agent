# Zen Agent — ACP v1 Coding Agent Specification

## 1. Overview

Zen Agent is a TypeScript coding agent speaking **Agent Client Protocol (ACP) v1** over **stdio**, launched by Zed (or any ACP client) on WSL2/Linux. It exposes exactly one tool — an unrestricted **`bash`** — executed through the client's ACP terminal, with approval policy **never**.

LLM providers are **pluggable**: DeepSeek (default) and OpenRouter are built in, and any OpenAI-compatible endpoint can be added with just a base URL + API key via `ZEN_AGENT_PROVIDERS` / `ZEN_AGENT_PROVIDERS_FILE` — models are auto-discovered from `GET {baseUrl}/models`. Providers are chosen **per session**; all of them run through pi-ai's `createProvider`/`Models` registry.

## 2. Protocol Surface

### 2.1 Transport

- stdio only; one JSON-RPC message per line of UTF-8 JSON.
- stdout carries only valid ACP messages; logs go to stderr.

### 2.2 Agent-Implemented Methods

| Method                      | Description                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `initialize`                | Negotiate protocol version and capabilities.                                                     |
| `authenticate`              | No-op; returns `{}`.                                                                             |
| `session/new`               | Create a persistent session under `<cwd>/.sessions/` and freeze the environment message.         |
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

Layout under the project's `.sessions/` directory:

```
.sessions/
  <sessionId>/state.json                      session state (below)
  <sessionId>/llm.jsonl                       LLM request/response transcript
  <sessionId>/terminals/input-<ts>-<callId>.sh    bash script per tool call
  <sessionId>/terminals/output-<ts>-<callId>.log  full `script -q -e` output
  client/<startupTs>_<uuid>/log.jsonl         per-process runtime diagnostic log
```

Provider model catalogs are cached globally under `$XDG_DATA_HOME/zen-agent/models/<providerId>.pi.json` (pi `ModelsStore`) plus `<providerId>.catalog.json` (Zen metadata: modalities, tool support, reasoning allowlist) so offline starts restore the last-known catalog.

`state.json` holds: `sessionId`, `cwd`, `createdAt`/`updatedAt`, `title`, `events` (ACP `session/update` payloads for replay), `llmMessages` (full conversation), `config` (`provider`, `model`, `thinkingEffort`, `systemPrompt`, `sandbox`, `toolsEnabled`), `usage` and `turnStats` (cumulative + per-turn statistics). A global index at `$XDG_DATA_HOME/zen-agent/index.json` maps session ids to their `cwd`.

- **`session/new`** validates an absolute `cwd`, creates the session and appends a frozen environment message (working directory, session time, git state) as a `user` message named `Environment` — byte-stable so provider prefix caches keep hitting. With `/tools off` the environment message is omitted (chat-only session). Returns `{ sessionId, configOptions }`. `mcpServers`/`additionalDirectories` are accepted and ignored.
- **`session/load`** replays persisted events through `prepareReplayEvents` (see §5.2) and returns the current `configOptions`; **`session/resume`** loads without replay.
- On load/resume, a fresh environment _continuation_ message is appended at the **end** of the conversation (the cached prefix is untouched); for `/tools off` sessions nothing is appended or backfilled, and any environment message left in the history by an older build is stripped (chat-only invariant).
- Sessions created before the `provider` field existed default to `deepseek`.

### 3.1 Config Options

| Option            | Values                                                                                                                                             |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `provider`        | any registered provider id (built-ins `deepseek`, `openrouter` + `ZEN_AGENT_PROVIDERS` entries) — switching resets `model` to the provider default |
| `model`           | DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro` · OpenRouter/discovery providers: live catalog (any slug via `set_config_option`)                 |
| `thinking_effort` | DeepSeek: `off`, `low`, `high`, `max` · allowlist providers: `off` plus the model's `supported_efforts`, sorted ascending · generic: full ladder   |

The provider selector is built from the registry; new sessions default to `ZEN_AGENT_DEFAULT_PROVIDER` (default `deepseek`).

`provider`, `model` and `thinking_effort` are **locked once the conversation contains a user message** (environment messages don't count): `session/set_config_option` then rejects the change with an error. The `/tools on|off` toggle and the `/prompt <new-prompt>` setter are locked the same way (their state is part of the cache prefix); `/tools`, `/tools status` and `/prompt` printing stay available.

## 4. Agent Turn Lifecycle (`session/prompt`)

1. Look up the session; convert `ContentBlock[]` via `promptBlocksToPromptContent`: text and resource links become text parts (as before), `image`/`audio` blocks become media parts (`ZEN_AGENT_MAX_MEDIA_BYTES`, default 10 MB decoded; oversize -> placeholder note). Transcript events keep the original blocks (Zed renders them); stored user messages keep plain-string content for pure-text prompts (cache-compatible) or part arrays otherwise. Media the active model cannot consume (per `getModelModalities`) degrades to placeholder text.
2. Slash commands are intercepted first (see §8). Otherwise the user message — named after `git config user.name`, fallback `User` — is appended to history.
3. Loop (max `ZEN_AGENT_MAX_TURN_STEPS`, default 25):
   a. Call `runLlmStep(provider, ...)` with the system prompt, full history, and the session's tool list: `bash` always, plus `read_media` when the model accepts image/audio input (stable per session - the list is part of the cached prefix). With `config.toolsEnabled` off (`/tools off`), the list is empty, the default system prompt drops every tool reference — the bash paragraph and the "use shell tools" file-modification guidance (`SYSTEM_PROMPT_NO_TOOLS`) — and any tool call the model still emits is refused with a failed result.
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
- The command script and full output are saved under `terminals/` (output via `script -q -e` to preserve TTY behavior). Only the last `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` (default 50 000) bytes go back to the model, with a pointer to the log file.
- Tool cards show `$ <command>` and append `⏱ <duration>` to the output.
- **Replay across restarts**: each `tool_call` carries `_meta.terminal_info = { terminal_id: "zen-<callId>", cwd }` (display-only terminal) and the final update streams `_meta.terminal_output = { terminal_id, data }` + `_meta.terminal_exit = { terminal_id, exit_code, signal }`. Zed re-creates the display terminal during `session/load`, so bash cards replay with collapsible terminal output. `prepareReplayEvents` rewrites stale real terminal ids and synthesizes the metadata for legacy sessions.

### 5.3 Sandboxing

- `bin/zen-agent-bwrap.sh` wraps the agent process: `--bind / / --ro-bind /mnt /mnt --dev /dev --bind /dev/pts /dev/pts --tmpfs /dev/shm` (`/mnt` = Windows drives, read-only; fresh devtmpfs; host PTYs). Runs in a new user+mount namespace.
- The bash tool runs on the host, so it needs its own sandbox: with `ZEN_AGENT_SANDBOX=1` (env policy) or `config.sandbox` (`/sandbox on`), each bash call is wrapped in its own `bwrap` with the same `/mnt` policy (`bashSandboxPrefix` in `tool-execution.ts`). The env policy always wins: `/sandbox off` is refused while `ZEN_AGENT_SANDBOX=1`.
- Inside the bash sandbox, `rm`/`grep`/`find` are shadowed (read-only mount) by `bin/zen-agent-sandbox-block.sh`, which refuses to run and suggests `trash`/`rg`/`fdfind`; the host is unaffected. `ZEN_AGENT_SANDBOX_CMD` overrides the whole bwrap command; `ZEN_AGENT_SANDBOX_BLOCK_SHIM` overrides the shim path.

### 5.2 `read_media` Tool (`src/media.ts`)

Offered only on sessions whose model accepts image/audio input (OpenRouter `architecture.input_modalities`; unknown catalog entries count as text-only). The model calls it with a file path; the agent resolves it against the session cwd, maps extensions to MIME types (png/jpeg/webp/gif/bmp images, wav/mp3 audio), enforces the size limit, then:

- returns a short metadata line as the normal tool result (keeps assistant tool_calls paired - DeepSeek 400s on unpaired calls), and
- injects the base64 payload as parts of a synthetic **user** message (the OpenAI-compatible tool role only accepts string content). For cache safety the synthetic message is inserted **before** the assistant tool-call message (the tool result still pairs with it): GLM/Z.AI's context cache drops to a 0% hit rate when a request ends with an image/audio-bearing user message (verified against a live `z-ai/glm-5.3-flash` session), while the identical prefix ending in a tool result keeps hitting ~99%. The request after a `read_media` step therefore ends with the tool result, not the media message.

Failures (missing file, unsupported extension, modality not accepted by the model) produce a failed tool result without injection. The system prompt is not modified for media; the model learns about `read_media` from the tool schema alone.

## 6. LLM Providers

Every provider runs through pi-ai's `createProvider`/`Models` collection and the shared OpenAI-compatible chat-completions adapter. DeepSeek and OpenRouter are built-in definitions; users add any OpenAI-compatible endpoint via `ZEN_AGENT_PROVIDERS` (inline JSON) or `ZEN_AGENT_PROVIDERS_FILE` (JSON file). When only an endpoint + API key are given (no `models` list), the model list is auto-discovered from `GET {baseUrl}/models`.

### 6.1 Provider registry (`src/provider-registry.ts`)

A `ProviderDefinition` is pure data: `id`, `name`, `label`, `baseUrl`, `apiKeyEnv`, `defaultModel`, `currency`, `discovery.enabled`, `staticModels`, `pinnedModelIds`, `pricing` (`catalog` / DeepSeek `table` / `fixed`), optional `balance` (path + JSON parser), `effort` (`static-map` / `allowlist` / `passthrough`), pi `compat`, `extraBody`/`extraHeaders`, and `sendSessionId`. The registry rebuilds automatically when any provider-relevant env var changes (same behavior the old OpenRouter cache had for base URL/key changes).

User providers (`ZEN_AGENT_PROVIDERS`):

```json
[
  {
    "id": "groq",
    "name": "Groq",
    "baseUrl": "https://api.groq.com/openai/v1",
    "apiKeyEnv": "GROQ_API_KEY",
    "defaultModel": "llama-3.3-70b-versatile",
    "currency": "USD"
  }
]
```

`models` (static list) is optional: omitted → auto-discovery. `apiKeyEnv` optional (keyless local endpoints). Duplicate ids and collisions with built-ins are rejected with clear errors.

### 6.2 pi integration (`src/provider-pi.ts`)

Each definition becomes a pi provider via `createProvider({ id, name, baseUrl, auth: envApiKeyAuth(...), models: staticModels, fetchModels, api: openAICompletionsApi() })` inside one `Models` collection. `fetchModels` runs the generic `/models` discovery and converts entries to pi `Model` objects carrying `cost`, `contextWindow`, `input` modalities, `compat` and a per-model `thinkingLevelMap` (see §6.4). pi's `ModelsStore` (a per-provider file under `$XDG_DATA_HOME/zen-agent/models/<providerId>.pi.json`) restores the catalog on offline starts; Zen's `provider-catalog.ts` keeps a parallel `<providerId>.catalog.json` for modalities (incl. audio), tool support and reasoning allowlists. Unknown slugs on discovery providers are synthesized from the catalog (or conservative defaults), so any slug stays usable.

### 6.3 Shared client (`src/chat-completions.ts`)

`runChatCompletions` takes the pi model (from the registry), the resolved API key, and Zen's session/step options, then consumes pi's `openai-completions` event stream. Pi owns endpoint construction, provider compatibility (auto-detected by base URL, overridden by the definition's `compat`), retry behavior, and SSE parsing; it emits live text/thinking deltas and normalized tool calls, finish reasons, and usage. Zen keeps its client-side rate limit, hard request timeout, session-specific routing fields, healing and local usage/timing rollup.

### 6.4 Thinking effort

Session `thinking_effort` values (`off`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`) map to the provider wire through the pi model's `thinkingLevelMap`:

- **DeepSeek** (`static-map`): `minimal`→`low`, `medium`→`high`, `xhigh`→`high`, `max`→`max`; `off` sends `thinking: {type:"disabled"}` (DeepSeek defaults thinking ON).
- **OpenRouter** (`allowlist`): per-model `reasoning.supported_efforts` from the catalog, remapped by ladder distance (ties break upward, so `medium` on `[max, high, low]` becomes `high`); `off` sends `none` when supported, else the lowest supported effort on mandatory-reasoning models, else omits the field. Unknown models accept every gateway value. The selector shows exactly `off` plus the model's supported tiers.
- **Generic** (`passthrough`): send the session value unchanged as `reasoning_effort` (OpenAI format).

### 6.5 Cost, context, modalities, balance

- **Pricing**: DeepSeek uses its static CNY rate table with Beijing peak/off-peak windows (peak 09:00-12:00 and 14:00-18:00; off-peak = half; `DEEPSEEK_PRICE_*` overrides still apply). Catalog providers use discovered prices (OpenRouter bills cache reads at the input rate); unknown entries fall back to the provider's `fallback` (USD 1/2 per 1M). Fixed providers use constants.
- **Context window**: static models use their definition value (DeepSeek `DEEPSEEK_CONTEXT_WINDOW`, default 1M); discovery models use `context_length` from `/models` (default 200K).
- **Modalities**: `getModelModalities` reads the catalog entry's `architecture.input_modalities` (image/audio) for discovery providers — `null` while unknown so the session retries instead of caching a wrong text-only answer — and is definitively text-only for static providers and static fallback models.
- **Balance**: definitions declare an optional endpoint + parser (`/user/balance` CNY for DeepSeek, `/auth/key` USD for OpenRouter); providers without one report `isAvailable: false`. Balance verification after each turn (`verifyTurnCost`) stays best-effort.

### 6.6 OpenRouter extras

- `OPENROUTER_PROVIDER_SORT` (default `price`; empty disables) sends the `provider: { sort }` routing block.
- `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` send `HTTP-Referer` / `X-Title` headers.
- The zen-agent session id is sent as `session_id` (top-level body field) so Z.AI pins the upstream context cache to the conversation.

## 7. Usage & Stats

- Per-step and per-turn token/cost/timing stats accumulate into `session.usage` (cumulative) and `turnStats` (per turn), persisted in `state.json`.
- ACP `usage_update` after each LLM step: `used`/`size` (context window from the provider: DeepSeek env or OpenRouter model) and `cost: { amount, currency }` — `CNY` (DeepSeek) or `USD` (OpenRouter) — which Zed renders as the token-usage ring in the agent panel header.
- With `ZEN_AGENT_SHOW_STATS` (default on), a per-turn stats line is emitted as a display-only `agent_message_chunk` (never pushed to `llmMessages`, so it costs no context): `Turn 3 · 4 steps · think 3.2s · answer 8.5s · tools 14.2s` + `in 45.6K · out 3.4K · cache hit 87% · ¥0.043 (session ¥0.12)`.
- The experimental `session/prompt` `usage` field carries cumulative input/output/thought/cache tokens.
- After each stats line, `verifyTurnCost` fetches the provider's balance and logs the delta vs. the locally estimated cost to `log.jsonl` ("turn stats balance verify") — data gathering only, never blocks the turn.

## 8. Slash Commands & Skills

After `session/new`/`load`/`resume`, an `available_commands_update` notification advertises:

| Command        | Behavior                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | Set the session's entire system prompt (multi-line supported) or print the current one; returns `end_turn` without calling the model. The set form is locked after the first user message (the system prompt is part of the cache prefix); printing stays open.                                                                                                                                                  |
| `sandbox`      | Toggle `config.sandbox` (`on`/`off`/status), persisted in `state.json`; refused while `ZEN_AGENT_SANDBOX=1`.                                                                                                                                                                                                                                                                                                     |
| `tools`        | Toggle `config.toolsEnabled` (`on`/`off`/status), persisted in `state.json`; locked after the first user message (tool list + environment are part of the cache prefix), status stays open. Off makes the session chat-only: the environment snapshot is dropped from the history and nothing is injected on load/resume; on restores the snapshot. Off sends no tool schemas and refuses any emitted tool call. |
| `<skill-name>` | One per installed skill: reads `SKILL.md` from `<cwd>/.agents/skills/` or `~/.agents/skills/`, injects it (plus the user's argument) as a user message, and runs a normal turn. Always available, independent of `ZEN_AGENT_SHOW_SKILLS_CATALOG`.                                                                                                                                                                |

Unknown commands reply `Unknown slash command` and return `end_turn`.

Skills: with `ZEN_AGENT_SHOW_SKILLS_CATALOG=1`, a compact catalog (name, description, load command) is frozen into the environment message at session creation. The model never loads skills on its own — invocation is by hand or slash command only.

## 9. Logs

- `llm.jsonl`: every LLM request (system prompt, messages) and response (text, tool calls, finish reason, usage).
- `client/<startupTs>_<uuid>/log.jsonl`: lifecycle events (session created/loaded, prompt received, terminal created/finished), `llm step stats`, `turn stats`, `turn stats balance verify`, skill invocations, graceful-cancel events.

## 10. Project Layout

```
zen-agent/
  bin/
    zen-agent-bwrap.sh         bwrap wrapper for the agent process
    zen-agent-sandbox-block.sh shim shadowing rm/grep/find in the bash sandbox
  src/
    index.ts           stdio entry point (ACP wiring)
    agent.ts           ACP handlers, session store, turn lifecycle, stats
    storage.ts         session persistence under <cwd>/.sessions/
    llm-client.ts      shared LlmUsage/LlmStep types + cost + bash schema
    chat-completions.ts pi-ai OpenAI-completions adapter (stream loop)
    provider-registry.ts provider definitions + built-ins + user providers
    provider-pi.ts     pi createProvider/Models collection + discovery
    provider-catalog.ts generic /models discovery + Zen metadata cache
    provider-balances.ts balance parsers (DeepSeek/OpenRouter)
    provider.ts        per-session provider facade (step, pricing, balance)
    prompt-content.ts  ACP ContentBlock[] -> user-message parts (text + media)
    media.ts           read_media path resolution/validation
    media-limit.ts     shared ZEN_AGENT_MAX_MEDIA_BYTES limit
    tool-execution.ts  bash tool via client terminals, sandboxing
    system-prompt.ts   system prompt, environment message, user naming
    skills.ts          Agent Skills discovery and invocation
    replay.ts          session/load event replay + terminal metadata synthesis
    stream-throttle.ts LLM delta batching for agent_message_chunk
    turn-stats.ts      stats formatting
    logger.ts          log.jsonl writer
  dist/                build output
```

## 11. Dependencies

- `@agentclientprotocol/sdk` — ACP v1 JSON-RPC/stdio plumbing and types.
- `@earendil-works/pi-ai` — the LLM provider layer: `createProvider`/`Models` for provider registration, `/models` discovery and offline catalog restore, and the OpenAI-compatible chat-completions stream (compat auto-detection, retries, thinking formats, usage normalization).
- `sonyflake` — message/session id generation.
- dev: `typescript`, `tsx`, `vitest`, `@types/node`.

## 12. Testing

`npm test` (vitest):

- `deepseek.test.ts` / `openrouter.test.ts` — provider SSE behavior through the registry facade against local HTTP servers: live reasoning streaming (timing-sensitive), streaming tool calls, wire format, usage parsing, effort allowlist mapping, routing/headers/session_id, balance endpoints.
- `provider-registry.test.ts` / `provider-pi.test.ts` / `provider-catalog.test.ts` — definitions + user-provider parsing, pi collection + effort maps + model options, generic `/models` discovery + parse + persistence.
- `agent.test.ts` / `agent.graceful.test.ts` — session lifecycle, config options + locking, graceful cancel, stats lines (with `runLlmStep` mocked).
- `skills.test.ts` / `skills-slash.test.ts` / `sandbox.test.ts` / `tools.test.ts` / `system-prompt.test.ts` / `tool-execution.test.ts` — skills, sandbox toggling, tools toggling + refusal, environment messages, terminal artifacts.
- `media.test.ts` / `prompt-content.test.ts` / `user-parts.test.ts` / `media-flow.test.ts` - media path resolution, prompt-block intake, OpenAI wire mapping (image_url data URIs, input_audio), and the end-to-end read_media turn flow (provider mocked).

## 13. Decisions

1. **Providers**: registry-driven (`provider-registry.ts`), DeepSeek default; any OpenAI-compatible endpoint is addable via `ZEN_AGENT_PROVIDERS` with auto-discovery through pi-ai's `createProvider`/`Models`. The wire/stream layer stays Zen-owned (`chat-completions.ts`) so healing, rate limiting, timeouts and usage rollup stay under test.
2. **ACP SDK**: official `@agentclientprotocol/sdk`.
3. **Sessions**: persisted under `<cwd>/.sessions/` with load/list/resume/delete/close.
4. **MCP servers**: ignored; only the `bash` tool is exposed.
5. **Bash execution**: always through the client's ACP terminal; no local subprocess.
6. **Cancellation**: graceful only (the current unit of work completes); hard abort on close/delete or the timeout escape hatch.
