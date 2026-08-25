# Zen Agent — ACP v1 Coding Agent Specification

## 1. Overview

Zen Agent is a TypeScript coding agent speaking **Agent Client Protocol (ACP) v1** over **stdio**, launched by Zed (or any ACP client) on WSL2/Linux. It exposes exactly one tool — an unrestricted **`bash`** — executed through the client's ACP terminal, with approval policy **never**.

Two OpenAI-compatible chat completions providers are supported, chosen **per session**: **DeepSeek** (default) and **OpenRouter**.

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
  client/models.openrouter.json               cached OpenRouter model catalog
```

`state.json` holds: `sessionId`, `cwd`, `createdAt`/`updatedAt`, `title`, `events` (ACP `session/update` payloads for replay), `llmMessages` (full conversation), `config` (`provider`, `model`, `thinkingEffort`, `systemPrompt`, `sandbox`), `usage` and `turnStats` (cumulative + per-turn statistics). A global index at `$XDG_DATA_HOME/zen-agent/index.json` maps session ids to their `cwd`.

- **`session/new`** validates an absolute `cwd`, creates the session and appends a frozen environment message (working directory, session time, git state) as a `user` message named `Environment` — byte-stable so provider prefix caches keep hitting. Returns `{ sessionId, configOptions }`. `mcpServers`/`additionalDirectories` are accepted and ignored.
- **`session/load`** replays persisted events through `prepareReplayEvents` (see §5.2) and returns the current `configOptions`; **`session/resume`** loads without replay.
- On load/resume, a fresh environment _continuation_ message is appended at the **end** of the conversation (the cached prefix is untouched).
- Sessions created before the `provider` field existed default to `deepseek`.

### 3.1 Config Options

| Option            | Values                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `provider`        | `deepseek`, `openrouter` — switching resets `model` to the provider default                                         |
| `model`           | DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro` · OpenRouter: `openrouter/free` (any slug via `set_config_option`) |
| `thinking_effort` | `off`, `high`, `max`                                                                                                |

`provider`, `model` and `thinking_effort` are **locked once the conversation contains a user message** (environment messages don't count): `session/set_config_option` then rejects the change with an error. Sessions default to `deepseek`.

## 4. Agent Turn Lifecycle (`session/prompt`)

1. Look up the session; convert `ContentBlock[]` via `promptBlocksToPromptContent`: text and resource links become text parts (as before), `image`/`audio` blocks become media parts (`ZEN_AGENT_MAX_MEDIA_BYTES`, default 10 MB decoded; oversize -> placeholder note). Transcript events keep the original blocks (Zed renders them); stored user messages keep plain-string content for pure-text prompts (cache-compatible) or part arrays otherwise. Media the active model cannot consume (per `getModelModalities`) degrades to placeholder text.
2. Slash commands are intercepted first (see §8). Otherwise the user message — named after `git config user.name`, fallback `User` — is appended to history.
3. Loop (max `ZEN_AGENT_MAX_TURN_STEPS`, default 25):
   a. Call `runLlmStep(provider, ...)` with the system prompt, full history, and the session's tool list: `bash` always, plus `read_media` when the model accepts image/audio input (stable per session - the list is part of the cached prefix).
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
- injects the base64 payload as parts of a synthetic **user** message right after the tool results (the OpenAI-compatible tool role only accepts string content).

Failures (missing file, unsupported extension, modality not accepted by the model) produce a failed tool result without injection. With `read_media` present, the system prompt gains a media-handling paragraph telling the model to perceive attachments natively and to load referenced files itself instead of asking the user.

## 6. LLM Providers

Both providers are OpenAI-compatible chat completions endpoints spoken to by one hand-rolled SSE client.

### 6.1 Shared Client (`src/llm-client.ts`)

`runChatCompletions` POSTs to `<baseUrl>/chat/completions` (retry on 429/5xx before the first byte, abort-aware) and parses the SSE stream directly. **The Vercel AI SDK was deliberately dropped**: `@ai-sdk/openai`'s `throwIfOpenAIStreamErrorBeforeOutput` reads ahead until the first output chunk — invisible to DeepSeek's `reasoning_content`-only reasoning phase — so the SDK buffered the whole thinking block, breaking live streaming. The raw parse also preserves provider-specific usage fields (cache tokens) that the SDK's zod schema strips.

Provider-specific knobs: reasoning delta fields, reasoning field in assistant history messages, extra body/headers, thinking-effort mapping, and a usage parser.

### 6.2 DeepSeek (`src/deepseek.ts`, default)

| Env                             | Default                    | Purpose                          |
| ------------------------------- | -------------------------- | -------------------------------- |
| `DEEPSEEK_API_KEY`              | —                          | API key (required).              |
| `DEEPSEEK_BASE_URL`             | `https://api.deepseek.com` | Base URL.                        |
| `DEEPSEEK_MODEL`                | `deepseek-v4-flash`        | Fallback model.                  |
| `DEEPSEEK_CONTEXT_WINDOW`       | `1_000_000`                | Context size for `usage_update`. |
| `DEEPSEEK_PRICE_*_CNY_PER_MTOK` | rate table                 | Per-rate price overrides.        |

- Reasoning streams as `delta.reasoning_content`; stored reasoning is echoed back as `reasoning_content` in assistant history (only alongside tool calls).
- Usage parsed from the raw chunk: `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens_details.reasoning_tokens`.
- `thinking_effort` maps directly to `reasoning_effort` (`off` omits the field).
- Cost: static CNY rate table with Beijing peak/off-peak windows (peak 09:00-12:00 and 14:00-18:00; off-peak = half), CNY per 1M tokens:

| Model               | Period   | Cache hit in | Cache miss in | Output |
| ------------------- | -------- | ------------ | ------------- | ------ |
| `deepseek-v4-flash` | Peak     | ¥0.10        | ¥3.00         | ¥9.00  |
| `deepseek-v4-flash` | Off-peak | ¥0.05        | ¥1.50         | ¥4.50  |
| `deepseek-v4-pro`   | Peak     | ¥0.30        | ¥9.00         | ¥27.00 |
| `deepseek-v4-pro`   | Off-peak | ¥0.15        | ¥4.50         | ¥13.50 |

- Balance verification: `GET /user/balance` (`fetchDeepSeekBalance`).

### 6.3 OpenRouter (`src/openrouter.ts`)

| Env                                           | Default                        | Purpose                             |
| --------------------------------------------- | ------------------------------ | ----------------------------------- |
| `OPENROUTER_API_KEY`                          | —                              | API key (required).                 |
| `OPENROUTER_BASE_URL`                         | `https://openrouter.ai/api/v1` | Base URL.                           |
| `OPENROUTER_MODEL`                            | `openrouter/free`              | Fallback model slug.                |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | —                              | `HTTP-Referer` / `X-Title` headers. |

- Reasoning streams as `delta.reasoning` (with `reasoning_content` accepted as passthrough for DeepSeek routes); stored reasoning is echoed back as `reasoning`.
- Sends `stream_options: { include_usage: true }` (OpenRouter omits usage otherwise). `parseOpenRouterUsage` reads generic `prompt_tokens`/`completion_tokens` plus optional passthrough cache (`prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens`) and reasoning fields.
- `reasoning_effort` uses the OpenAI vocabulary: `off` omits it, `high`/`max` → `high`.
- Cost/context: `GET /models` (fetched once per base URL+key, cached) provides USD pricing and `context_length`; static fallbacks cover `openrouter/free` ($0) and generic defaults for unknown slugs. OpenRouter bills cached reads at the regular input rate. Balance verification: `GET /auth/key` (remaining = limit − usage).
- **Model catalog**: the catalog is auto-fetched from `/models` (5s timeout) the first time an OpenRouter session's config options are requested, kept in memory per process, and persisted to `<cwd>/.sessions/client/models.openrouter.json` (versioned; best-effort). Offline starts load that file. The session `model` selector is built from the catalog — tool-capable models only (`supported_parameters` missing = assume yes), `openrouter/free` pinned first, then alphabetical — with the static list (`openrouter/free`) as final fallback. v2 cache files also carry `architecture.input_modalities`, exposed via `getOpenRouterModelModalities` / `provider.getModelModalities` to gate media input.

### 6.4 Dispatch (`src/provider.ts`)

`runLlmStep(provider, options)`, `getModelPricing(provider, model)`, `getContextWindowTokens(provider, model)` and `fetchBalanceSnapshot(provider)` select the implementation by the session's `config.provider`, so DeepSeek and OpenRouter sessions coexist in one process. `costFromUsage(usage, pricing)` computes cost in the provider's currency.

## 7. Usage & Stats

- Per-step and per-turn token/cost/timing stats accumulate into `session.usage` (cumulative) and `turnStats` (per turn), persisted in `state.json`.
- ACP `usage_update` after each LLM step: `used`/`size` (context window from the provider: DeepSeek env or OpenRouter model) and `cost: { amount, currency }` — `CNY` (DeepSeek) or `USD` (OpenRouter) — which Zed renders as the token-usage ring in the agent panel header.
- With `ZEN_AGENT_SHOW_STATS` (default on), a per-turn stats line is emitted as a display-only `agent_message_chunk` (never pushed to `llmMessages`, so it costs no context): `Turn 3 · 4 steps · think 3.2s · answer 8.5s · tools 14.2s` + `in 45.6K · out 3.4K · cache hit 87% · ¥0.043 (session ¥0.12)`.
- The experimental `session/prompt` `usage` field carries cumulative input/output/thought/cache tokens.
- After each stats line, `verifyTurnCost` fetches the provider's balance and logs the delta vs. the locally estimated cost to `log.jsonl` ("turn stats balance verify") — data gathering only, never blocks the turn.

## 8. Slash Commands & Skills

After `session/new`/`load`/`resume`, an `available_commands_update` notification advertises:

| Command        | Behavior                                                                                                                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | Set the session's entire system prompt (multi-line supported) or print the current one; returns `end_turn` without calling the model.                                                                                                             |
| `sandbox`      | Toggle `config.sandbox` (`on`/`off`/status), persisted in `state.json`; refused while `ZEN_AGENT_SANDBOX=1`.                                                                                                                                      |
| `<skill-name>` | One per installed skill: reads `SKILL.md` from `<cwd>/.agents/skills/` or `~/.agents/skills/`, injects it (plus the user's argument) as a user message, and runs a normal turn. Always available, independent of `ZEN_AGENT_SHOW_SKILLS_CATALOG`. |

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
    llm-client.ts      shared OpenAI-compatible SSE client + bash schema
    prompt-content.ts  ACP ContentBlock[] -> user-message parts (text + media)
    media.ts           read_media path resolution/validation
    media-limit.ts     shared ZEN_AGENT_MAX_MEDIA_BYTES limit
    deepseek.ts        DeepSeek provider (pricing, usage, balance)
    openrouter.ts      OpenRouter provider (models catalog, usage, balance)
    provider.ts        per-session provider dispatch + pricing/balance facade
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
- `ai` — `ModelMessage` type only (the AI SDK is **not** used for LLM calls).
- `sonyflake` — message/session id generation.
- dev: `typescript`, `tsx`, `vitest`, `@types/node`.

## 12. Testing

`npm test` (vitest):

- `deepseek.test.ts` / `openrouter.test.ts` — provider SSE behavior against local HTTP servers: live reasoning streaming (timing-sensitive), streaming tool calls, wire format, usage parsing, retries, balance/model endpoints.
- `agent.test.ts` / `agent.graceful.test.ts` — session lifecycle, config options + locking, graceful cancel, stats lines (with `runLlmStep` mocked).
- `skills.test.ts` / `skills-slash.test.ts` / `sandbox.test.ts` / `system-prompt.test.ts` / `tool-execution.test.ts` — skills, sandbox toggling, environment messages, terminal artifacts.
- `media.test.ts` / `prompt-content.test.ts` / `user-parts.test.ts` / `media-flow.test.ts` - media path resolution, prompt-block intake, OpenAI wire mapping (image_url data URIs, input_audio), and the end-to-end read_media turn flow (provider mocked).

## 13. Decisions

1. **Providers**: DeepSeek default, OpenRouter per session via the `provider` config option; one hand-rolled OpenAI-compatible SSE client (`llm-client.ts`) — the AI SDK was dropped because it buffers reasoning-phase streaming (see §6.1).
2. **ACP SDK**: official `@agentclientprotocol/sdk`.
3. **Sessions**: persisted under `<cwd>/.sessions/` with load/list/resume/delete/close.
4. **MCP servers**: ignored; only the `bash` tool is exposed.
5. **Bash execution**: always through the client's ACP terminal; no local subprocess.
6. **Cancellation**: graceful only (the current unit of work completes); hard abort on close/delete or the timeout escape hatch.
