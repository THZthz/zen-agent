# Zen Agent

An [Agent Client Protocol](https://agentclientprotocol.com) v1 coding agent for use with [Zed](https://zed.dev) on WSL2.

- Single tool: `bash`
- Bash execution always uses Zed's ACP terminal (no local bash subprocess)
- No permission prompts (approval policy: `never`)
- Session state is stored in `<project>/.sessions/<sessionId>/state.json`
- LLM request/response transcripts are stored as `<project>/.sessions/<sessionId>/llm.jsonl`
- Runtime diagnostics are stored in a per-startup log at `<project>/.sessions/client/<startupTimestamp>_<uuid>/log.jsonl`, including per-LLM-step and per-turn stats (tokens, cache hit ratio, cost, timing)
- Bash command scripts are saved to `<project>/.sessions/<sessionId>/terminals/input-<timestamp>-<callId>.sh`; full terminal output is saved to `<project>/.sessions/<sessionId>/terminals/output-<timestamp>-<callId>.log` via `script`; the model receives truncated output plus the log path
- Bash tool call cards survive Zed restarts: each call registers a display-only terminal (`_meta.terminal_info`) and streams its output into it (`_meta.terminal_output`/`terminal_exit`), so a resumed session re-renders the terminal card with its output instead of dropping it. On `session/load` the replay layer synthesizes the same display-terminal metadata for sessions created before this mechanism existed, so old bash cards render with an expand toggle and visible output too
- LLM provider: Deepseek via its OpenAI-compatible chat completions API, with a direct SSE client (`runLlmStep` in `src/deepseek.ts`)
- Environment context (working directory, session time, git branch/commit/status) is sent to the model as a `user` message named `Environment`, separate from the system prompt. It is frozen into the session at creation and persisted, so the request prefix stays byte-identical and DeepSeek's context cache keeps hitting across steps and session restarts; on every load/resume a fresh continuation notification is appended at the end of the conversation
- The human's own messages are sent with `name` set to `git config user.name` (defaults to `User`)
- Selectable models:
  - `deepseek-v4-flash`
  - `deepseek-v4-pro`
- Selectable thinking effort:
  - `off`
  - `high`
  - `max`
- Streams LLM thinking/reasoning content to Zed as `agent_thought_chunk`

## Requirements

- Node.js 22+
- WSL2/Linux
- `DEEPSEEK_API_KEY` environment variable

## Setup

```bash
npm install
npm run build
```

## Run

```bash
DEEPSEEK_API_KEY=... node dist/index.js
```

## Zed Configuration

Point Zed's ACP agent at the built entrypoint. The recommended setup runs
the agent inside a bubblewrap sandbox (see [Sandboxing](#sandboxing-with-bubblewrap)):

```json
{
  "agent_servers": {
    "Zen Agent": {
      "default_config_options": {
        "model": "deepseek-v4-flash",
        "thinking_effort": "max"
      },
      "type": "custom",
      "command": "/home/amias/projects/zen-agent/bin/zen-agent-bwrap.sh",
      "args": [],
      "env": {
        "DEEPSEEK_API_KEY": "your-deepseek-api-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        "ZEN_AGENT_MAX_TURN_STEPS": "1000",
        "ZEN_AGENT_SANDBOX": "1"
      }
    }
  }
}
```

Without the sandbox, point the command at `node` directly:

```json
{
  "agent_servers": {
    "Zen Agent": {
      "type": "custom",
      "command": "node",
      "args": ["/home/amias/projects/zen-agent/dist/index.js"],
      "env": {
        "DEEPSEEK_API_KEY": "your-deepseek-api-key"
      }
    }
  }
}
```

The agent reads newline-delimited JSON-RPC from stdin and writes responses to stdout.

## Sandboxing with Bubblewrap

`bin/zen-agent-bwrap.sh` runs the agent's node process inside `bwrap` with the
following policy:

- `--bind / /` — the whole root filesystem behaves exactly as on the host
  (writable where it was writable, read-only where it was read-only).
- `--ro-bind /mnt /mnt` — `/mnt` (Windows drives `C:`, `D:`, ..., WSL mounts)
  becomes **read-only**: reads still work, every write fails with `EROFS`.
- `--dev /dev`, `--bind /dev/pts /dev/pts`, `--tmpfs /dev/shm` — a fresh
  device filesystem plus host PTYs so terminals and `/dev/null` keep working.

The sandboxed process runs as your normal uid in a new user + mount
namespace; it cannot remount `/mnt` read-write.

Important: the agent's `bash` tool executes in a PTY owned by Zed on the
**host**, outside the agent process's sandbox. To sandbox the bash tool too,
set `ZEN_AGENT_SANDBOX=1` (as in the config above) or run `/sandbox on` in
the session: every bash call is then wrapped in its own `bwrap` with the
same `/mnt` read-only policy, so the agent can never write into `/mnt` even
through its bash tool.

The bash-tool sandbox also shadows `rm`, `grep` and `find`: inside the bwrap
namespace their real binaries are replaced (read-only) by a shim that
refuses to run and tells the agent to use `trash`, `rg` and `fdfind`
instead. This only happens inside the sandbox's mount namespace — processes
and scripts on the host keep using the real `rm`/`grep`/`find` untouched, so
nothing else on the machine is affected.

## Slash Commands

Zen Agent advertises slash commands in the agent panel:

| Command | Description |
| --- | --- |
| `/prompt <text>` | Replace the entire system prompt for this session |
| `/prompt` | Print the current system prompt |
| `/sandbox on` | Wrap every bash tool call in its own `bwrap` sandbox for this session (persisted across restarts) |
| `/sandbox off` | Disable the per-session bash sandbox (cannot turn it off while `ZEN_AGENT_SANDBOX=1`) |
| `/sandbox` | Show the current bash sandbox status |
| `/skill-name <input>` | Invoke an installed Agent Skill, one command per skill (see [Skills](#skills-skillssh)) |

Examples:

Single-line:

```text
/prompt Always prefer safe refactors and add tests.
```

Multi-line:

```text
/prompt
Always prefer safe refactors.
Add tests for all changes.
```

Sandboxing:

```text
/sandbox on
/sandbox
```

Unlike `ZEN_AGENT_SANDBOX=1` (a global env policy), `/sandbox` toggles the
per-session `config.sandbox` flag at runtime and persists it in the session
state. The env policy still applies on top: with `ZEN_AGENT_SANDBOX=1` the
sandbox is always on and `/sandbox off` is refused.

Everything after `/prompt` — including newlines — replaces the default system prompt entirely for the rest of the session. Running `/prompt` with no content prints the current effective system prompt.

## Skills (skills.sh)

Zen Agent can use [Agent Skills](https://www.skills.sh/) — the open skill
format used by the skills.sh registry. Skills are folders containing a
`SKILL.md` (YAML frontmatter + Markdown instructions) plus optional
`scripts/`, `references/` and `assets/`.

Zed loads skills from `~/.agents/skills/` (global) and
`<project>/.agents/skills/` (project-local), and Zen Agent reads the exact
same locations — so you install skills exactly like you would for Zed:

```bash
# Project-local (committed with the repo)
npx skills add vercel-labs/agent-skills -a zed

# Global (available in every project)
npx skills add vercel-labs/agent-skills -a zed -g
```

By default Zen Agent stays minimal and passes **no skill information** to
the model. Set `ZEN_AGENT_SHOW_SKILLS_CATALOG=1` to opt in: at session
creation Zen Agent scans both directories and freezes a compact catalog
(skill name, description, scope, and the `cat` command to load it) into the
environment message, alongside the working directory and git state.

Skill invocation is **by hand only**: the model loads a skill's `SKILL.md`
with the bash tool — the read shows up as a normal terminal card — only
when you explicitly ask for that skill by name; it never loads a skill on
its own.

Every installed skill also gets its own slash command, advertised alongside
`/prompt` and `/sandbox`: `/grill-me <grill what>` loads `grill-me`'s
`SKILL.md` straight into the conversation and runs a model turn that follows
it (so skills with `disable-model-invocation: true` — user-invocation only,
like `grill-me` — are fully usable). Skill slash commands are always
available, **regardless of `ZEN_AGENT_SHOW_SKILLS_CATALOG`**: that flag only
controls whether the skill catalog is listed in the environment message, not
whether `/skill-name` works.

Notes:

- The catalog is frozen when the session is created (so the cached LLM
  prefix stays byte-identical). Skills installed after that are picked up by
  the next session.
- Zed only exposes skills to its built-in agent, not to ACP agents, which is
  why Zen Agent discovers them itself.

## Models

| Config value | API model version | Context | Max output | Thinking |
| --- | --- | --- | --- | --- |
| `deepseek-v4-flash` | `DeepSeek-V4-Flash-0731` | 1M | 384K | non-thinking + thinking (default) |
| `deepseek-v4-pro` | `DeepSeek-V4-Pro-0813` | 1M | 384K | non-thinking + thinking (default) |

Both models support JSON output, tool calls, the Responses API, and the Anthropic API.

## Session Configuration

When a session is created or loaded, Zed can display two configuration selectors:

| Option | Values |
| --- | --- |
| Model | `deepseek-v4-flash`, `deepseek-v4-pro` |
| Thinking Effort | `off`, `high`, `max` |

These are exposed as ACP session config options and can be changed with `session/set_config_option`.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | — | Deepseek API key (required) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible base URL (official; `https://api.deepseek.com/v1` also works) |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Fallback model when no session config is present |
| `ZEN_AGENT_MAX_TURN_STEPS` | `25` | Maximum number of LLM/tool rounds per user prompt |
| `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` | `50000` | Max UTF-8 bytes of bash terminal output sent to the model per tool call (keeps the tail; the full output stays in the log file and the terminal card) |
| `DEEPSEEK_CONTEXT_WINDOW` | `1000000` (1M) | Session context window size in tokens, reported via ACP `usage_update` |
| `DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M input tokens served from DeepSeek's context cache (overrides the effective rate for the current period) |
| `DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M input tokens not served from cache (overrides the effective rate for the current period) |
| `DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M output tokens (overrides the effective rate for the current period) |
| `ZEN_AGENT_SHOW_STATS` | `1` | Set to `0` to hide the per-turn stats line in the conversation |
| `ZEN_AGENT_SHOW_SKILLS_CATALOG` | — | Set to `1` to inject the installed Agent Skills catalog (name, description, load command) into the session's environment message (see [Skills](#skills-skillssh)); off by default |
| `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` | `0` (wait forever) | Hard-abort escape hatch: if a graceful cancel (user follow-up or Stop) is pending longer than this, the in-flight LLM step / bash tool is forcibly aborted. `0` waits indefinitely |
| `ZEN_AGENT_SANDBOX` | — | Set to `1` to run every bash tool call inside `bwrap` with `/mnt` mounted read-only (see [Sandboxing](#sandboxing-with-bubblewrap)) |
| `ZEN_AGENT_SANDBOX_CMD` | default bwrap policy | Override the exact bwrap command used for sandboxed bash tool calls |
| `ZEN_AGENT_SANDBOX_BLOCK_SHIM` | repo `bin/zen-agent-sandbox-block.sh` | Shim mounted (read-only) over `rm`/`grep`/`find` inside the bash-tool sandbox; refuses to run and suggests `trash`/`rg`/`fdfind` |

## Information Displayed to the User

Zen Agent reports what it can through ACP, and Zed renders it natively in the agent panel:

| Information | How it is displayed |
| --- | --- |
| Context window (used / max) | ACP `usage_update` → token-usage ring in the agent panel header; tooltip shows `Context: 45% • 90K / 200K` |
| Consumption (China yuan) | ACP `usage_update.cost` (`CNY`) → `Cost: ¥0.05` in the same tooltip |
| Turns, steps, thinking/answering time, tool time, cache hit ratio, input/output tokens, turn cost | Per-turn stats line emitted as a separate message, e.g. `Turn 3 · 4 steps · think 3.2s · answer 8.5s · tools 14.2s` + `in 45.6K · out 3.4K · cache hit 87% · ¥0.043 (session ¥0.12)` |
| Bash tool call duration | Appended to each tool call's output card as `⏱ 3.2s` |

The stats line is display-only: it is never added to the LLM message history, so it does not consume context tokens. Cumulative input/output/thought/cache token counts are also returned in the experimental `usage` field of the `session/prompt` response, which Zed reads when its ACP beta flag is enabled.

### Stats survive resume

Cumulative stats (`usage`) and per-turn stats (`turnStats`) are persisted in
`<project>/.sessions/<sessionId>/state.json`, so a resumed session keeps its
turns, steps, thinking/answering/tool time, cache hit ratio, tokens and CNY
cost across Zed restarts.


## Cancellation & Force-Send

When you send a new message (or press Stop) while Zen Agent is working, Zed sends an ACP `session/cancel` notification and **waits for the current turn to respond before delivering your new message** (`thread_view.rs`). Zen Agent honors this with a *graceful cancel* — it never kills the agent mid-work:

- **While the model is thinking/answering** — the current LLM step runs to completion, then the turn ends with `stopReason: "cancelled"`. If the step proposed tool calls, they are discarded (your follow-up supersedes them).
- **While a bash tool is running** — the command finishes and its results are saved to the conversation history (so the next turn has full context), then the turn ends with `stopReason: "cancelled"`. The tool card shows `completed`.
- **Between steps** — the turn stops at the next boundary check.

A hard abort (terminal killed, stream cut) only happens on `session/close`, `session/delete`, or after `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` as an escape hatch for runaway commands. Zed's Stop button uses the same `session/cancel` notification as force-send (the protocol carries no reason field), so it is graceful too.

## Live Streaming of Thinking & Answers

Zen Agent talks to DeepSeek's OpenAI-compatible chat completions endpoint **directly** (own SSE parser in `runLlmStep`) instead of going through the Vercel AI SDK's `streamText` + `@ai-sdk/openai`. Reason:

`@ai-sdk/openai` runs `throwIfOpenAIStreamErrorBeforeOutput`, which reads ahead from the response until it sees the first "output" chunk (non-empty `delta.content` or a tool call). In DeepSeek thinking mode the reasoning phase only carries `delta.reasoning_content`, which that check cannot see — so the SDK swallowed the **entire reasoning phase** and delivered it as a buffered burst once the answer started. Visually the thinking block appeared to not stream at all (and the answer was delayed behind the flushed backlog).

Our direct client parses each SSE event as it arrives, so:

- `delta.reasoning_content` is forwarded to Zed as `agent_thought_chunk` **live**, token by token.
- `delta.content` is forwarded as `agent_message_chunk` live.
- Streaming `delta.tool_calls` fragments are accumulated per index and emitted as complete tool calls.
- Usage is read from DeepSeek's raw chunk (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens_details.reasoning_tokens`), which the SDK's zod schema strips — so the cache hit ratio and CNY cost are now accurate.
- Retries (429/5xx) happen only before the first byte, so an in-flight stream is never replayed.

Default pricing (official DeepSeek V4 pricing, CNY per 1M tokens). Off-peak price is half the peak price. Peak hours are Beijing time 09:00-12:00 and 14:00-18:00; all other hours are off-peak:

| Model | Period | Cache hit input | Cache miss input | Output |
| --- | --- | --- | --- | --- |
| `deepseek-v4-flash` | Peak | ¥0.10 | ¥3.00 | ¥9.00 |
| `deepseek-v4-flash` | Off-peak | ¥0.05 | ¥1.50 | ¥4.50 |
| `deepseek-v4-pro` | Peak | ¥0.30 | ¥9.00 | ¥27.00 |
| `deepseek-v4-pro` | Off-peak | ¥0.15 | ¥4.50 | ¥13.50 |

The cost shown in Zed's usage tooltip (and the per-turn stats line) is computed with the rate for the current Beijing-time period.

## Logs

Inside each project's `.sessions/` directory:

| Path | Purpose |
| --- | --- |
| `<sessionId>/state.json` | Session state for resume/load |
| `<sessionId>/llm.jsonl` | LLM request/response transcript |
| `<sessionId>/terminals/input-<timestamp>-<callId>.sh` | Saved bash command script for a tool call |
| `<sessionId>/terminals/output-<timestamp>-<callId>.log` | Full terminal output for a bash tool call |
| `client/<startupTimestamp>_<uuid>/log.jsonl` | Per-startup runtime diagnostic log |

## Development

```bash
npm run typecheck
npm test
npm run build
```
