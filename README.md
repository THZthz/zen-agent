# Zen Agent

An [Agent Client Protocol](https://agentclientprotocol.com) v1 coding agent for use with [Zed](https://zed.dev) on WSL2.

- Single tool: `bash`
- Bash execution always uses Zed's ACP terminal (no local bash subprocess)
- No permission prompts (approval policy: `never`)
- Session state is stored in `<project>/.sessions/sessions/<sessionId>.json`
- LLM request/response transcripts are stored as `<project>/.sessions/llm/<sessionId>.jsonl`
- Runtime diagnostics are stored in `<project>/.sessions/logs/zen-agent.log`
- Full terminal output is saved to `<project>/.sessions/terminals/<sessionId>/terminal-<callId>.log` via `script`; the model receives truncated output plus the log path
- LLM provider: Deepseek via the Vercel AI SDK
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

Point Zed's ACP agent at the built entrypoint:

```json
{
  "agent_servers": {
    "Zen Agent": {
      "type": "custom",
      "command": "node",
      "args": ["/home/amias/zen-agent/dist/index.js"],
      "env": {
        "DEEPSEEK_API_KEY": "your-deepseek-api-key"
      }
    }
  }
}
```

The agent reads newline-delimited JSON-RPC from stdin and writes responses to stdout.

## Slash Command

Zen Agent advertises a slash command in the agent panel:

| Command | Description |
| --- | --- |
| `/prompt <text>` | Replace the entire system prompt for this session |
| `/prompt` | Print the current system prompt |

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

Everything after `/prompt` — including newlines — replaces the default system prompt entirely for the rest of the session. Running `/prompt` with no content prints the current effective system prompt.

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
| `DEEPSEEK_CONTEXT_WINDOW` | `1000000` (1M) | Session context window size in tokens, reported via ACP `usage_update` |
| `DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M input tokens served from DeepSeek's context cache (overrides the effective rate for the current period) |
| `DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M input tokens not served from cache (overrides the effective rate for the current period) |
| `DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK` | per-model, peak/off-peak | CNY per 1M output tokens (overrides the effective rate for the current period) |
| `ZEN_AGENT_SHOW_STATS` | `1` | Set to `0` to hide the per-turn stats line in the conversation |
| `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` | `0` (wait forever) | Hard-abort escape hatch: if a graceful cancel (user follow-up or Stop) is pending longer than this, the in-flight LLM step / bash tool is forcibly aborted. `0` waits indefinitely |

## Information Displayed to the User

Zen Agent reports what it can through ACP, and Zed renders it natively in the agent panel:

| Information | How it is displayed |
| --- | --- |
| Context window (used / max) | ACP `usage_update` → token-usage ring in the agent panel header; tooltip shows `Context: 45% • 90K / 200K` |
| Consumption (China yuan) | ACP `usage_update.cost` (`CNY`) → `Cost: ¥0.05` in the same tooltip |
| Turns, steps, thinking/answering time, tool time, cache hit ratio, input/output tokens, turn cost | Per-turn stats line emitted as a separate message, e.g. `Turn 3 · 4 steps · think 3.2s · answer 8.5s · tools 14.2s` + `in 45.6K · out 3.4K · cache hit 87% · ¥0.043 (session ¥0.12)` |
| Bash tool call duration | Appended to each tool call's output card as `⏱ 3.2s` |

The stats line is display-only: it is never added to the LLM message history, so it does not consume context tokens. Cumulative input/output/thought/cache token counts are also returned in the experimental `usage` field of the `session/prompt` response, which Zed reads when its ACP beta flag is enabled.

## Cancellation & Force-Send

When you send a new message (or press Stop) while Zen Agent is working, Zed sends an ACP `session/cancel` notification and **waits for the current turn to respond before delivering your new message** (`thread_view.rs`). Zen Agent honors this with a *graceful cancel* — it never kills the agent mid-work:

- **While the model is thinking/answering** — the current LLM step runs to completion, then the turn ends with `stopReason: "cancelled"`. If the step proposed tool calls, they are discarded (your follow-up supersedes them).
- **While a bash tool is running** — the command finishes and its results are saved to the conversation history (so the next turn has full context), then the turn ends with `stopReason: "cancelled"`. The tool card shows `completed`.
- **Between steps** — the turn stops at the next boundary check.

A hard abort (terminal killed, stream cut) only happens on `session/close`, `session/delete`, or after `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` as an escape hatch for runaway commands. Zed's Stop button uses the same `session/cancel` notification as force-send (the protocol carries no reason field), so it is graceful too.

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
| `sessions/<sessionId>.json` | Session state for resume/load |
| `llm/<sessionId>.jsonl` | LLM request/response transcript |
| `logs/zen-agent.log` | Runtime diagnostic log |
| `terminals/<sessionId>/terminal-<callId>.log` | Full terminal output for a bash tool call |
| `terminals/<sessionId>/terminal-<callId>.sh` | Saved bash command script for a tool call |

## Development

```bash
npm run typecheck
npm test
npm run build
```
