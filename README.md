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
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible base URL |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Fallback model when no session config is present |
| `ZEN_AGENT_MAX_TURN_STEPS` | `25` | Maximum number of LLM/tool rounds per user prompt |

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
