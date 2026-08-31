# Zen Agent

An [Agent Client Protocol](https://agentclientprotocol.com) v1 coding agent for [Zed](https://zed.dev) on WSL2/Linux. It gives the model a single, unrestricted `bash` tool — executed through Zed's terminal, with no approval prompts. Implementation details live in [SPEC.md](SPEC.md).

## Features

- Single `bash` tool; no approval prompts
- Image/audio input on multimodal OpenRouter models: paste, drag & drop or @-mention images in Zed, and the model can load local screenshots/audio itself via a `read_media` tool
- Live streaming of thinking and answers
- DeepSeek (default) and OpenRouter providers, plus any OpenAI-compatible endpoint via endpoint + API key (auto-discovers models); chosen per session
- Persistent sessions (`<project>/.sessions/`) with resume/load across Zed restarts
- Slash commands: `/prompt`, `/sandbox`, `/tools`, one per installed skill
- [Agent Skills](https://www.skills.sh/) support
- Token usage, cache-hit ratio, cost and timing reporting (CNY for DeepSeek, USD for OpenRouter)
- Optional [bubblewrap](#sandboxing) sandboxing with read-only `/mnt`

## Requirements

- Node.js 22+
- WSL2/Linux
- `DEEPSEEK_API_KEY` and/or `OPENROUTER_API_KEY` (or your own provider's key)

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

Point Zed's ACP agent at the built entrypoint (`agent_servers` → type `custom`). The agent reads newline-delimited JSON-RPC from stdin and writes responses to stdout.

```json
{
  "agent_servers": {
    "Zen Agent": {
      "default_config_options": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "thinking_effort": "max"
      },
      "type": "custom",
      "command": "node",
      "args": ["/home/amias/projects/zen-agent/dist/index.js"],
      "env": {
        "DEEPSEEK_API_KEY": "your-deepseek-api-key",
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "OPENROUTER_SITE_URL": "https://zed.dev",
        "OPENROUTER_APP_NAME": "Zen Agent"
      }
    }
  }
}
```

### Sandboxed

To run the agent process itself inside bubblewrap, use `bin/zen-agent-bwrap.sh` as `command` and add `"ZEN_AGENT_SANDBOX": "1"` to `env` to sandbox the bash tool as well (see [Sandboxing](#sandboxing)).

## Session Settings

Zed shows three selectors per session (also settable via `default_config_options`):

| Option          | Values                                                                                                                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Provider        | `deepseek`, `openrouter`, plus every `ZEN_AGENT_PROVIDERS` entry                                                                                                                                                                            |
| Model           | DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro` · OpenRouter/auto-discovered: live model catalog (fetched through pi-ai, cached in `$XDG_DATA_HOME/zen-agent/models/`; `openrouter/free` default, any slug via env or `set_config_option`) |
| Thinking effort | DeepSeek: `off`, `low`, `high`, `max` · allowlist providers: `off` plus the selected model's supported levels, sorted ascending · generic: full ladder (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`)                          |

Provider, model, thinking effort, the `/tools` toggle and the `/prompt` system-prompt setter are **locked after the session's first message** — set them before you start the conversation (status queries stay available).

### Custom Providers

Add any OpenAI-compatible provider with a base URL + API key; models are auto-discovered from `GET {baseUrl}/models`:

```json
{
  "env": {
    "GROQ_API_KEY": "gsk_...",
    "ZEN_AGENT_PROVIDERS": "[{\"id\":\"groq\",\"name\":\"Groq\",\"baseUrl\":\"https://api.groq.com/openai/v1\",\"apiKeyEnv\":\"GROQ_API_KEY\",\"defaultModel\":\"llama-3.3-70b-versatile\"}]"
  }
}
```

Or point `ZEN_AGENT_PROVIDERS_FILE` at a JSON file with the same array shape. Optional fields: `name`, `currency` (default `USD`), `apiKeyEnv` (omit for keyless local endpoints like Ollama), and `models` (a static list; omit it to auto-discover). Set `ZEN_AGENT_DEFAULT_PROVIDER` to change the default session provider from `deepseek`.

### Image & Audio Input

Sessions on OpenRouter models whose catalog entry lists `image`/`audio` in `architecture.input_modalities` accept media:

- **Attach**: paste from the clipboard, drag & drop into the panel, or @-mention an image file. Zed sends it as an ACP content block; the transcript shows the original image while the LLM message carries the payload (`image_url` data URI / `input_audio`).
- **Self-directed reading**: vision/audio-capable sessions also get a `read_media` tool. When the user references a screenshot or recording by path, the model loads and perceives it itself instead of asking for a description; payloads ride in a synthetic user message inserted before the assistant tool-call (so the request never ends with the media message — GLM/Z.AI's cache drops to 0% on that shape). OpenRouter requests carry the session id so Z.AI pins the cache to the conversation.
- On text-only models (all DeepSeek models, non-vision OpenRouter slugs) attached media degrades to a placeholder note in the prompt, so the turn still runs. Media above `ZEN_AGENT_MAX_MEDIA_BYTES` (default 10 MB decoded) is omitted with a note.

## Slash Commands

| Command                 | Description                                                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/prompt <text>`        | Replace the session's system prompt (no argument: print the current one); locked after the first user message                                                                                                         |
| `/sandbox on\|off`      | Toggle the per-session bash sandbox (no argument: show status)                                                                                                                                                        |
| `/tools on\|off`        | Enable/disable all tools (`bash`, `read_media`) for the session (no argument: show status); locked after the first user message. `off` makes the session chat-only (environment messages dropped); `on` restores them |
| `/<skill-name> <input>` | Invoke an installed Agent Skill                                                                                                                                                                                       |

## Skills

Zen Agent reads Agent Skills from the same locations as Zed (`~/.agents/skills/` and `<project>/.agents/skills/`); install them the usual way:

```bash
npx skills add vercel-labs/agent-skills -a zed    # project-local
npx skills add vercel-labs/agent-skills -a zed -g # global
```

By default no skill information reaches the model; set `ZEN_AGENT_SHOW_SKILLS_CATALOG=1` to freeze a catalog into the session's environment message. Skills are invoked by hand only — the model never loads one on its own.

## Sandboxing

`bin/zen-agent-bwrap.sh` runs the agent inside `bwrap` with `/mnt` (Windows drives) mounted read-only. The bash tool runs in a PTY owned by Zed on the host, so it needs its own sandbox: set `ZEN_AGENT_SANDBOX=1` (or run `/sandbox on`) to wrap every bash call in `bwrap` with the same `/mnt` policy. Inside that sandbox `rm`, `grep` and `find` are shadowed by shims suggesting `trash`, `rg` and `fdfind`; the host is unaffected.

## Environment Variables

| Variable                                 | Default                          | Description                                                                                                       |
| ---------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `DEEPSEEK_API_KEY`                       | —                                | DeepSeek API key (required for DeepSeek sessions)                                                                 |
| `DEEPSEEK_BASE_URL`                      | `https://api.deepseek.com`       | DeepSeek-compatible base URL                                                                                      |
| `DEEPSEEK_MODEL`                         | `deepseek-v4-flash`              | Fallback DeepSeek model                                                                                           |
| `DEEPSEEK_CONTEXT_WINDOW`                | `1000000`                        | Context window size in tokens for the usage display                                                               |
| `DEEPSEEK_PRICE_CACHE_HIT_CNY_PER_MTOK`  | per model, peak/off-peak         | Override DeepSeek cache-hit input price (CNY per 1M tokens)                                                       |
| `DEEPSEEK_PRICE_CACHE_MISS_CNY_PER_MTOK` | per model, peak/off-peak         | Override DeepSeek cache-miss input price                                                                          |
| `DEEPSEEK_PRICE_OUTPUT_CNY_PER_MTOK`     | per model, peak/off-peak         | Override DeepSeek output price                                                                                    |
| `OPENROUTER_API_KEY`                     | —                                | OpenRouter API key (required for OpenRouter sessions)                                                             |
| `OPENROUTER_BASE_URL`                    | `https://openrouter.ai/api/v1`   | OpenRouter-compatible base URL                                                                                    |
| `OPENROUTER_MODEL`                       | `openrouter/free`                | Fallback OpenRouter model slug                                                                                    |
| `OPENROUTER_PROVIDER_SORT`               | `price`                          | OpenRouter provider routing sort (`price`, `throughput`, `latency`; empty disables the `provider` block)          |
| `OPENROUTER_SITE_URL`                    | —                                | Sent as `HTTP-Referer` (app attribution)                                                                          |
| `OPENROUTER_APP_NAME`                    | —                                | Sent as `X-Title` (app attribution)                                                                               |
| `ZEN_AGENT_PROVIDERS`                    | —                                | JSON array of user-defined providers (endpoint + API key env + optional model list; omit models to auto-discover) |
| `ZEN_AGENT_PROVIDERS_FILE`               | —                                | Path to a JSON file with the same provider array                                                                  |
| `ZEN_AGENT_DEFAULT_PROVIDER`             | `deepseek`                       | Default provider for new sessions                                                                                 |
| `ZEN_AGENT_MAX_TURN_STEPS`               | `25`                             | Max LLM/tool rounds per user prompt                                                                               |
| `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT`   | `50000`                          | Max bytes of bash output sent to the model per tool call (tail kept)                                              |
| `ZEN_AGENT_SHOW_STATS`                   | `1`                              | Set `0` to hide the per-turn stats line                                                                           |
| `ZEN_AGENT_SHOW_SKILLS_CATALOG`          | off                              | Inject the skills catalog into the environment message                                                            |
| `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS`   | `0` (wait forever)               | Hard-abort escape hatch for pending cancels                                                                       |
| `ZEN_AGENT_SANDBOX`                      | —                                | `1` = always sandbox bash tool calls                                                                              |
| `ZEN_AGENT_SANDBOX_CMD`                  | default policy                   | Override the bwrap command for bash tool calls                                                                    |
| `ZEN_AGENT_SANDBOX_BLOCK_SHIM`           | `bin/zen-agent-sandbox-block.sh` | Override the shim shadowing `rm`/`grep`/`find`                                                                    |

## Displayed Info

- Context usage ring and cost in the agent panel header (ACP `usage_update`)
- Per-turn stats line: steps, thinking/answering/tool time, cache hit ratio, tokens, cost (`ZEN_AGENT_SHOW_STATS`)
- Tool call cards with terminal-style output and duration

Stats are display-only (never sent to the model) and survive Zed restarts via the persisted session.

## Development

```bash
npm run typecheck
npm test
npm run build
```
