# Zen Agent

An [Agent Client Protocol](https://agentclientprotocol.com) v1 coding agent for [Zed](https://zed.dev) on WSL2/Linux. It gives the model a single, unrestricted `bash` tool — executed through Zed's terminal, with no approval prompts. Implementation details live in [SPEC.md](SPEC.md).

## Features

- Single `bash` tool; no approval prompts
- Image/audio input on multimodal OpenRouter models: paste, drag & drop or @-mention images in Zed, and the model can load local screenshots/audio itself via a `read_media` tool
- Live streaming of thinking and answers
- User-defined OpenAI-compatible providers (no built-ins): declare models in `ZEN_AGENT_PROVIDERS` or auto-discover them with `fetchModels: true`
- Persistent sessions in a single SQLite database (`ZEN_AGENT_DB_FILE`) with resume/load across Zed restarts
- Slash commands: `/prompt`, `/sandbox`, `/tools`, one per installed skill
- [Agent Skills](https://www.skills.sh/) support
- Token usage, cache-hit ratio, cost and timing reporting (per-provider currency)
- Optional [bubblewrap](#sandboxing) sandboxing with read-only `/mnt`

## Requirements

- Node.js 24+ (`node:sqlite` is built in)
- WSL2/Linux
- At least one provider configured via `ZEN_AGENT_PROVIDERS` / `ZEN_AGENT_PROVIDERS_FILE` (no providers are built in)

## Setup

```bash
npm install
npm run build
```

## Run

```bash
ZEN_AGENT_PROVIDERS_FILE="$HOME/.config/zen-agent/providers.json" \
DEEPSEEK_API_KEY=... node dist/index.js
```

There are **no built-in providers** — see [Providers](#providers-default-zen_agent_providers-config) for a ready-to-copy `providers.json` that declares DeepSeek, OpenRouter, Z.ai and Groq.

## Zed Configuration

Point Zed's ACP agent at the built entrypoint (`agent_servers` → type `custom`). The agent reads newline-delimited JSON-RPC from stdin and writes responses to stdout.

```json
{
  "agent_servers": {
    "Zen Agent": {
      "default_config_options": {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "thinking_effort": "max"
      },
      "type": "custom",
      "command": "node",
      "args": ["/home/amias/projects/zen-agent/dist/index.js"],
      "env": {
        "XDG_DATA_HOME": "/home/amias/.local/share",
        "ZEN_AGENT_PROVIDERS_FILE": "/home/amias/.config/zen-agent/providers.json",
        "ZEN_AGENT_DEFAULT_PROVIDER": "deepseek",
        "ZEN_AGENT_MAX_TURN_STEPS": "25",
        "ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT": "50000",
        "ZEN_AGENT_CHAT_TIMEOUT_MS": "6600000",
        "ZEN_AGENT_CHAT_RPM": "0",
        "ZEN_AGENT_MAX_MEDIA_BYTES": "10000000",
        "ZEN_AGENT_MAX_RESOURCE_BYTES": "262144",
        "ZEN_AGENT_SHOW_STATS": "1",
        "ZEN_AGENT_SHOW_SKILLS_CATALOG": "0",
        "ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS": "0",
        "ZEN_AGENT_SANDBOX": "0",
        "ZEN_AGENT_SANDBOX_CMD": "",
        "ZEN_AGENT_SANDBOX_BLOCK_SHIM": "/home/amias/projects/zen-agent/bin/zen-agent-sandbox-block.sh",
        "DEEPSEEK_API_KEY": "your-deepseek-api-key",
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "ZAI_API_KEY": "zai-...",
        "GROQ_API_KEY": "gsk_..."
      }
    }
  }
}
```

- Use `ZEN_AGENT_PROVIDERS` (inline JSON) **or** `ZEN_AGENT_PROVIDERS_FILE` (a JSON file) — not both.
- The `*_API_KEY` variables above are the `apiKeyEnv` names referenced by the provider definitions in `providers.json`; any env var name works as long as the provider config points at it.

### Sandboxed

To run the agent process itself inside bubblewrap, use `bin/zen-agent-bwrap.sh` as `command` and add `"ZEN_AGENT_SANDBOX": "1"` to `env` to sandbox the bash tool as well (see [Sandboxing](#sandboxing)).

## Session Settings

Zed shows three selectors per session (also settable via `default_config_options`):

| Option          | Values                                                                                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Provider        | every `ZEN_AGENT_PROVIDERS` entry (no built-ins; `ZEN_AGENT_DEFAULT_PROVIDER` or the first entry is the default)                                                                                                          |
| Model           | declared models, or the live catalog when `fetchModels: true` (fetched through pi-ai, cached in `$XDG_DATA_HOME/zen-agent/models/`; any slug via `set_config_option`)                                                     |
| Thinking effort | full ladder (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) by default; models with a declared `thinkingEfforts` list show exactly that list (see [Providers](#providers-default-zen_agent_providers-config)) |

Provider, model, thinking effort, the `/tools` toggle and the `/prompt` system-prompt setter are **locked after the session's first message** — set them before you start the conversation (status queries stay available).

### Providers (default `ZEN_AGENT_PROVIDERS` config)

There are **no built-in providers** — every provider is defined by you. The default configuration below is a ready-to-copy starting point: DeepSeek and OpenRouter (the former built-ins) plus Z.ai and Groq. It exercises every provider/model field. Save it as `~/.config/zen-agent/providers.json` (or inline it into `ZEN_AGENT_PROVIDERS` as a JSON-escaped string).

```json
[
  {
    "id": "deepseek",
    "name": "DeepSeek",
    "baseUrl": "https://api.deepseek.com",
    "apiKeyEnv": "DEEPSEEK_API_KEY",
    "defaultModel": "deepseek-v4-flash",
    "currency": "CNY",
    "thinkingMode": "deepseek",
    "models": [
      {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "description": "Fast model for everyday coding tasks",
        "contextLength": 1000000,
        "cost": { "inputPerM": 3, "outputPerM": 9 }
      },
      {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "description": "More powerful model for complex tasks",
        "contextLength": 1000000,
        "cost": { "inputPerM": 9, "outputPerM": 27 }
      }
    ]
  },
  {
    "id": "openrouter",
    "name": "OpenRouter",
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKeyEnv": "OPENROUTER_API_KEY",
    "defaultModel": "openrouter/free",
    "currency": "USD",
    "fetchModels": true,
    "models": [
      {
        "id": "openrouter/free",
        "name": "OpenRouter Free",
        "description": "OpenRouter's free-tier routing model",
        "contextLength": 128000,
        "cost": { "inputPerM": 0, "outputPerM": 0 }
      }
    ]
  },
  {
    "id": "zai",
    "name": "Z.ai",
    "baseUrl": "https://api.z.ai/api/paas/v4",
    "apiKeyEnv": "ZAI_API_KEY",
    "defaultModel": "glm-5.3-flash",
    "currency": "USD",
    "models": [
      {
        "id": "glm-5.3",
        "name": "GLM 5.3",
        "description": "Text-only flagship, 1M context (mandatory reasoning)",
        "contextLength": 1048576,
        "cost": { "inputPerM": 1.4, "outputPerM": 4.4 },
        "thinkingEfforts": ["low", "high", "max"]
      },
      {
        "id": "glm-5.3-flash",
        "name": "GLM 5.3 Flash",
        "description": "Native multimodal, 1M context",
        "contextLength": 1048576,
        "cost": { "inputPerM": 0.075, "outputPerM": 0.25 },
        "modalities": ["image"],
        "thinkingEfforts": ["off", "low", "high", "max"]
      }
    ]
  },
  {
    "id": "groq",
    "name": "Groq",
    "baseUrl": "https://api.groq.com/openai/v1",
    "apiKeyEnv": "GROQ_API_KEY",
    "defaultModel": "llama-3.3-70b-versatile",
    "currency": "USD",
    "fetchModels": true
  }
]
```

**Provider fields**

| Field          | Required | Description                                                                                                                                                                                                                                                                                               |
| -------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`           | yes      | Unique provider id used in sessions and `ZEN_AGENT_DEFAULT_PROVIDER`                                                                                                                                                                                                                                      |
| `baseUrl`      | yes      | OpenAI-compatible base URL, e.g. `https://api.deepseek.com`                                                                                                                                                                                                                                               |
| `name`         | no       | Display name (defaults to `id`)                                                                                                                                                                                                                                                                           |
| `apiKeyEnv`    | no       | Env var holding the API key; omit for keyless local endpoints (Ollama, LM Studio)                                                                                                                                                                                                                         |
| `defaultModel` | no*      | Fallback model; required when `fetchModels: true`, otherwise defaults to the first declared model                                                                                                                                                                                                         |
| `currency`     | no       | Billing currency for cost reporting (default `USD`)                                                                                                                                                                                                                                                       |
| `thinkingMode` | no       | `openai` (default): send `reasoning_effort`, `off` omits the field · `deepseek`: send `thinking: {type}` so `off` truly disables thinking; effort values pass through unchanged (DeepSeek's API auto-maps them)                                                                                           |
| `fetchModels`  | no       | `true` auto-discovers models from `GET {baseUrl}/models`; declared `models` are still offered alongside (default `false`)                                                                                                                                                                                 |
| `models`       | no*      | Declared model list; required when `fetchModels` is false/absent. Entries: `id` (required), `name`, `description`, `contextLength`, `cost` (`{inputPerM, outputPerM}` per 1M tokens in the provider's currency), `modalities` (`["image"]` / `["audio"]`; `text` implicit), `thinkingEfforts` (see below) |

**Per-model `thinkingEfforts`** — restrict the session's thinking-effort selector to the listed values (in declared order) and remap anything else:

- `"thinkingEfforts": ["off", "low", "high", "max"]` — the selector shows exactly these; unsupported session values (e.g. `minimal`, `medium`, `xhigh`) are sent as the nearest declared effort (ties resolve upward: `medium` → `high`, `xhigh` → `max`); `off` omits the field (provider default applies).
- `"thinkingEfforts": ["low", "high", "max"]` (no `off`) — mandatory reasoning: selecting `off` sends the lowest declared effort (`low`).
- Omit the field to accept the full ladder (passthrough) — the session value is sent verbatim as `reasoning_effort` and `off` omits the field (the provider picks its default). With `thinkingMode: "deepseek"`, `off` instead sends `thinking: {type: "disabled"}` so thinking is actually turned off.

\* A provider must declare `models`, set `fetchModels: true`, or both — otherwise configuration fails with a clear error. Provider ids must be unique.

### Image & Audio Input

Sessions on models that declare `image`/`audio` in their `modalities` (or whose fetched catalog lists them in `architecture.input_modalities`) accept media:

- **Attach**: paste from the clipboard, drag & drop into the panel, or @-mention an image file. Zed sends it as an ACP content block; the transcript shows the original image while the LLM message carries the payload (`image_url` data URI / `input_audio`).
- **Self-directed reading**: vision/audio-capable sessions also get a `read_media` tool. When the user references a screenshot or recording by path, the model loads and perceives it itself instead of asking for a description; payloads ride in a synthetic user message inserted before the assistant tool-call (so the request never ends with the media message).
- On text-only models attached media degrades to a placeholder note in the prompt, so the turn still runs. Media above `ZEN_AGENT_MAX_MEDIA_BYTES` (default 10 MB decoded) is omitted with a note.

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

| Variable                               | Default                          | Description                                                                                                                      |
| -------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `ZEN_AGENT_PROVIDERS`                  | —                                | JSON array of provider definitions (endpoint + API key env + `models` and/or `fetchModels: true`)                                |
| `ZEN_AGENT_PROVIDERS_FILE`             | —                                | Path to a JSON file with the same provider array (alternative to `ZEN_AGENT_PROVIDERS`)                                          |
| `ZEN_AGENT_DEFAULT_PROVIDER`           | first provider                   | Default provider for new sessions                                                                                                |
| `ZEN_AGENT_DB_FILE`                    | next to the package              | SQLite database file holding sessions, transcripts, logs and terminal records (relative paths resolve against the agent's cwd)   |
| `XDG_DATA_HOME`                        | `~/.local/share`                 | Global data dir: the model catalog cache under `zen-agent/`                                                                      |
| `ZEN_AGENT_MAX_TURN_STEPS`             | `25`                             | Max LLM/tool rounds per user prompt                                                                                              |
| `ZEN_AGENT_TERMINAL_OUTPUT_BYTE_LIMIT` | `50000`                          | Max bytes of bash output sent to the model per tool call (tail kept)                                                             |
| `ZEN_AGENT_CHAT_TIMEOUT_MS`            | `6600000` (6.6 min)              | Hard timeout for a single LLM chat request                                                                                       |
| `ZEN_AGENT_CHAT_RPM`                   | `0` (disabled)                   | Client-side chat request rate limit in requests/minute                                                                           |
| `ZEN_AGENT_MAX_MEDIA_BYTES`            | `10000000` (10 MB)               | Max decoded bytes of an attached image/audio block (larger is omitted with a note)                                               |
| `ZEN_AGENT_MAX_RESOURCE_BYTES`         | `262144` (256 KB)                | Max bytes of a resource-link file sent to the model (larger is truncated)                                                        |
| `ZEN_AGENT_SHOW_STATS`                 | `1`                              | Set `0` to hide the per-turn stats line                                                                                          |
| `ZEN_AGENT_SHOW_SKILLS_CATALOG`        | off                              | Inject the skills catalog into the environment message                                                                           |
| `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` | `0` (wait forever)               | Hard-abort escape hatch for pending cancels                                                                                      |
| `ZEN_AGENT_SANDBOX`                    | —                                | `1` = always sandbox bash tool calls                                                                                             |
| `ZEN_AGENT_SANDBOX_CMD`                | default policy                   | Override the bwrap command for bash tool calls                                                                                   |
| `ZEN_AGENT_SANDBOX_BLOCK_SHIM`         | `bin/zen-agent-sandbox-block.sh` | Override the shim shadowing `rm`/`grep`/`find`                                                                                   |
| `<provider apiKeyEnv>`                 | —                                | Any env var referenced by a provider's `apiKeyEnv`, e.g. `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `GROQ_API_KEY` |

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
