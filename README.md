# Zen Agent

An [Agent Client Protocol](https://agentclientprotocol.com) v1 coding agent for use with [Zed](https://zed.dev) on WSL2.

- Single tool: `bash`
- No permission prompts (approval policy: `never`)
- Sessions are stored in `<project>/sessions/`
- LLM provider: Deepseek via the Vercel AI SDK

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
  "agent": {
    "command": "node",
    "args": ["/path/to/zen-agent/dist/index.js"]
  }
}
```

The agent reads newline-delimited JSON-RPC from stdin and writes responses to stdout.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | — | Deepseek API key (required) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Deepseek model name |

## Development

```bash
npm run typecheck
npm test
npm run build
```
