# Zen Agent — ACP v1 Coding Agent Specification

## 1. Overview

Zen Agent is a TypeScript coding agent that speaks the **Agent Client Protocol (ACP) v1** over **stdio**. It is designed to run as a subprocess launched by Zed (or any ACP-compatible client) on WSL2.

The agent has exactly one tool: **`bash`**. The tool has **no restrictions**: Zen Agent will execute any command the model requests, in the session working directory, without asking the user for permission (approval policy is **never**).

This document is the implementation specification. It will be updated as decisions are confirmed.

## 2. Goals

- Provide a working ACP v1 agent binary usable from Zed.
- Implement the required ACP baseline methods:
  - `initialize`
  - `authenticate`
  - `session/new`
  - `session/prompt`
  - `session/cancel` (notification)
  - `session/update` (client notification)
- Stream model output to the client as `agent_message_chunk` updates.
- Report `bash` invocations as ACP `tool_call` / `tool_call_update` updates.
- Never request permission before running `bash`.
- Support cancellation of in-flight LLM requests and `bash` processes.
- Keep the implementation small and understandable; use Pi as a reference for shell execution details.

## 3. Non-Goals (initial version)

- No MCP server connections (accepted and ignored for compatibility).
- No session persistence / `session/load`, `session/list`, `session/delete`, `session/resume`, or `session/close`.
- No client filesystem (`fs/*`) or terminal (`terminal/*`) usage.
- No session modes, config options, elicitation, slash commands, or plans.
- No Windows-native support; this targets WSL2/Linux.

## 4. Protocol Surface

### 4.1 Transport

- **stdio** only.
- Each JSON-RPC message is a single line of UTF-8 JSON, newline-delimited.
- Nothing except valid ACP messages is written to stdout.
- Logs go to stderr.

### 4.2 Agent-Implemented Methods

| Method | Description |
| --- | --- |
| `initialize` | Negotiate protocol version and capabilities. |
| `authenticate` | No-op; returns `{}`. |
| `session/new` | Create an in-memory session with a generated ID and the provided `cwd`. |
| `session/prompt` | Run a full agent turn: call the LLM, execute `bash` tool calls, stream updates, return `stopReason`. |
| `session/cancel` | Abort the active prompt for the given session. |

### 4.3 Client-Implemented Methods Used

| Method | When |
| --- | --- |
| `session/update` | Send text chunks, tool call creation, and tool call status/content updates. |
| `session/request_permission` | **Never used.** Approval policy is `never`. |

### 4.4 `initialize` Response Shape

```json
{
  "protocolVersion": 1,
  "agentCapabilities": {
    "loadSession": false
  },
  "agentInfo": {
    "name": "zen-agent",
    "title": "Zen Agent",
    "version": "0.1.0"
  },
  "authMethods": []
}
```

We do not advertise `promptCapabilities.image`, `audio`, or `embeddedContext` in the first version; text and resource links are baseline-supported by ACP.

### 4.5 `session/new` Behavior

- Validate `cwd` is absolute and exists.
- Create a session record:
  - `sessionId`
  - `cwd`
  - conversation history (initially empty)
  - active `AbortController` (initially none)
- Return `{ sessionId }`.
- `mcpServers` and `additionalDirectories` are accepted but ignored.

## 5. Agent Turn Lifecycle (`session/prompt`)

1. Look up the session. If missing, return a JSON-RPC error.
2. Convert the incoming `ContentBlock[]` into LLM user message(s):
   - `text` → plain text.
   - `resource_link` with a `file://` URI → read the local file and inline its text as context; if unreadable, include the URI/name as text.
   - Any unsupported block type → return a clear error.
3. Append the user message to the session history.
4. Create an `AbortController` for this turn.
5. Loop:
   a. Call the LLM with:
      - a system prompt describing the agent and the `bash` tool,
      - full session history,
      - a single `bash` function/tool definition.
   b. Stream text deltas from the LLM as `session/update` with `sessionUpdate: "agent_message_chunk"`.
   c. If the LLM returns one or more `bash` tool calls:
      - For each call:
        1. Send `session/update` with `sessionUpdate: "tool_call"`:
           - `toolCallId`
           - `title: "Run bash command"`
           - `kind: "execute"`
           - `status: "pending"`
           - `rawInput: { command }`
        2. Send `session/update` with `sessionUpdate: "tool_call_update"` and `status: "in_progress"`.
        3. Execute `bash` in the session `cwd` (see §6).
        4. Send `session/update` with `sessionUpdate: "tool_call_update"` and:
           - `status: "completed"` (or `"failed"` on non-zero exit / spawn error),
           - `content` with the command output as text,
           - `rawOutput: { output, exitCode, cancelled, truncated }`.
      - Append the assistant tool-call message and the tool result to session history.
      - Continue the loop.
   d. If the LLM returns no tool calls, finish the turn.
6. Return `{ stopReason: "end_turn" }`.
7. If `session/cancel` is received:
   - Abort the active turn's `AbortController`.
   - Kill any running `bash` child process tree.
   - Send any final cleanup updates.
   - Return `{ stopReason: "cancelled" }` from the pending `session/prompt` (not a JSON-RPC error).
   - Suppress cancellation exceptions from the LLM client and process spawn.

## 6. `bash` Tool

### 6.1 Tool Definition Sent to the LLM

```json
{
  "type": "function",
  "function": {
    "name": "bash",
    "description": "Execute a bash command in the session working directory. The command is completely unrestricted.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "description": "The bash command to execute."
        }
      },
      "required": ["command"]
    }
  }
}
```

### 6.2 Execution

- Use `child_process.spawn` with `bash`, `-lc`, and the command string.
- Set `cwd` to the session working directory.
- Inherit/merge the current process environment.
- Capture stdout and stderr combined.
- Strip ANSI escape sequences and replace binary garbage with safe placeholders (reference: Pi's `sanitizeBinaryOutput`).
- Limit the returned output to a reasonable size (e.g. 30,000 chars) and indicate truncation in `rawOutput`.
- On cancellation, kill the entire descendant process tree (`process.kill(-pid)` on Linux/WSL2).
- No permission flow is used.

## 7. LLM Provider

The agent must call a generative model that supports tool/function calling.

### 7.1 Proposed Provider Interface

```ts
interface LlmProvider {
  complete(request: LlmRequest, signal: AbortSignal): AsyncIterable<LlmEvent>;
}

type LlmRequest = {
  system: string;
  messages: LlmMessage[];
  tools: ToolDefinition[];
};

type LlmEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_call"; id: string; name: string; args: unknown }
  | { type: "usage"; inputTokens: number; outputTokens: number };
```

### 7.2 Environment Configuration (OpenAI-compatible)

Proposed default:

| Variable | Purpose |
| --- | --- |
| `ZEN_AGENT_MODEL` | Model name (required). |
| `ZEN_AGENT_BASE_URL` | OpenAI-compatible base URL (default: `https://api.openai.com/v1`). |
| `ZEN_AGENT_API_KEY` | API key (fallback to `OPENAI_API_KEY`). |

Support for Anthropic can be added as a second adapter if desired.

## 8. Session History and Context

- Store conversation messages in memory only.
- Each session stores the full LLM message history, including tool calls and results.
- No compaction/truncation in v1; if the context window is exceeded, return `stopReason: "max_tokens"` or a clear error.

## 9. Project Layout

```
zen-agent/
  package.json
  tsconfig.json
  SPEC.md
  README.md
  src/
    index.ts          # entry point: stdio stream + agent app
    agent.ts          # ACP handlers and session store
    llm/
      types.ts
      provider.ts     # provider interface
      openai.ts       # OpenAI-compatible chat completions adapter
      anthropic.ts    # optional Anthropic adapter
    tools/
      bash.ts         # bash execution with streaming/cancellation
```

## 10. Dependencies

- `@agentclientprotocol/sdk` — official ACP v1 TypeScript SDK for JSON-RPC/stdio plumbing and types.
- `typescript`, `tsx`, `@types/node` — development/build tooling.
- No heavy framework; HTTP calls use Node's built-in `fetch`.

Using the official SDK is the proposed approach. If you prefer a hand-rolled JSON-RPC layer, that is a small change to this spec.

## 11. Testing

- Unit tests for:
  - Bash execution (output, exit code, truncation, cancellation).
  - Prompt content conversion.
- Integration test with the official SDK's in-memory client to drive `initialize` → `session/new` → `session/prompt` with a fake LLM.
- Manual smoke test from a terminal using `node dist/index.js` and piping JSON-RPC lines.
- Final validation from Zed with a sample project.

## 12. Open Questions

1. **LLM provider**: Which provider/model should be the default? (OpenAI-compatible, Anthropic, Gemini, local model?)
2. **SDK vs raw**: Is using the official `@agentclientprotocol/sdk` acceptable, or should the protocol be implemented directly?
3. **Session persistence**: Should v1 include `session/load` / disk persistence, or is in-memory only sufficient for Zed use?
4. **MCP servers**: Should we ignore MCP servers entirely, or implement stdio MCP connectivity in a later iteration?
