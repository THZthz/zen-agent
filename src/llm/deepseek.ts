import { createOpenAI } from "@ai-sdk/openai";
import { isStepCount, streamText, tool } from "ai";
import { z } from "zod";
import type { ModelMessage } from "ai";
import type { ModelId, ThinkingEffort } from "../storage.js";

export interface LlmToolCall {
  id: string;
  name: string;
  input: unknown;
}

export interface LlmStepResult {
  text: string;
  toolCalls: LlmToolCall[];
  finishReason: string;
}

export const SYSTEM_PROMPT = `You are an experiened software engineer.

You have exactly one tool: bash.
You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.
There is no approval gate: every command you run is executed immediately.
Prefer small, targeted bash commands. Avoid large output from using bash tool.
When modifying files, use shell tools such as cat, sed, awk, or tee. Prefer using rg, fdfind (fd), jq, uv if they exist.`;

export const bashTool = tool({
  description: "Execute a bash command in current OS. The command is completely unrestricted. Your command will be wrapped inside `script -q -e -c \"bash -lc <original command>\" \"<log path>\"`. If output is large, this tool will tell you to check the log file instead of showing all.",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute."),
  }),
});

export function createDeepseekModel(model?: ModelId | string) {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is required");
  }

  const baseURL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com/v1";
  const modelName = model ?? process.env.DEEPSEEK_MODEL ?? "deepseek-v4-flash";

  const provider = createOpenAI({
    name: "deepseek",
    apiKey,
    baseURL,
  });

  return provider.chat(modelName);
}

export async function runLlmStep(options: {
  messages: ModelMessage[];
  signal?: AbortSignal;
  onTextDelta?: (delta: string) => void | Promise<void>;
  onReasoningDelta?: (delta: string) => void | Promise<void>;
  model?: ModelId;
  thinkingEffort?: ThinkingEffort;
  system?: string;
}): Promise<LlmStepResult> {
  const model = createDeepseekModel(options.model);

  let text = "";
  let finishReason = "unknown";
  const toolCalls: LlmToolCall[] = [];

  const providerOptions =
    options.thinkingEffort && options.thinkingEffort !== "off"
      ? {
          openai: {
            reasoningEffort: options.thinkingEffort,
          },
        }
      : undefined;

  const result = streamText({
    model,
    system: options.system ?? SYSTEM_PROMPT,
    messages: options.messages,
    tools: {
      bash: bashTool,
    },
    stopWhen: isStepCount(1),
    abortSignal: options.signal,
    providerOptions,
    includeRawChunks: true,
  });

  for await (const part of result.fullStream) {
    switch (part.type) {
      case "text-delta": {
        text += part.text;
        await options.onTextDelta?.(part.text);
        break;
      }
      case "reasoning-delta": {
        await options.onReasoningDelta?.(part.text);
        break;
      }
      case "tool-call": {
        toolCalls.push({
          id: part.toolCallId,
          name: part.toolName,
          input: part.input,
        });
        break;
      }
      case "finish-step": {
        finishReason = part.finishReason;
        break;
      }
      case "raw": {
        const raw = part.rawValue as {
          choices?: Array<{
            delta?: {
              reasoning_content?: unknown;
            };
          }>;
        };
        const reasoningContent = raw.choices?.[0]?.delta?.reasoning_content;
        if (typeof reasoningContent === "string" && reasoningContent.length > 0) {
          await options.onReasoningDelta?.(reasoningContent);
        }
        break;
      }
      default:
        break;
    }
  }

  return { text, toolCalls, finishReason };
}
