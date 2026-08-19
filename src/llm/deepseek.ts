import { createOpenAI } from "@ai-sdk/openai";
import { isStepCount, streamText, tool } from "ai";
import { z } from "zod";
import type { ModelMessage } from "ai";

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

export const SYSTEM_PROMPT = `You are Zen Agent, an autonomous coding agent running on WSL2.
You have exactly one tool: bash.
You can use bash to inspect files, edit files, run tests, install packages, or perform any other shell operation.
There is no approval gate: every command you run is executed immediately.
Prefer small, targeted bash commands. When modifying files, use shell tools such as cat, sed, awk, or tee.
Always work from the session working directory unless you need to reference absolute paths.
When you have completed the user's request, respond with a concise summary of what you did.`;

export const bashTool = tool({
  description:
    "Execute a bash command in the session working directory. The command is completely unrestricted.",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute."),
  }),
});

export function createDeepseekModel() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is required");
  }

  const baseURL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com/v1";
  const modelName = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";

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
}): Promise<LlmStepResult> {
  const model = createDeepseekModel();

  let text = "";
  let finishReason = "unknown";
  const toolCalls: LlmToolCall[] = [];

  const result = streamText({
    model,
    system: SYSTEM_PROMPT,
    messages: options.messages,
    tools: {
      bash: bashTool,
    },
    stopWhen: isStepCount(1),
    abortSignal: options.signal,
  });

  for await (const part of result.fullStream) {
    switch (part.type) {
      case "text-delta": {
        text += part.text;
        await options.onTextDelta?.(part.text);
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
      default:
        break;
    }
  }

  return { text, toolCalls, finishReason };
}
