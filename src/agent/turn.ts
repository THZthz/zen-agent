import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './core.js';
import { MAX_TURN_STEPS, newMessageId } from './config.js';
import { buildSystemPrompt, ENVIRONMENT_MESSAGE_NAME } from '../session/system-prompt.js';
import { runLlmStep } from '../providers/index.js';
import type { AssistantMessage } from '../session/storage.js';
import { StreamThrottle } from './stream-throttle.js';
import { emptyTurnStats, showTurnStats } from './stats.js';
import {
  accumulateTurnUsage,
  buildExperimentalUsage,
  mergeTurnStats,
  type TurnReportingSurface,
} from './reporting.js';
import type { ToolExecutionResult } from '../tools/execution.js';

/** Turn-running surface the prompt/slash-command mixin relies on. */
export interface TurnSurface {
  runTurnTracked(
    active: ActiveSession,
    cx: acp.AgentContext,
    signal: AbortSignal,
  ): Promise<acp.PromptResponse>;
}

/**
 * The LLM turn loop: one prompt's worth of repeated model steps + bash tool
 * executions. Usage/cost/stats bookkeeping and all formatting lives in
 * turn-reporting.ts; this mixin owns the loop, the exactly-once finalize
 * guard and the hard-abort/tool-pairing logic.
 *
 * The ACP prompt entry point lives in agent-prompt.ts; this mixin only knows
 * how to *run* a turn once its user message has been appended to the session
 * history.
 */
export function withTurnExecution<
  T extends Constructor<ZenAgentCore> & Constructor<TurnReportingSurface>,
>(Base: T): T & Constructor<TurnSurface> {
  class TurnExecution extends Base {
    constructor(...args: any[]) {
      super(...args);
    }
    /**
     * Runs one turn and records its promise on the session so concurrent entry
     * points (new prompt, load/resume, close/delete) can wait out the old unit
     * of work instead of racing it on the shared history.
     */
    async runTurnTracked(
      active: ActiveSession,
      cx: acp.AgentContext,
      signal: AbortSignal,
    ): Promise<acp.PromptResponse> {
      const turn = this.runTurn(active, cx, signal).finally(() => {
        if (active.turnPromise === turn) {
          active.turnPromise = null;
        }
      });
      active.turnPromise = turn;
      return turn;
    }

    private async runTurn(
      active: ActiveSession,
      cx: acp.AgentContext,
      signal: AbortSignal,
    ): Promise<acp.PromptResponse> {
      const turn = emptyTurnStats();
      let contextUsed: number | undefined;
      let finalized = false;
      const tools = await this.sessionToolSchemas(active);

      const finalize = async (stopReason: acp.StopReason): Promise<acp.PromptResponse> => {
        if (!finalized) {
          finalized = true;
          mergeTurnStats(active, turn);
          this.logTurnStats(active, turn, stopReason);
          await this.reportUsage(active, cx, contextUsed);
          if (showTurnStats()) {
            await this.emitTurnStats(active, cx, turn);
          }
        }
        return {
          stopReason,
          usage: buildExperimentalUsage(active.session),
        };
      };

      try {
        for (let step = 0; step < MAX_TURN_STEPS; step++) {
          // Graceful-cancel boundary: stop before starting a new LLM step (i.e.
          // when the previous tool call step already completed, or when the
          // cancel arrived between steps).
          if (active.gracefulCancel) {
            return finalize('cancelled');
          }

          const assistantMessageId = newMessageId();
          // Captured once per step: the system prompt is part of the byte-stable
          // prefix the provider's context cache keys on, so it must be identical
          // for the request, the transcript, and the cache diagnostics.
          const system = buildSystemPrompt(active.session);

          void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
            type: 'llm_request',
            timestamp: new Date().toISOString(),
            model: active.session.config.model,
            thinkingEffort: active.session.config.thinkingEffort,
            system,
            messages: active.session.llmMessages,
          });

          const stream = new StreamThrottle(async (kind, delta) => {
            if (kind === 'thought') {
              await this.emit(active, cx, {
                sessionUpdate: 'agent_thought_chunk',
                messageId: assistantMessageId,
                content: { type: 'text', text: delta },
              });
            } else {
              await this.emit(active, cx, {
                sessionUpdate: 'agent_message_chunk',
                messageId: assistantMessageId,
                content: { type: 'text', text: delta },
              });
            }
          });

          let llmResult: Awaited<ReturnType<typeof runLlmStep>>;
          try {
            llmResult = await runLlmStep(active.session.config.provider, {
              messages: active.session.llmMessages,
              tools,
              // OpenRouter uses this as its `session_id`: it derives the Z.AI
              // session affinity key from it so the upstream context cache is
              // pinned to this conversation from the first request (see
              // openrouter.ts). Without it, requests can be re-keyed and the
              // GLM cache drops to a 0% hit rate.
              sessionId: active.session.sessionId,
              signal,
              logRuntime: (level, message, details) => {
                void this.logRuntime(active.session.cwd, level, message, details);
              },
              model: active.session.config.model,
              thinkingEffort: active.session.config.thinkingEffort,
              system,
              onTextDelta: async (delta) => {
                stream.push('message', delta);
              },
              onReasoningDelta: async (delta) => {
                stream.push('thought', delta);
              },
            });
            await stream.drain();
          } catch (error) {
            stream.discard();
            throw error;
          }

          void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
            type: 'llm_response',
            timestamp: new Date().toISOString(),
            text: llmResult.text,
            toolCalls: llmResult.toolCalls,
            finishReason: llmResult.finishReason,
            usage: llmResult.usage,
          });

          if (llmResult.usage) {
            await accumulateTurnUsage(active, turn, llmResult.usage);
            await this.recordCacheDiagnostic(active, llmResult.usage, system);
            contextUsed = llmResult.usage.inputTokens;
            await this.reportUsage(active, cx, contextUsed);
            await this.logStepStats(
              active,
              step,
              llmResult.usage,
              llmResult.finishReason,
              llmResult.toolCalls.length,
            );
          }

          if (llmResult.toolCalls.length === 0) {
            // The model finished its answer. If a cancel arrived while it was
            // streaming, the turn still ends "cancelled" (the client interrupted
            // it), but the completed answer is kept in the conversation history.
            const content: AssistantMessage['content'] = [];
            if (
              llmResult.reasoning.length > 0 ||
              llmResult.reasoningSignature !== undefined ||
              active.session.config.thinkingEffort !== 'off'
            ) {
              content.push({
                type: 'reasoning',
                text: llmResult.reasoning,
                ...(llmResult.reasoningSignature !== undefined
                  ? { reasoningSignature: llmResult.reasoningSignature }
                  : {}),
              });
            }
            if (llmResult.text.length > 0) {
              content.push({ type: 'text', text: llmResult.text });
            }
            active.session.llmMessages.push({ role: 'assistant', content });
            await this.save(active);
            return finalize(
              active.gracefulCancel
                ? 'cancelled'
                : this.stopReasonFromFinish(llmResult.finishReason),
            );
          }

          // A cancel requested during the LLM step means "stop after thinking
          // completes". The step is done, so stop here and DISCARD the tool calls
          // it proposed: the user's follow-up supersedes them. They were never
          // executed, so nothing is persisted (and nothing was emitted to Zed
          // either — tool_call entries are only created in executeLlmToolCall).
          if (active.gracefulCancel) {
            return finalize('cancelled');
          }

          const assistantParts: AssistantMessage['content'] = [];
          if (
            llmResult.reasoning.length > 0 ||
            llmResult.reasoningSignature !== undefined ||
            active.session.config.thinkingEffort !== 'off'
          ) {
            assistantParts.push({
              type: 'reasoning',
              text: llmResult.reasoning,
              ...(llmResult.reasoningSignature !== undefined
                ? { reasoningSignature: llmResult.reasoningSignature }
                : {}),
            });
          }
          if (llmResult.text.length > 0) {
            assistantParts.push({ type: 'text', text: llmResult.text });
          }
          const toolResults: ToolExecutionResult[] = [];
          let hardAborted = false;

          for (const call of llmResult.toolCalls) {
            // The bash tool runs to completion even when a graceful cancel is
            // pending: the abort listener in executeLlmToolCall only fires on a
            // HARD abort (session close/delete or the timeout escape hatch), so
            // the command is never killed mid-run by a user follow-up.
            const toolStart = Date.now();
            const result = await this.executeLlmToolCall(active, cx, call, signal);
            turn.toolMs += Date.now() - toolStart;
            toolResults.push(result);
            assistantParts.push({
              type: 'tool-call',
              toolCallId: call.id,
              toolName: call.name,
              input: call.input,
            });
            if (signal.aborted) {
              hardAborted = true;
              break;
            }
            // Graceful-cancel boundary: stop right after this tool call finishes
            // ("insert after the tool call step"). assistantParts only includes
            // the calls that actually ran, keeping the LLM message history
            // consistent (no unresolved tool calls).
            if (active.gracefulCancel) {
              break;
            }
          }

          // read_media payload injection. The OpenAI-compatible tool role only
          // accepts text, so the actual image/audio parts ride in a synthetic
          // user message (the same trick Claude Code's Read tool uses).
          //
          // ORDER MATTERS for the provider's context cache: GLM/Z.AI (and
          // similarity-based caches in general) drop to a 0% hit rate when a
          // request ENDS with a user message carrying image/audio content -
          // every such step in the live aurelia session showed cacheRead=0,
          // while the identical prefix ending in a tool message hit ~99%. The
          // synthetic media message is therefore inserted BEFORE the assistant
          // tool-call message that requested it (which the tool result below
          // still pairs with), so the following LLM request always ends with a
          // tool result and keeps hitting the cached prefix.
          const mediaResults = toolResults.filter((result) => result.attachedMedia);
          if (mediaResults.length > 0) {
            active.session.llmMessages.push({
              role: 'user',
              // Reuse the environment snapshot's name so the model can tell
              // this message is machine-generated (the read_media payload
              // injected by the agent), not something the user typed.
              name: ENVIRONMENT_MESSAGE_NAME,
              content: mediaResults.flatMap((result) => {
                const media = result.attachedMedia!;
                return [
                  {
                    type: 'text' as const,
                    text: `[read_media] ${media.path} (${media.mimeType}, ${media.decodedBytes} bytes):`,
                  },
                  media.modality === 'image'
                    ? { type: 'image' as const, mimeType: media.mimeType, data: media.data }
                    : { type: 'audio' as const, mimeType: media.mimeType, data: media.data },
                ];
              }),
            });
          }
          active.session.llmMessages.push({
            role: 'assistant',
            content: assistantParts,
          });
          active.session.llmMessages.push({
            role: 'tool',
            content: toolResults.map((result) => ({
              type: 'tool-result',
              toolCallId: result.toolCallId,
              toolName: result.toolName,
              output: result.output,
            })),
          });
          await this.save(active);

          if (hardAborted) {
            throw signal.reason instanceof Error ? signal.reason : new Error('aborted');
          }

          // The completed tool call's results were persisted above, so the next
          // turn (the user's follow-up) has full context of what ran.
          if (active.gracefulCancel) {
            return finalize('cancelled');
          }
        }

        return finalize('max_turn_requests');
      } catch (error) {
        // Preserve already-billed steps and completed tool time even when a
        // later provider/tool/notification operation fails. The prompt's final
        // save persists this exactly-once merge before the error is surfaced.
        if (!finalized && (turn.steps > 0 || turn.toolMs > 0)) {
          try {
            await finalize(signal.aborted ? 'cancelled' : 'end_turn');
          } catch (finalizeError) {
            void this.logRuntime(
              active.session.cwd,
              'error',
              'failed to finalize interrupted turn',
              {
                sessionId: active.session.sessionId,
                error:
                  finalizeError instanceof Error ? finalizeError.message : String(finalizeError),
              },
            );
          }
        }
        throw error;
      }
    }
  }
  return TurnExecution;
}
