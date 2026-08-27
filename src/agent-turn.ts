import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './agent-core.js';
import { MAX_TURN_STEPS, newMessageId } from './agent-config.js';
import {
  buildSystemPrompt,
  ENVIRONMENT_MESSAGE_NAME,
  isEnvironmentMessage,
} from './system-prompt.js';
import {
  costFromUsage,
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getModelPricing,
  runLlmStep,
  type LlmUsage,
} from './provider.js';
import type { StoredSession } from './storage.js';
import {
  appendCacheDiagnostic,
  buildCacheDiagnostic,
  latestCacheDiagnostic,
  prefixDiagnosticHashes,
} from './cache-diagnostics.js';
import { StreamThrottle } from './stream-throttle.js';
import {
  cacheHitPercent,
  emptyTurnStats,
  formatCost,
  formatMs,
  formatTokens,
  roundCost,
  showTurnStats,
  type TurnStats,
} from './turn-stats.js';
import type { ToolExecutionResult } from './tool-execution.js';

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
 * executions, plus the usage/cost/stats bookkeeping that follows each step
 * and each completed turn. The ACP prompt entry point lives in
 * agent-prompt.ts; this mixin only knows how to *run* a turn once its user
 * message has been appended to the session history.
 */
export function withTurnExecution<T extends Constructor<ZenAgentCore>>(
  Base: T,
): T & Constructor<TurnSurface> {
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
      const tools = await this.sessionToolSchemas(active);

      const finalize = async (stopReason: acp.StopReason): Promise<acp.PromptResponse> => {
        this.mergeTurnStats(active, turn);
        this.logTurnStats(active, turn, stopReason);
        await this.reportUsage(active, cx, contextUsed);
        if (showTurnStats()) {
          await this.emitTurnStats(active, cx, turn);
        }
        return {
          stopReason,
          usage: this.buildExperimentalUsage(active.session),
        };
      };

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

        const llmResult = await runLlmStep(active.session.config.provider, {
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

        void this.logLlmExchange(active.session.cwd, active.session.sessionId, {
          type: 'llm_response',
          timestamp: new Date().toISOString(),
          text: llmResult.text,
          toolCalls: llmResult.toolCalls,
          finishReason: llmResult.finishReason,
          usage: llmResult.usage,
        });

        if (llmResult.usage) {
          await this.accumulateTurnUsage(active, turn, llmResult.usage);
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
          const content =
            llmResult.text.length > 0 ? [{ type: 'text' as const, text: llmResult.text }] : [];
          active.session.llmMessages.push({
            role: 'assistant',
            content,
          });
          await this.save(active);
          return finalize(
            active.gracefulCancel ? 'cancelled' : this.stopReasonFromFinish(llmResult.finishReason),
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

        const assistantParts: Array<
          | {
              type: 'text';
              text: string;
            }
          | {
              type: 'reasoning';
              text: string;
            }
          | {
              type: 'tool-call';
              toolCallId: string;
              toolName: string;
              input: unknown;
            }
        > = [];
        if (llmResult.reasoning.length > 0 || active.session.config.thinkingEffort !== 'off') {
          assistantParts.push({ type: 'reasoning', text: llmResult.reasoning });
        }
        if (llmResult.text.length > 0) {
          assistantParts.push({ type: 'text', text: llmResult.text });
        }
        const toolResults: ToolExecutionResult[] = [];

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
            throw new Error('aborted');
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

        // The completed tool call's results were persisted above, so the next
        // turn (the user's follow-up) has full context of what ran.
        if (active.gracefulCancel) {
          return finalize('cancelled');
        }
      }

      return finalize('max_turn_requests');
    }

    private async accumulateTurnUsage(
      active: ActiveSession,
      turn: TurnStats,
      usage: LlmUsage,
    ): Promise<void> {
      turn.steps += 1;
      turn.inputTokens += usage.inputTokens;
      turn.outputTokens += usage.outputTokens;
      turn.cacheReadTokens += usage.cacheReadTokens;
      turn.cacheMissTokens += usage.cacheMissTokens;
      turn.reasoningTokens += usage.reasoningTokens;
      turn.llmMs += usage.llmMs;
      turn.thinkingMs += usage.thinkingMs;
      turn.answeringMs += usage.answeringMs;
      turn.cost += costFromUsage(
        usage,
        await getModelPricing(active.session.config.provider, active.session.config.model),
      );
    }

    /**
     * Per-step cache diagnostics: hash the stable prefix (system prompt, tool
     * schemas, frozen environment snapshot) that the provider's context cache
     * keys on, infer the miss reason locally from hash deltas, and warn when
     * actionable churn is detected. Entries persist on the session (ring-
     * buffered at 50) so the history survives restarts.
     */
    private async recordCacheDiagnostic(
      active: ActiveSession,
      usage: LlmUsage,
      system: string,
    ): Promise<void> {
      const pricing = await getModelPricing(
        active.session.config.provider,
        active.session.config.model,
      );
      const prefix = prefixDiagnosticHashes({
        system,
        toolSpecs: await this.sessionToolSchemas(active),
        env: active.session.llmMessages.find(isEnvironmentMessage) ?? null,
      });
      const entry = buildCacheDiagnostic({
        turn: active.session.usage.turns + 1,
        model: active.session.config.model,
        usage: {
          inputTokens: usage.inputTokens,
          cachedTokens: usage.cacheReadTokens,
          missTokens: usage.cacheMissTokens,
        },
        pricing,
        prefix,
        previous: latestCacheDiagnostic(active.session.cacheDiagnostics),
      });
      active.session.cacheDiagnostics = appendCacheDiagnostic(
        active.session.cacheDiagnostics,
        entry,
      );
      if (
        entry.missReason === 'system-prompt-changed' ||
        entry.missReason === 'env-snapshot-changed' ||
        entry.missReason === 'tool-list-changed' ||
        entry.missReason === 'tool-schema-or-order-changed'
      ) {
        await this.logRuntime(active.session.cwd, 'warn', 'cache miss', {
          sessionId: active.session.sessionId,
          turn: entry.turn,
          model: entry.model,
          hitRate: entry.cacheHitRate,
          reason: entry.missReason,
          detail: entry.missReasonDetail,
        });
      }
    }

    private mergeTurnStats(active: ActiveSession, turn: TurnStats): void {
      const usage = active.session.usage;
      usage.turns += 1;
      usage.steps += turn.steps;
      usage.inputTokens += turn.inputTokens;
      usage.outputTokens += turn.outputTokens;
      usage.cacheReadTokens += turn.cacheReadTokens;
      usage.cacheMissTokens += turn.cacheMissTokens;
      usage.reasoningTokens += turn.reasoningTokens;
      usage.cost += turn.cost;
      usage.llmMs += turn.llmMs;
      usage.thinkingMs += turn.thinkingMs;
      usage.answeringMs += turn.answeringMs;
      usage.toolMs += turn.toolMs;
      // Keep per-turn stats so they survive process restarts (resume/load).
      active.session.turnStats.push(turn);
    }

    /**
     * Report context window usage and cumulative cost to the client.
     *
     * Zed maps this ACP update in `acp_thread.rs` (SessionUpdate::UsageUpdate)
     * into its TokenUsage/SessionCost and renders it in the agent panel header
     * (`agent_ui/.../thread_view.rs::render_token_usage`) as a token-usage ring
     * whose tooltip shows "Context: used / max" and "Cost: amount <currency>"
     * (CNY for DeepSeek, USD for OpenRouter). The ring also warns at 80% and
     * marks the thread exceeded at 100% of `size`.
     */
    private async reportUsage(
      active: ActiveSession,
      cx: acp.AgentContext,
      contextUsed: number | undefined,
    ): Promise<void> {
      if (contextUsed === undefined) {
        return;
      }
      const size = await getContextWindowTokens(
        active.session.config.provider,
        active.session.config.model,
      );
      await this.emit(active, cx, {
        sessionUpdate: 'usage_update',
        used: Math.min(contextUsed, size),
        size,
        cost: {
          amount: roundCost(active.session.usage.cost),
          currency: this.costCurrency(active),
        },
      });
    }

    /** Billing currency of the session's provider (CNY for DeepSeek, USD for OpenRouter). */
    private costCurrency(active: ActiveSession): 'CNY' | 'USD' {
      return active.session.config.provider === 'openrouter' ? 'USD' : 'CNY';
    }

    /**
     * Emit a compact per-turn stats line as its own message in the thread.
     *
     * There is no ACP field for turns/steps/timing/cache-ratio, so this is
     * displayed as a regular agent message. It is deliberately NOT pushed to
     * `llmMessages` (it would waste context tokens on the next LLM request),
     * and it uses a fresh messageId so Zed renders it as its own bubble. Note
     * Zed auto-sends queued follow-up messages on the turn's Stopped event
     * (`agent_ui/.../conversation_view.rs`), so this stats bubble is always
     * followed by the user's next message rather than being dropped.
     */
    private async emitTurnStats(
      active: ActiveSession,
      cx: acp.AgentContext,
      turn: TurnStats,
    ): Promise<void> {
      if (turn.steps === 0) {
        return;
      }
      const symbol = active.session.config.provider === 'openrouter' ? '$' : '¥';
      const text = [
        `Turn ${active.session.usage.turns} · ${turn.steps} step${turn.steps === 1 ? '' : 's'} · think ${formatMs(turn.thinkingMs)} · answer ${formatMs(turn.answeringMs)} · tools ${formatMs(turn.toolMs)}`,
        `in ${formatTokens(turn.inputTokens)} · out ${formatTokens(turn.outputTokens)} · cache hit ${cacheHitPercent(turn)} · ${symbol}${formatCost(turn.cost)} (session ${symbol}${formatCost(active.session.usage.cost)})`,
      ].join('\n');
      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        messageId: newMessageId(),
        content: { type: 'text', text },
      });
      // Cross-check the locally estimated cost against the provider's actual
      // billing; fire-and-forget so a slow/failed balance request never
      // delays or breaks the stats bubble.
      void this.verifyTurnCost(active, turn);
    }

    /**
     * Verify turn stats against the provider's billing and log the result to
     * log.jsonl ("turn stats balance verify").
     *
     * DeepSeek's /user/balance (and OpenRouter's /auth/key) returns the current
     * account balance; the delta between this turn's snapshot and the previous
     * turn's snapshot should equal the turn's locally estimated cost
     * (turn.cost, derived from the usage fields the provider streams back
     * per step). A persistent mismatch means the pricing table or token
     * counting is wrong. Balance values are only two-decimal-precise, so
     * single-turn deltas are noisy — this is data gathering only, no behavior
     * change.
     */
    private async verifyTurnCost(active: ActiveSession, turn: TurnStats): Promise<void> {
      try {
        const provider = active.session.config.provider;
        const snapshot = await fetchBalanceSnapshot(provider);
        const details: Record<string, unknown> = {
          sessionId: active.session.sessionId,
          turn: active.session.usage.turns,
          provider,
          model: active.session.config.model,
          estimatedTurnCost: roundCost(turn.cost),
          sessionEstimatedCost: roundCost(active.session.usage.cost),
          balanceIsAvailable: snapshot.isAvailable,
          balanceCurrency: snapshot.currency,
          balanceTotal: snapshot.total,
          ...snapshot.details,
        };
        const before = this.lastBalanceByProvider.get(provider) ?? null;
        if (before !== null && before.currency === snapshot.currency) {
          const balanceDelta = before.total - snapshot.total;
          details.balanceBefore = before.total;
          details.balanceDelta = roundCost(balanceDelta);
          details.deltaVsEstimated = roundCost(balanceDelta - turn.cost);
        }
        this.lastBalanceByProvider.set(provider, {
          currency: snapshot.currency,
          total: snapshot.total,
        });
        await this.logRuntime(active.session.cwd, 'info', 'turn stats balance verify', details);
      } catch (error) {
        await this.logRuntime(active.session.cwd, 'warn', 'turn stats balance verify failed', {
          sessionId: active.session.sessionId,
          turn: active.session.usage.turns,
          provider: active.session.config.provider,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    /**
     * Experimental ACP PromptResponse.usage.
     *
     * The field is marked UNSTABLE in the ACP spec. Zed only consumes it behind
     * its `AcpBetaFeatureFlag` (`acp_thread.rs`, "response.usage"), where it
     * updates the thread's cumulative input/output token counters. We still
     * send it unconditionally — a stable fallback for other clients and for
     * when Zed's beta flag is enabled.
     */
    private buildExperimentalUsage(session: StoredSession): acp.Usage | null {
      const usage = session.usage;
      if (usage.turns === 0 && usage.inputTokens === 0 && usage.outputTokens === 0) {
        return null;
      }
      return {
        totalTokens: usage.inputTokens + usage.outputTokens,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        thoughtTokens: usage.reasoningTokens,
        cachedReadTokens: usage.cacheReadTokens,
        cachedWriteTokens: null,
      };
    }

    /**
     * Per-LLM-step stats for the per-startup debug log (log.jsonl): token
     * usage, cache hit ratio, cost and timing for one model request inside a
     * turn. Skipped when the provider reported no usage (the numbers would
     * all be zero/meaningless).
     */
    private async logStepStats(
      active: ActiveSession,
      step: number,
      usage: LlmUsage,
      finishReason: string,
      toolCallCount: number,
    ): Promise<void> {
      const pricing = await getModelPricing(
        active.session.config.provider,
        active.session.config.model,
      );
      void this.logRuntime(active.session.cwd, 'info', 'llm step stats', {
        sessionId: active.session.sessionId,
        step: step + 1,
        provider: active.session.config.provider,
        model: active.session.config.model,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        cacheReadTokens: usage.cacheReadTokens,
        cacheMissTokens: usage.cacheMissTokens,
        cacheHitPercent: cacheHitPercent(usage),
        reasoningTokens: usage.reasoningTokens,
        cost: costFromUsage(usage, pricing),
        llmMs: usage.llmMs,
        thinkingMs: usage.thinkingMs,
        answeringMs: usage.answeringMs,
        finishReason,
        toolCalls: toolCallCount,
      });
    }

    /**
     * Per-turn stats for the per-startup debug log (log.jsonl): aggregate of
     * all LLM steps in the turn plus tool execution time and the stop reason.
     * Called from finalize() after mergeTurnStats() so `usage.turns` already
     * points at this turn.
     */
    private logTurnStats(active: ActiveSession, turn: TurnStats, stopReason: acp.StopReason): void {
      void this.logRuntime(active.session.cwd, 'info', 'turn stats', {
        sessionId: active.session.sessionId,
        turn: active.session.usage.turns,
        provider: active.session.config.provider,
        model: active.session.config.model,
        stopReason,
        steps: turn.steps,
        inputTokens: turn.inputTokens,
        outputTokens: turn.outputTokens,
        cacheReadTokens: turn.cacheReadTokens,
        cacheMissTokens: turn.cacheMissTokens,
        cacheHitPercent: cacheHitPercent(turn),
        reasoningTokens: turn.reasoningTokens,
        cost: turn.cost,
        llmMs: turn.llmMs,
        thinkingMs: turn.thinkingMs,
        answeringMs: turn.answeringMs,
        toolMs: turn.toolMs,
      });
    }
  }
  return TurnExecution;
}
