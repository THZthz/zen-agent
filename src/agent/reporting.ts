import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './core.js';
import { newMessageId } from './config.js';
import {
  costFromUsage,
  fetchBalanceSnapshot,
  getContextWindowTokens,
  getModelPricing,
  getProviderCurrency,
  type LlmUsage,
} from '../providers/index.js';
import type { StoredSession } from '../session/storage.js';
import {
  appendCacheDiagnostic,
  buildCacheDiagnostic,
  latestCacheDiagnostic,
  prefixDiagnosticHashes,
} from '../providers/cache-diagnostics.js';
import { isEnvironmentMessage } from '../session/system-prompt.js';
import {
  cacheHitPercent,
  formatCost,
  formatMs,
  formatTokens,
  roundCost,
  type TurnStats,
} from './stats.js';

/**
 * Turn accounting & reporting (see the ownership map in agent.ts): usage
 * accumulation, cache diagnostics, cost/usage reporting to the client and
 * stats formatting/logging. The turn loop in agent-turn.ts owns the exactly-
 * once finalize guard and delegates all emission here; nothing in this module
 * knows about the loop itself (reporting depends on the turn, never the
 * reverse).
 *
 * Helpers that need no agent state are free functions; the ones that emit to
 * the client or write runtime logs live on the withTurnReporting mixin so
 * they can reach the protected core plumbing (emit/logRuntime/tool schemas).
 */

/** Accumulate one LLM step's usage into the running per-turn stats. */
export async function accumulateTurnUsage(
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

/** Merge a completed turn's stats into the cumulative session usage. */
export function mergeTurnStats(active: ActiveSession, turn: TurnStats): void {
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

/** Billing currency of the session's provider (CNY for DeepSeek, USD for OpenRouter). */
export function costCurrency(active: ActiveSession): string {
  return getProviderCurrency(active.session.config.provider);
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
export function buildExperimentalUsage(session: StoredSession): acp.Usage | null {
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

/** Reporting surface the turn loop (agent-turn.ts) relies on. */
export interface TurnReportingSurface {
  recordCacheDiagnostic(active: ActiveSession, usage: LlmUsage, system: string): Promise<void>;
  reportUsage(
    active: ActiveSession,
    cx: acp.AgentContext,
    contextUsed: number | undefined,
  ): Promise<void>;
  emitTurnStats(active: ActiveSession, cx: acp.AgentContext, turn: TurnStats): Promise<void>;
  logStepStats(
    active: ActiveSession,
    step: number,
    usage: LlmUsage,
    finishReason: string,
    toolCallCount: number,
  ): Promise<void>;
  logTurnStats(active: ActiveSession, turn: TurnStats, stopReason: acp.StopReason): void;
}

export function withTurnReporting<T extends Constructor<ZenAgentCore>>(
  Base: T,
): T & Constructor<TurnReportingSurface> {
  class TurnReporting extends Base {
    constructor(...args: any[]) {
      super(...args);
    }

    /**
     * Per-step cache diagnostics: hash the stable prefix (system prompt, tool
     * schemas, frozen environment snapshot) that the provider's context cache
     * keys on, infer the miss reason locally from hash deltas, and warn when
     * actionable churn is detected. Entries persist on the session (ring-
     * buffered at 50) so the history survives restarts.
     */
    async recordCacheDiagnostic(
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
    async reportUsage(
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
          currency: costCurrency(active),
        },
      });
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
    async emitTurnStats(
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
     * Per-LLM-step stats for the per-startup debug log (log.jsonl): token
     * usage, cache hit ratio, cost and timing for one model request inside a
     * turn. Skipped when the provider reported no usage (the numbers would
     * all be zero/meaningless).
     */
    async logStepStats(
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
    logTurnStats(active: ActiveSession, turn: TurnStats, stopReason: acp.StopReason): void {
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
  return TurnReporting;
}
