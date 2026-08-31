import { ZenAgentCore } from './agent-core.js';
import { withSessionManagement } from './agent-session.js';
import { withTurnExecution } from './agent-turn.js';
import { withPromptExecution } from './agent-prompt.js';
import { withTurnReporting } from './turn-reporting.js';

/**
 * Zen Agent, composed from mixins with explicit ownership boundaries
 * (see the ownership map below):
 *
 * | Concern                          | Owner                              |
 * | -------------------------------- | ---------------------------------- |
 * | Serialization / cancellation     | withSessionOperation queue + AbortController in core/prompt/session mixins |
 * | Turn accounting & reporting      | turn-reporting.ts (withTurnReporting) |
 * | Persistence boundaries (save())  | turn/prompt owners (agent-turn.ts, agent-prompt.ts) |
 * | Terminal lifecycle & kill-on-abort | tool-execution.ts (bash tool path) |
 * | Stream cleanup                   | StreamThrottle.discard() at the step owner (agent-turn.ts) |
 * | Wire conversion                  | chat-completions.ts pure layer |
 *
 * Layering: session lifecycle (agent-session.ts), turn reporting
 * (turn-reporting.ts), the LLM turn loop (agent-turn.ts), and the
 * prompt/slash-command entry point (agent-prompt.ts), all over shared state
 * and plumbing in agent-core.ts.
 */
export class ZenAgent extends withPromptExecution(
  withTurnExecution(withTurnReporting(withSessionManagement(ZenAgentCore))),
) {}
