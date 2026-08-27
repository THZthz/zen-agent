import { ZenAgentCore } from './agent-core.js';
import { withSessionManagement } from './agent-session.js';
import { withTurnExecution } from './agent-turn.js';
import { withPromptExecution } from './agent-prompt.js';

/**
 * Zen Agent: ACP session lifecycle (agent-session.ts), the LLM turn loop and
 * usage bookkeeping (agent-turn.ts), and the prompt/slash-command entry point
 * (agent-prompt.ts), layered over shared state and plumbing in agent-core.ts.
 */
export class ZenAgent extends withPromptExecution(
  withTurnExecution(withSessionManagement(ZenAgentCore)),
) {}
