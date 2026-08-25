import type { LlmMessage } from './storage.js';

/**
 * Heal message history before it is sent to the LLM API, ported from
 * Reasonix's loop/healing.ts. DeepSeek 400s on unpaired assistant tool_calls
 * and on stray tool messages; a hard abort or crash can leave either shape in
 * a persisted session. Healthy histories pass through unchanged.
 */

export interface HealResult {
  messages: LlmMessage[];
  /** Assistant messages dropped for unpaired tool calls. */
  droppedAssistants: number;
  /** Tool messages dropped (stray, or unpaired leftovers of a dropped assistant). */
  droppedTools: number;
}

let stampSeq = 0;

/**
 * Drops unpaired assistant tool-call messages and stray tool messages, and
 * stamps missing tool-call ids (corrupted sessions can carry calls without an
 * id — DeepSeek 400s on those). Never mutates the input messages.
 */
export function healMessages(messages: readonly LlmMessage[]): HealResult {
  const out: LlmMessage[] = [];
  let droppedAssistants = 0;
  let droppedTools = 0;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]!;

    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      const hasCalls = msg.content.some((part) => part.type === 'tool-call');
      if (!hasCalls) {
        out.push(msg);
        continue;
      }

      const stamped = msg.content.map((part) =>
        part.type === 'tool-call' && !part.toolCallId
          ? { ...part, toolCallId: `zen-${Date.now()}-${stampSeq++}` }
          : part,
      );
      const needed = new Set(
        stamped.filter((part) => part.type === 'tool-call').map((part) => part.toolCallId),
      );

      // The following consecutive tool messages must provide a result for
      // every call. A tool message is consumed only when ALL of its results
      // are still needed — a stray result inside it makes the whole message
      // unpaired (DeepSeek 400s on orphan tool results).
      const candidates: LlmMessage[] = [];
      let j = i + 1;
      while (j < messages.length && needed.size > 0) {
        const next = messages[j]!;
        if (next.role !== 'tool' || !Array.isArray(next.content)) break;
        const resultIds = next.content
          .filter((part) => part.type === 'tool-result')
          .map((part) => part.toolCallId);
        if (resultIds.length === 0) break;
        if (!resultIds.every((id) => needed.has(id))) break;
        for (const id of resultIds) needed.delete(id);
        candidates.push(next);
        j++;
      }

      if (needed.size === 0) {
        out.push({ ...msg, content: stamped });
        for (const candidate of candidates) out.push(candidate);
        i = j - 1;
      } else {
        droppedAssistants += 1;
        // The consumed candidates are dropped with the assistant; any tool
        // messages after them fall through to the stray branch below.
        droppedTools += candidates.length;
        i = j - 1;
      }
      continue;
    }

    if (msg.role === 'tool') {
      droppedTools += 1;
      continue;
    }

    out.push(msg);
  }

  return { messages: out, droppedAssistants, droppedTools };
}
