import type { SessionUpdate } from '@agentclientprotocol/sdk';

/*
 * WHY THIS FILE EXISTS (bash tool cards after a Zed restart)
 * ----------------------------------------------------------
 *
 * Symptom: after restarting Zed and resuming a session, bash tool calls
 * showed no output and no expand/unfold arrow on their cards.
 *
 * Root causes (Zed v1.16.x, `crates/acp_thread`, `crates/agent_ui`):
 *
 * 1. `terminal` content items are resolved against LIVE terminal entities
 *    registered via `terminal/create`. After a restart those entities are
 *    gone, so replaying a `tool_call_update` that still references the old
 *    real terminal id makes Zed's `ToolCallContent::from_acp` fail with
 *    "Terminal with id `...` not found". The whole update is dropped and the
 *    tool call stays pending with no visible result.
 *
 * 2. Zed renders tool calls of kind `execute` through `render_terminal_tool_call`
 *    (a `TerminalToolHeader` that OWNS the expand/unfold toggle) only when
 *    the tool call has a `terminal` content item. A card that only carries
 *    text content is rendered via `render_collapsible_command`, which has NO
 *    toggle at all — so even the persisted text was unreachable. That is why
 *    the earlier fix of "strip the terminal, keep the text" still left cards
 *    looking empty: there was no way to expand them.
 *
 * Fix: use Zed's display-only terminal metadata — the same mechanism Codex
 * uses (Zed PR #39419 "Display-only ACP terminals"):
 *
 * - `tool-execution.ts` attaches `_meta.terminal_info = { terminal_id, cwd }`
 *   to every bash `tool_call`. On EVERY `session/update` notification —
 *   including the replayed ones during `session/load` — Zed's
 *   `handle_session_notification` pre-creates a display-only terminal for
 *   that id and registers it in the thread's terminal map.
 * - The final `tool_call_update` streams the recorded output into that
 *   terminal via `_meta.terminal_output` and marks completion via
 *   `_meta.terminal_exit`. Replaying the update therefore resolves the
 *   `terminal` content item, `ViewEvent::NewTerminal` fires and the card
 *   auto-expands (`expand_terminal_card` defaults to true), showing the
 *   output inside a real terminal-style card with an unfold toggle.
 *
 * The real `terminal/create` id is deliberately never embedded in content:
 * it only exists in the Zed process that ran the command. Replay rewrites it
 * to the display-only id (below). For sessions persisted BEFORE this
 * mechanism existed, the same metadata is synthesized from the persisted raw
 * output so old bash cards render identically.
 */
const TERMINAL_INFO_META = 'terminal_info';

function readMeta(event: SessionUpdate): Record<string, unknown> | undefined {
  return (event as { _meta?: Record<string, unknown> | null })._meta ?? undefined;
}

function displayTerminalIdFromMeta(event: SessionUpdate): string | undefined {
  if (event.sessionUpdate !== 'tool_call') {
    return undefined;
  }
  const info = readMeta(event)?.[TERMINAL_INFO_META];
  if (
    typeof info === 'object' &&
    info !== null &&
    typeof (info as { terminal_id?: unknown }).terminal_id === 'string'
  ) {
    return (info as { terminal_id: string }).terminal_id;
  }
  return undefined;
}

type ToolCallUpdateLike = SessionUpdate & {
  content?: Array<{
    type: string;
    terminalId?: string;
    content?: { type?: string; text?: unknown };
    [k: string]: unknown;
  }>;
};

function hasTerminalContent(event: SessionUpdate): boolean {
  return ((event as ToolCallUpdateLike).content ?? []).some((item) => item.type === 'terminal');
}

function textContentOf(event: SessionUpdate): string | undefined {
  const item = ((event as ToolCallUpdateLike).content ?? []).find(
    (candidate) =>
      candidate.type === 'content' &&
      candidate.content?.type === 'text' &&
      typeof candidate.content.text === 'string',
  );
  if (!item || item.type !== 'content' || item.content?.type !== 'text') {
    return undefined;
  }
  return item.content.text as string;
}

/** Deterministic display-only terminal id used for legacy sessions. */
function synthesizedDisplayId(toolCallId: string): string {
  return `zen-${toolCallId}`;
}

/**
 * Remap the tool call events so they replay cleanly after a Zed restart.
 *
 * The live `tool_call_update` events reference the REAL terminal id returned
 * by `terminal/create`; that terminal only exists inside the Zed process that
 * ran the command and is gone after a restart. Resolving it during replay
 * would make Zed's `ToolCallContent::from_acp` fail with
 * "Terminal with id `...` not found", dropping the whole update and leaving
 * the tool call pending with no visible result.
 *
 * Zed renders bash (tool kind `execute`) cards with a terminal-style header
 * and an expand/unfold toggle ONLY when the tool call actually contains a
 * `terminal` content item (`render_terminal_tool_call` /
 * `TerminalToolHeader`). A text-only card gets no toggle at all, so the
 * persisted output stays invisible. To make every bash card replay with its
 * output visible:
 *
 * - When the `tool_call` event carries `_meta.terminal_info` (new sessions),
 *   the terminal content item is REWRITTEN to that display-only terminal id,
 *   which Zed re-registers from the replayed `tool_call` event.
 * - For legacy sessions (persisted before terminal_info existed) the same is
 *   done with a synthesized id (`zen-<toolCallId>`): the replayed `tool_call`
 *   event is given `_meta.terminal_info` and the final update is given
 *   `_meta.terminal_output`/`_meta.terminal_exit` derived from the persisted
 *   raw output, so Zed streams the output into the display-only terminal.
 *
 * The replayed card then auto-expands (`ViewEvent::NewTerminal`,
 * `expand_terminal_card` defaults to true) and shows the output.
 */
export function prepareReplayEvents(events: SessionUpdate[], cwd?: string): SessionUpdate[] {
  // Only replay tool calls that have BOTH an initial `tool_call` event and a
  // final (completed/failed) `tool_call_update`. Calls interrupted mid-run
  // (e.g. hard-aborted) have no result worth showing, and replaying them as
  // "pending" would leave phantom entries in the thread. Zed also creates a
  // spurious "Tool call not found" failed entry for any update without a
  // matching initial event (`acp_thread.rs::update_tool_call`), which is
  // exactly what we want to avoid on reload.
  //
  // Note: a tool call that finished during a GRACEFUL cancel has both its
  // initial event and a final completed update, so it replays normally.
  const initialCallIds = new Set<string>();
  const finalizedCallIds = new Set<string>();
  const displayTerminalIds = new Map<string, string>();
  const callsWithTerminal = new Set<string>();
  const rawOutputs = new Map<string, Record<string, unknown>>();

  for (const event of events) {
    if (event.sessionUpdate === 'tool_call') {
      initialCallIds.add(event.toolCallId);
      const displayId = displayTerminalIdFromMeta(event);
      if (displayId) {
        displayTerminalIds.set(event.toolCallId, displayId);
      }
    } else if (
      event.sessionUpdate === 'tool_call_update' &&
      (event.status === 'completed' || event.status === 'failed')
    ) {
      finalizedCallIds.add(event.toolCallId);
      if (hasTerminalContent(event)) {
        callsWithTerminal.add(event.toolCallId);
      }
      const rawOutput = event.rawOutput as Record<string, unknown> | undefined;
      if (rawOutput && typeof rawOutput === 'object') {
        rawOutputs.set(event.toolCallId, rawOutput);
      }
    }
  }

  const replayableCallIds = new Set([...initialCallIds].filter((id) => finalizedCallIds.has(id)));

  // Legacy sessions: give bash calls that had terminal cards a synthesized
  // display-only terminal id so they render with a toggle and visible output.
  for (const id of callsWithTerminal) {
    if (!displayTerminalIds.has(id)) {
      displayTerminalIds.set(id, synthesizedDisplayId(id));
    }
  }

  const result: SessionUpdate[] = [];
  for (const event of events) {
    if (event.sessionUpdate === 'tool_call') {
      if (!replayableCallIds.has(event.toolCallId)) {
        continue;
      }
      const displayId = displayTerminalIds.get(event.toolCallId);
      if (displayId && !displayTerminalIdFromMeta(event)) {
        // Attach the terminal_info that Zed's pre-handle needs to register
        // the display-only terminal during replay.
        result.push({
          ...event,
          _meta: {
            ...(readMeta(event) ?? {}),
            [TERMINAL_INFO_META]: {
              terminal_id: displayId,
              ...(cwd ? { cwd } : {}),
            },
          },
        } as SessionUpdate);
      } else {
        result.push(event);
      }
      continue;
    }
    if (event.sessionUpdate === 'tool_call_update') {
      if (!replayableCallIds.has(event.toolCallId) || event.status === 'in_progress') {
        // Drop transient in-progress updates. The final update carries the
        // full status and content, so replaying the intermediate one would
        // only cause extra work (and fail on stale terminal references).
        continue;
      }
      result.push(
        remapTerminalContent(
          event,
          displayTerminalIds.get(event.toolCallId),
          rawOutputs.get(event.toolCallId),
        ),
      );
      continue;
    }
    result.push(event);
  }
  return result;
}

/**
 * Rewrites a replayed `tool_call_update`'s terminal content items to the
 * display-only terminal id (stale real ids from the previous Zed process
 * would otherwise fail to resolve). For legacy sessions the output is also
 * streamed into the display-only terminal via `_meta.terminal_output` /
 * `_meta.terminal_exit`, derived from the persisted raw output.
 */
function remapTerminalContent(
  event: SessionUpdate,
  displayTerminalId: string | undefined,
  rawOutput: Record<string, unknown> | undefined,
): SessionUpdate {
  if (event.sessionUpdate !== 'tool_call_update' || !event.content) {
    return event;
  }

  let changed = false;
  const content: typeof event.content = [];
  for (const item of event.content) {
    if (item.type !== 'terminal') {
      content.push(item);
      continue;
    }
    if (displayTerminalId) {
      if (item.terminalId !== displayTerminalId) {
        changed = true;
        content.push({ ...item, terminalId: displayTerminalId });
      } else {
        content.push(item);
      }
      continue;
    }
    // No display-only terminal to resolve against (non-bash tool call):
    // drop the terminal item and keep any persisted text output.
    changed = true;
  }

  // Shallow-clone: meta may alias the persisted event's _meta object, and
  // writing derived keys into it would mutate session.events (and get
  // re-persisted on the next save). Only new keys are written here, so a
  // shallow copy is sufficient.
  const meta = { ...(readMeta(event) ?? {}) };
  if (displayTerminalId && typeof meta.terminal_output !== 'object') {
    const raw = rawOutput as { output?: unknown; exitCode?: unknown; signal?: unknown } | undefined;
    const data = typeof raw?.output === 'string' ? raw.output : textContentOf(event);
    if (data !== undefined) {
      changed = true;
      meta.terminal_output = {
        terminal_id: displayTerminalId,
        data,
      };
      meta.terminal_exit = {
        terminal_id: displayTerminalId,
        exit_code: typeof raw?.exitCode === 'number' ? raw.exitCode : null,
        signal: typeof raw?.signal === 'string' ? raw.signal : null,
      };
    }
  }

  if (!changed) {
    return event;
  }
  return {
    ...event,
    content,
    ...(Object.keys(meta).length > 0 ? { _meta: meta } : {}),
  } as SessionUpdate;
}

export function coalesceReplayEvents(events: SessionUpdate[]): SessionUpdate[] {
  const result: SessionUpdate[] = [];

  for (const event of events) {
    const enriched = enrichReplayEvent(event);
    const last = result[result.length - 1];

    if (
      last &&
      (enriched.sessionUpdate === 'agent_thought_chunk' ||
        enriched.sessionUpdate === 'agent_message_chunk') &&
      last.sessionUpdate === enriched.sessionUpdate &&
      'messageId' in last &&
      'messageId' in enriched &&
      last.messageId === enriched.messageId &&
      last.content.type === 'text' &&
      enriched.content.type === 'text'
    ) {
      result[result.length - 1] = {
        ...last,
        content: {
          type: 'text',
          text: last.content.text + enriched.content.text,
        },
      } as SessionUpdate;
    } else {
      result.push(enriched);
    }
  }

  return result;
}

export function enrichReplayEvent(event: SessionUpdate): SessionUpdate {
  if (event.sessionUpdate !== 'tool_call_update') {
    return event;
  }
  const rawOutput = event.rawOutput as { output?: unknown } | undefined;
  if (!rawOutput || typeof rawOutput.output !== 'string') {
    return event;
  }

  const hasTextContent = (event.content ?? []).some(
    (item) =>
      item.type === 'content' &&
      item.content.type === 'text' &&
      typeof item.content.text === 'string',
  );
  if (hasTextContent) {
    return event;
  }

  return {
    ...event,
    content: [
      ...(event.content ?? []),
      {
        type: 'content',
        content: { type: 'text', text: rawOutput.output },
      },
    ],
  } as SessionUpdate;
}
