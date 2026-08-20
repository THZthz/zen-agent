import type { SessionUpdate } from "@agentclientprotocol/sdk";

export function prepareReplayEvents(events: SessionUpdate[]): SessionUpdate[] {
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
  for (const event of events) {
    if (event.sessionUpdate === "tool_call") {
      initialCallIds.add(event.toolCallId);
    } else if (
      event.sessionUpdate === "tool_call_update" &&
      (event.status === "completed" || event.status === "failed")
    ) {
      finalizedCallIds.add(event.toolCallId);
    }
  }

  const replayableCallIds = new Set(
    [...initialCallIds].filter((id) => finalizedCallIds.has(id)),
  );

  const result: SessionUpdate[] = [];
  for (const event of events) {
    if (event.sessionUpdate === "tool_call") {
      if (replayableCallIds.has(event.toolCallId)) {
        result.push(event);
      }
      continue;
    }
    if (event.sessionUpdate === "tool_call_update") {
      if (
        !replayableCallIds.has(event.toolCallId) ||
        event.status === "in_progress"
      ) {
        // Drop transient in-progress updates. The final update carries the
        // full status and content, so replaying the intermediate one would
        // only cause extra work (and fail on stale terminal references).
        continue;
      }
      result.push(stripTerminalContent(event));
      continue;
    }
    result.push(event);
  }
  return result;
}

export function stripTerminalContent(event: SessionUpdate): SessionUpdate {
  // Zed binds terminal cards to live terminals by ID; a stale terminalId
  // from a previous Zed run cannot be re-attached, so replay the text
  // output instead of the terminal reference.
  if (event.sessionUpdate !== "tool_call_update" || !event.content) {
    return event;
  }
  const content = event.content.filter((item) => item.type !== "terminal");
  if (content.length === event.content.length) {
    return event;
  }
  return { ...event, content } as SessionUpdate;
}

export function coalesceReplayEvents(events: SessionUpdate[]): SessionUpdate[] {
  const result: SessionUpdate[] = [];

  for (const event of events) {
    const enriched = enrichReplayEvent(event);
    const last = result[result.length - 1];

    if (
      last &&
      (enriched.sessionUpdate === "agent_thought_chunk" ||
        enriched.sessionUpdate === "agent_message_chunk") &&
      last.sessionUpdate === enriched.sessionUpdate &&
      "messageId" in last &&
      "messageId" in enriched &&
      last.messageId === enriched.messageId &&
      last.content.type === "text" &&
      enriched.content.type === "text"
    ) {
      result[result.length - 1] = {
        ...last,
        content: {
          type: "text",
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
  if (event.sessionUpdate !== "tool_call_update") {
    return event;
  }
  const rawOutput = event.rawOutput as { output?: unknown } | undefined;
  if (!rawOutput || typeof rawOutput.output !== "string") {
    return event;
  }

  const hasTextContent = (event.content ?? []).some(
    (item) =>
      item.type === "content" &&
      item.content.type === "text" &&
      typeof item.content.text === "string",
  );
  if (hasTextContent) {
    return event;
  }

  return {
    ...event,
    content: [
      ...(event.content ?? []),
      {
        type: "content",
        content: { type: "text", text: rawOutput.output },
      },
    ],
  } as SessionUpdate;
}
