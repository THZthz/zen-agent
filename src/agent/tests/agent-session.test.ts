import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ZenAgent } from '../index.js';
import type { ActiveSession } from '../core.js';
import { emptySessionUsage, writeSession, type StoredSession } from '../../session/storage.js';
import { openDb } from '../../session/db.js';

/** Parsed runtime_log entries for one agent process (startup key). */
function runtimeEntries(startupKey: string): Array<Record<string, unknown>> {
  return (
    openDb()
      .prepare('SELECT entry FROM runtime_log WHERE startup_key = ? ORDER BY seq')
      .all(startupKey) as Array<{ entry: string }>
  ).map((row) => JSON.parse(row.entry) as Record<string, unknown>);
}

function makeSession(sessionId: string, cwd: string, title: string): StoredSession {
  const now = new Date().toISOString();
  return {
    sessionId,
    cwd,
    createdAt: now,
    updatedAt: now,
    title,
    events: [],
    llmMessages: [],
    config: {
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      thinkingEffort: 'off',
      systemPrompt: '',
      sandbox: false,
      roBindPaths: [],
      toolsEnabled: false,
    },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

type TestAgent = ZenAgent & {
  sessions: Map<string, ActiveSession>;
};

describe('session load/resume lifecycle', () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-session-lifecycle-'));
  });

  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  for (const method of ['loadSession', 'resumeSession'] as const) {
    it(`${method} aborts and settles the active turn before reading the session row`, async () => {
      const sessionId = `sess_${method}`;
      const stale = makeSession(sessionId, cwd, 'before final save');
      await writeSession(stale);

      const agent = new ZenAgent() as TestAgent;
      const abortController = new AbortController();
      let signalAborted!: () => void;
      const aborted = new Promise<void>((resolve) => {
        signalAborted = resolve;
      });
      abortController.signal.addEventListener('abort', signalAborted, { once: true });

      let settleTurn!: (response: acp.PromptResponse) => void;
      const turnPromise = new Promise<acp.PromptResponse>((resolve) => {
        settleTurn = resolve;
      });
      agent.sessions.set(sessionId, {
        session: stale,
        abortController,
        gracefulCancel: false,
        cancelTimer: null,
        mediaModalities: null,
        turnPromise,
        mediaModalitiesUnknownLogged: false,
      });

      const cx = {
        notify: async () => {},
        request: async () => {
          throw new Error('unexpected client request');
        },
      } as unknown as acp.AgentContext;

      const lifecycle =
        method === 'loadSession'
          ? agent.loadSession({ cwd, sessionId } as acp.LoadSessionRequest, cx)
          : agent.resumeSession({ cwd, sessionId } as acp.ResumeSessionRequest, cx);

      await aborted;
      // While the old turn settles, prompts cannot retrieve its stale object.
      expect(agent.sessions.has(sessionId)).toBe(false);

      const finalState = {
        ...stale,
        title: 'from final save',
        updatedAt: new Date(Date.now() + 1_000).toISOString(),
      };
      await writeSession(finalState);
      settleTurn({ stopReason: 'cancelled' });

      await lifecycle;
      expect(abortController.signal.aborted).toBe(true);
      expect(agent.sessions.get(sessionId)?.session.title).toBe('from final save');

      // Let the deferred available-commands notification finish before cleanup.
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
});

describe('session load normalization reporting', () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-load-diagnostics-'));
  });

  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it('loadSession surfaces the dropped-entry count on the runtime log line', async () => {
    const sessionId = 'sess_dropped';
    const stale = makeSession(sessionId, cwd, 'with garbage');
    stale.events.push({ noSessionUpdate: true } as never);
    stale.llmMessages.push({ role: 'bogus' } as never);
    await writeSession(stale);

    const agent = new ZenAgent() as TestAgent;
    const cx = {
      notify: async () => {},
      request: async () => {
        throw new Error('unexpected client request');
      },
    } as unknown as acp.AgentContext;

    await agent.loadSession({ cwd, sessionId } as acp.LoadSessionRequest, cx);

    const startupKey = (agent as unknown as { startupLogKey: string }).startupLogKey;
    // logRuntime is fire-and-forget; poll until the entry lands.
    await vi.waitFor(() => {
      const loaded = runtimeEntries(startupKey).find((entry) => entry.message === 'session loaded');
      expect(loaded).toBeDefined();
      expect(loaded?.droppedEntries).toBe(2);
    });

    // Let the deferred available-commands notification finish before cleanup.
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});
