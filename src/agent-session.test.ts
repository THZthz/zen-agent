import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ZenAgent } from './agent.js';
import type { ActiveSession } from './agent-core.js';
import { emptySessionUsage, type StoredSession } from './storage.js';

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
  let previousDataHome: string | undefined;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-session-lifecycle-'));
    previousDataHome = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = join(cwd, 'xdg');
  });

  afterEach(() => {
    if (previousDataHome === undefined) delete process.env.XDG_DATA_HOME;
    else process.env.XDG_DATA_HOME = previousDataHome;
    rmSync(cwd, { recursive: true, force: true });
  });

  for (const method of ['loadSession', 'resumeSession'] as const) {
    it(`${method} aborts and settles the active turn before reading state.json`, async () => {
      const sessionId = `sess_${method}`;
      const stale = makeSession(sessionId, cwd, 'before final save');
      const statePath = join(cwd, '.sessions', sessionId, 'state.json');
      mkdirSync(join(cwd, '.sessions', sessionId), { recursive: true });
      writeFileSync(statePath, JSON.stringify(stale), 'utf8');

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
      writeFileSync(statePath, JSON.stringify(finalState), 'utf8');
      settleTurn({ stopReason: 'cancelled' });

      await lifecycle;
      expect(abortController.signal.aborted).toBe(true);
      expect(agent.sessions.get(sessionId)?.session.title).toBe('from final save');

      // Let the deferred available-commands notification finish before cleanup.
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
});
