import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ZenAgent } from '../index.js';
import type { ActiveSession } from '../core.js';
import { clientLogPath, emptySessionUsage, type StoredSession } from '../../session/storage.js';

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

describe('session load normalization reporting', () => {
  let cwd: string;
  let previousDataHome: string | undefined;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-load-diagnostics-'));
    previousDataHome = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = join(cwd, 'xdg');
  });

  afterEach(() => {
    if (previousDataHome === undefined) delete process.env.XDG_DATA_HOME;
    else process.env.XDG_DATA_HOME = previousDataHome;
    rmSync(cwd, { recursive: true, force: true });
  });

  it('loadSession surfaces the dropped-entry count on the runtime log line', async () => {
    const sessionId = 'sess_dropped';
    const stale = makeSession(sessionId, cwd, 'with garbage');
    stale.events.push({ noSessionUpdate: true } as never);
    stale.llmMessages.push({ role: 'bogus' } as never);
    mkdirSync(join(cwd, '.sessions', sessionId), { recursive: true });
    writeFileSync(join(cwd, '.sessions', sessionId, 'state.json'), JSON.stringify(stale), 'utf8');

    const agent = new ZenAgent() as TestAgent;
    const cx = {
      notify: async () => {},
      request: async () => {
        throw new Error('unexpected client request');
      },
    } as unknown as acp.AgentContext;

    await agent.loadSession({ cwd, sessionId } as acp.LoadSessionRequest, cx);

    const startupKey = (agent as unknown as { startupLogKey: string }).startupLogKey;
    const logFile = clientLogPath(cwd, startupKey);
    // logRuntime is fire-and-forget; poll until the line lands.
    await vi.waitFor(async () => {
      const raw = await readFile(logFile, 'utf8');
      const entries = raw
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      const loaded = entries.find((entry) => entry.message === 'session loaded');
      expect(loaded).toBeDefined();
      expect(loaded?.droppedEntries).toBe(2);
    });

    // Let the deferred available-commands notification finish before cleanup.
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
});
