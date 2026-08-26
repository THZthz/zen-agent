import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';
import type { SessionUpdate } from '@agentclientprotocol/sdk';
import { ZenAgent } from './agent.js';
import { prepareReplayEvents } from './replay.js';
import { emptySessionUsage, type StoredSession } from './storage.js';

type ReplayEvent = {
  sessionUpdate: string;
  toolCallId?: string;
  status?: string;
  content?: unknown;
  [k: string]: unknown;
};

function prepare(events: ReplayEvent[], cwd?: string): SessionUpdate[] {
  return prepareReplayEvents(events as unknown as SessionUpdate[], cwd);
}

function makeSession(sessionId = 's1'): StoredSession {
  return {
    sessionId,
    cwd: '/tmp',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    title: null,
    events: [],
    llmMessages: [],
    config: {
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      thinkingEffort: 'off',
      systemPrompt: '',
      sandbox: false,
      toolsEnabled: true,
    },
    usage: emptySessionUsage(),
    turnStats: [],
  };
}

// Cast through unknown: ZenAgent's `sessions` map is private, so a plain
// intersection with a public duplicate would collapse to `never`.
type TestAgent = {
  sessions: Map<
    string,
    {
      session: StoredSession;
      abortController: AbortController | null;
      gracefulCancel: boolean;
      cancelTimer: NodeJS.Timeout | null;
    }
  >;
  abortActiveSession(sessionId: string): void;
  cancel(params: { sessionId: string }): void;
};

describe('newSession default provider', () => {
  it('creates sessions on deepseek with DeepSeek model options', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-test-'));
    try {
      const agent = new ZenAgent();
      await agent.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
      } as acp.InitializeRequest);
      const cx = {
        notify: async () => {},
        request: async () => {
          throw new Error('unexpected client request');
        },
      } as unknown as acp.AgentContext;
      const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
      const providerOption = created.configOptions?.find((o) => o.id === 'provider') as
        { currentValue?: string } | undefined;
      expect(providerOption?.currentValue).toBe('deepseek');
      const modelOption = created.configOptions?.find((o) => o.id === 'model') as
        { options?: Array<{ value: string }> } | undefined;
      expect(modelOption?.options?.some((o) => o.value === 'deepseek-v4-flash')).toBe(true);
      const thinkingOption = created.configOptions?.find((o) => o.id === 'thinking_effort') as
        { options?: Array<{ value: string }> } | undefined;
      expect(thinkingOption?.options?.map((o) => o.value)).toEqual(['off', 'low', 'high', 'max']);
      const active = (
        agent as unknown as {
          sessions: Map<string, { session: StoredSession }>;
        }
      ).sessions.get(created.sessionId);
      expect(active?.session.config.provider).toBe('deepseek');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});

describe('setSessionConfigOption', () => {
  const originalEnv = { ...process.env };

  function register(agent: ZenAgent, session: StoredSession): void {
    (agent as unknown as TestAgent).sessions.set(session.sessionId, {
      session,
      abortController: null,
      gracefulCancel: false,
      cancelTimer: null,
    });
  }

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it('switches provider before the first message and resets the model to the provider default', async () => {
    delete process.env.OPENROUTER_MODEL;
    const agent = new ZenAgent();
    const session = makeSession();
    register(agent, session);

    const res = await agent.setSessionConfigOption({
      sessionId: session.sessionId,
      configId: 'provider',
      value: 'openrouter',
    });

    expect(session.config.provider).toBe('openrouter');
    expect(session.config.model).toBe('openrouter/free');
    const providerOption = res.configOptions?.find((o) => o.id === 'provider') as
      { currentValue?: string } | undefined;
    expect(providerOption?.currentValue).toBe('openrouter');
    const modelOption = res.configOptions?.find((o) => o.id === 'model') as
      { options?: Array<{ value: string }> } | undefined;
    expect(modelOption?.options?.some((o) => o.value === 'deepseek-v4-flash')).toBe(false);
    expect(modelOption?.options?.some((o) => o.value === 'openrouter/free')).toBe(true);
  });

  it('rejects unknown providers', async () => {
    const agent = new ZenAgent();
    const session = makeSession();
    register(agent, session);

    await expect(
      agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'provider',
        value: 'anthropic',
      }),
    ).rejects.toThrow(/Unknown provider/);
  });

  it('locks provider, model and thinking effort after the first user message', async () => {
    const agent = new ZenAgent();
    const session = makeSession();
    session.llmMessages = [{ role: 'user', content: 'hi', name: 'Amias' }];
    register(agent, session);

    const base = { sessionId: session.sessionId };
    await expect(
      agent.setSessionConfigOption({ ...base, configId: 'provider', value: 'openrouter' }),
    ).rejects.toThrow(/after the first message/);
    await expect(
      agent.setSessionConfigOption({ ...base, configId: 'model', value: 'deepseek-v4-pro' }),
    ).rejects.toThrow(/after the first message/);
    await expect(
      agent.setSessionConfigOption({ ...base, configId: 'thinking_effort', value: 'high' }),
    ).rejects.toThrow(/after the first message/);
    // Nothing was applied.
    expect(session.config.provider).toBe('deepseek');
    expect(session.config.model).toBe('deepseek-v4-flash');
    expect(session.config.thinkingEffort).toBe('off');
  });

  it('still allows changes when only environment messages exist', async () => {
    const agent = new ZenAgent();
    const session = makeSession();
    session.llmMessages = [{ role: 'user', content: '<environment>...', name: 'Environment' }];
    register(agent, session);

    const res = await agent.setSessionConfigOption({
      sessionId: session.sessionId,
      configId: 'thinking_effort',
      value: 'max',
    });

    expect(session.config.thinkingEffort).toBe('max');
    expect(res.configOptions?.find((o) => o.id === 'thinking_effort')).toBeDefined();
  });

  it('accepts the full thinking-effort vocabulary and rejects unknown values', async () => {
    const agent = new ZenAgent();
    const session = makeSession();
    register(agent, session);

    for (const effort of ['minimal', 'low', 'medium', 'xhigh'] as const) {
      const res = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'thinking_effort',
        value: effort,
      });
      expect(session.config.thinkingEffort).toBe(effort);
      expect(res.configOptions?.find((o) => o.id === 'thinking_effort')).toBeDefined();
    }
    await expect(
      agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'thinking_effort',
        value: 'turbo',
      }),
    ).rejects.toThrow(/Unknown thinking effort/);
  });

  it("offers the OpenRouter model's supported_efforts in the thinking selector", async () => {
    let server: import('node:http').Server | undefined;
    let catalogHits = 0;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            catalogHits += 1;
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(
              JSON.stringify({
                data: [
                  {
                    id: 'vendor/glm',
                    reasoning: {
                      supported_efforts: ['max', 'high', 'low'],
                      default_effort: 'max',
                    },
                  },
                ],
              }),
            );
            return;
          }
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({}));
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    try {
      process.env.OPENROUTER_API_KEY = 'test';
      process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
      const agent = new ZenAgent();
      const session = makeSession();
      register(agent, session);

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'provider',
        value: 'openrouter',
      });
      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'model',
        value: 'vendor/glm',
      });

      const res = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'thinking_effort',
        value: 'max',
      });
      const thinkingOption = res.configOptions?.find((o) => o.id === 'thinking_effort') as
        { options?: Array<{ value: string }> } | undefined;
      expect(thinkingOption?.options?.map((o) => o.value)).toEqual(['off', 'low', 'high', 'max']);
      expect(session.config.thinkingEffort).toBe('max');
      expect(catalogHits).toBeGreaterThan(0);
    } finally {
      server?.close();
      server = undefined;
      // The models cache is keyed by baseUrl, so the next test's port gets a
      // fresh lookup; nothing else to reset here.
    }
  });

  it('folds minimal into low in the thinking selector and sorts ascending', async () => {
    let server: import('node:http').Server | undefined;
    const port = await new Promise<number>((resolve) => {
      const srv = require('node:http').createServer(
        (req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse) => {
          if (req.url?.endsWith('/models')) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(
              JSON.stringify({
                data: [
                  {
                    // gemini-style: minimal but no low tier.
                    id: 'vendor/gemini',
                    reasoning: {
                      supported_efforts: ['high', 'minimal'],
                      default_effort: 'medium',
                    },
                  },
                  {
                    // gpt-5-style: both minimal and low.
                    id: 'vendor/gpt5',
                    reasoning: {
                      supported_efforts: ['high', 'medium', 'low', 'minimal'],
                      default_effort: 'medium',
                    },
                  },
                ],
              }),
            );
            return;
          }
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({}));
        },
      );
      server = srv;
      srv.listen(0, () => {
        const addr = srv.address() as import('node:net').AddressInfo;
        resolve(addr.port);
      });
    });

    try {
      process.env.OPENROUTER_API_KEY = 'test';
      process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${port}/api/v1`;
      const agent = new ZenAgent();
      const session = makeSession();
      register(agent, session);

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'provider',
        value: 'openrouter',
      });

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'model',
        value: 'vendor/gemini',
      });
      let res = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'thinking_effort',
        value: 'low',
      });
      let thinkingOption = res.configOptions?.find((o) => o.id === 'thinking_effort') as
        { options?: Array<{ value: string }> } | undefined;
      // Minimal-only model still offers Low (maps to minimal on the wire),
      // and the list is ascending — no duplicate low-ish tier.
      expect(thinkingOption?.options?.map((o) => o.value)).toEqual(['off', 'low', 'high']);

      await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'model',
        value: 'vendor/gpt5',
      });
      res = await agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: 'thinking_effort',
        value: 'medium',
      });
      thinkingOption = res.configOptions?.find((o) => o.id === 'thinking_effort') as
        { options?: Array<{ value: string }> } | undefined;
      // minimal never appears as its own option alongside low.
      expect(thinkingOption?.options?.map((o) => o.value)).toEqual([
        'off',
        'low',
        'medium',
        'high',
      ]);
    } finally {
      server?.close();
      server = undefined;
    }
  });
});

describe('prepareReplayEvents', () => {
  it('synthesizes a display-only terminal for legacy bash calls so output stays visible', () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'in_progress',
        content: [{ type: 'terminal', terminalId: 't1' }],
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 't1' },
          { type: 'content', content: { type: 'text', text: 'out' } },
        ],
        rawOutput: { output: 'out' },
      },
      // Interrupted call: initial event but no final update -> dropped.
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c2',
        title: '$ sleep',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'sleep' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c2',
        status: 'in_progress',
        content: [{ type: 'terminal', terminalId: 't2' }],
      },
      // Orphan update without a tool_call event -> dropped.
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c3',
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: 'orphan' } }],
      },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'hi' } },
    ];

    expect(prepare(events, '/tmp')).toEqual([
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
        // Synthesized so Zed re-registers a display-only terminal on load.
        _meta: {
          terminal_info: { terminal_id: 'zen-c1', cwd: '/tmp' },
        },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 'zen-c1' },
          { type: 'content', content: { type: 'text', text: 'out' } },
        ],
        rawOutput: { output: 'out' },
        // Synthesized from the persisted raw output so Zed streams the
        // output into the display-only terminal on load.
        _meta: {
          terminal_output: { terminal_id: 'zen-c1', data: 'out' },
          terminal_exit: { terminal_id: 'zen-c1', exit_code: null, signal: null },
        },
      },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'hi' } },
    ]);
  });

  it('keeps failed bash calls with a synthesized display terminal and exit code', () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'f1',
        title: '$ nope',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'nope' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'f1',
        status: 'failed',
        content: [
          { type: 'terminal', terminalId: 't9' },
          { type: 'content', content: { type: 'text', text: 'boom' } },
        ],
        rawOutput: { output: 'boom', exitCode: 127 },
      },
    ];
    expect(prepare(events)).toEqual([
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'f1',
        title: '$ nope',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'nope' },
        _meta: {
          terminal_info: { terminal_id: 'zen-f1' },
        },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'f1',
        status: 'failed',
        content: [
          { type: 'terminal', terminalId: 'zen-f1' },
          { type: 'content', content: { type: 'text', text: 'boom' } },
        ],
        rawOutput: { output: 'boom', exitCode: 127 },
        _meta: {
          terminal_output: { terminal_id: 'zen-f1', data: 'boom' },
          terminal_exit: { terminal_id: 'zen-f1', exit_code: 127, signal: null },
        },
      },
    ]);
  });

  it('leaves text-only tool calls (no terminal card) untouched', () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'g1',
        title: '$ npm test',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'npm test' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'g1',
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: 'passed' } }],
        rawOutput: { output: 'passed' },
      },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Turn stats' } },
    ];
    const out = prepare(events);
    expect(out).toEqual(events);
    expect(out.filter((e) => e.sessionUpdate === 'tool_call')).toHaveLength(1);
  });
});

describe('replay preparation on a session with terminal content', () => {
  it('produces a replay stream with no terminal content and no orphan/in-progress updates', () => {
    // Representative event stream as persisted by the agent: tool calls with
    // terminal cards, an in-progress update, an orphan update and a failed
    // call with a text result.
    const events: ReplayEvent[] = [
      { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'hello' } },
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'in_progress',
        content: [{ type: 'terminal', terminalId: 't1' }],
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 't1' },
          { type: 'content', content: { type: 'text', text: 'a.txt' } },
        ],
        rawOutput: { output: 'a.txt' },
      },
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c2',
        title: '$ git status',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'git status' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c2',
        status: 'in_progress',
        content: [{ type: 'terminal', terminalId: 't2' }],
      },
      // Orphan update: no matching tool_call, must be dropped.
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'orphan',
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: 'x' } }],
      },
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c3',
        title: '$ nope',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'nope' },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'c3',
        status: 'failed',
        content: [{ type: 'content', content: { type: 'text', text: 'boom' } }],
        rawOutput: { output: 'boom', exitCode: 127 },
      },
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'done' } },
    ];
    const out = prepare(events);

    const toolCalls = out.filter((e) => e.sessionUpdate === 'tool_call');
    const updates = out.filter((e) => e.sessionUpdate === 'tool_call_update');
    const finalUpdates = updates.filter((e) => e.status === 'completed' || e.status === 'failed');
    const inProgress = updates.filter((e) => e.status === 'in_progress');

    expect(inProgress).toHaveLength(0);
    expect(toolCalls).toHaveLength(finalUpdates.length);
    // c1 (completed) and c3 (failed) survive; c2 has no final update and
    // the orphan update is dropped.
    expect(toolCalls).toHaveLength(2);
    // c1 had a terminal card: it replays with a synthesized display-only
    // terminal (zen-c1) plus its text output, so Zed can re-render the card
    // expanded with the output visible.
    const c1Call = toolCalls.find((e) => e.toolCallId === 'c1')!;
    expect(
      (c1Call as unknown as { _meta?: { terminal_info?: unknown } })._meta?.terminal_info,
    ).toEqual({ terminal_id: 'zen-c1' });
    const c1Update = updates.find((u) => u.toolCallId === 'c1')!;
    expect(c1Update.content ?? []).toContainEqual(
      expect.objectContaining({ type: 'terminal', terminalId: 'zen-c1' }),
    );
    expect(
      (c1Update as unknown as { _meta?: { terminal_output?: { data?: unknown } } })._meta
        ?.terminal_output?.data,
    ).toBe('a.txt');
    // c3 failed before a terminal card existed: text-only, no terminal.
    const c3Update = updates.find((u) => u.toolCallId === 'c3')!;
    expect(c3Update.content ?? []).not.toContainEqual(
      expect.objectContaining({ type: 'terminal' }),
    );
    // All kept updates must have a matching kept tool_call.
    const ids = new Set(toolCalls.map((e) => e.toolCallId));
    for (const u of updates) {
      expect(ids.has(u.toolCallId!)).toBe(true);
    }
  });
});

describe('graceful cancel', () => {
  it('sets gracefulCancel instead of aborting when a turn is running', () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    const controller = new AbortController();
    agent.sessions.set('s1', {
      session: makeSession(),
      abortController: controller,
      gracefulCancel: false,
      cancelTimer: null,
    });

    agent.cancel({ sessionId: 's1' });

    expect(controller.signal.aborted).toBe(false);
    const entry = agent.sessions.get('s1')!;
    expect(entry.gracefulCancel).toBe(true);
    expect(entry.cancelTimer).toBeNull();
  });

  it('does nothing when no turn is running', () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    agent.sessions.set('s1', {
      session: makeSession(),
      abortController: null,
      gracefulCancel: false,
      cancelTimer: null,
    });

    agent.cancel({ sessionId: 's1' });

    const entry = agent.sessions.get('s1')!;
    expect(entry.gracefulCancel).toBe(false);
  });

  it('does nothing for an unknown session', () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    expect(() => agent.cancel({ sessionId: 'missing' })).not.toThrow();
  });

  it('hard abort (session close) clears graceful state and aborts', () => {
    const agent = new ZenAgent() as unknown as TestAgent;
    const controller = new AbortController();
    agent.sessions.set('s1', {
      session: makeSession(),
      abortController: controller,
      gracefulCancel: true,
      cancelTimer: null,
    });

    agent.abortActiveSession('s1');

    expect(controller.signal.aborted).toBe(true);
    const entry = agent.sessions.get('s1')!;
    expect(entry.gracefulCancel).toBe(false);
  });

  it('schedules a hard-abort timer when the timeout env is set', async () => {
    const { vi } = await import('vitest');
    vi.resetModules();
    process.env.ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS = '10';
    try {
      const { ZenAgent: ReloadedZenAgent } = await import('./agent.js');
      const agent = new ReloadedZenAgent() as unknown as TestAgent;
      const controller = new AbortController();
      agent.sessions.set('s1', {
        session: makeSession(),
        abortController: controller,
        gracefulCancel: false,
        cancelTimer: null,
      });

      agent.cancel({ sessionId: 's1' });

      const entry = agent.sessions.get('s1')!;
      expect(entry.gracefulCancel).toBe(true);
      expect(entry.cancelTimer).not.toBeNull();

      await new Promise((resolve) => setTimeout(resolve, 40));
      expect(controller.signal.aborted).toBe(true);
      expect(agent.sessions.get('s1')!.cancelTimer).toBeNull();
    } finally {
      delete process.env.ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS;
    }
  });
});

describe('loadSession environment backfill', () => {
  it('prepends a frozen environment message and appends a continuation notice for empty sessions', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const { vi } = await import('vitest');

    const dir = mkdtempSync(join(tmpdir(), 'zen-agent-load-empty-'));
    try {
      const agent = new ZenAgent() as unknown as TestAgent & {
        loadSession(
          params: { cwd: string; sessionId: string },
          cx: {
            notify: (
              method: string,
              params: { sessionId: string; update: unknown },
            ) => Promise<void>;
          },
        ): Promise<unknown>;
      };

      const session = makeSession('sess_empty');
      session.cwd = dir;
      mkdirSync(join(dir, '.sessions', 'sess_empty'), { recursive: true });
      writeFileSync(
        join(dir, '.sessions', 'sess_empty', 'state.json'),
        JSON.stringify(session),
        'utf8',
      );

      const notify = vi.fn(async () => {});
      const cx = { notify } as unknown as Parameters<typeof agent.loadSession>[1];

      await agent.loadSession({ cwd: dir, sessionId: 'sess_empty' }, cx);

      const loaded = agent.sessions.get('sess_empty')!.session;
      expect(loaded.llmMessages).toHaveLength(2);
      expect(loaded.llmMessages[0]).toMatchObject({ role: 'user', name: 'Environment' });
      expect(String(loaded.llmMessages[0]!.content)).toContain('<working-directory>');
      expect(loaded.llmMessages[1]).toMatchObject({ role: 'user', name: 'Environment' });
      expect(String(loaded.llmMessages[1]!.content)).toContain(
        '<session-state>resumed</session-state>',
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('injects no environment messages when the session has tools off', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const { vi } = await import('vitest');

    const dir = mkdtempSync(join(tmpdir(), 'zen-agent-load-noenv-'));
    try {
      const agent = new ZenAgent() as unknown as TestAgent & {
        loadSession(
          params: { cwd: string; sessionId: string },
          cx: {
            notify: (
              method: string,
              params: { sessionId: string; update: unknown },
            ) => Promise<void>;
          },
        ): Promise<unknown>;
      };

      const session = makeSession('sess_noenv');
      session.cwd = dir;
      session.config.toolsEnabled = false;
      session.llmMessages = [];
      mkdirSync(join(dir, '.sessions', 'sess_noenv'), { recursive: true });
      writeFileSync(
        join(dir, '.sessions', 'sess_noenv', 'state.json'),
        JSON.stringify(session),
        'utf8',
      );

      const notify = vi.fn(async () => {});
      const cx = { notify } as unknown as Parameters<typeof agent.loadSession>[1];

      await agent.loadSession({ cwd: dir, sessionId: 'sess_noenv' }, cx);

      const loaded = agent.sessions.get('sess_noenv')!.session;
      expect(loaded.llmMessages).toHaveLength(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('prepareReplayEvents with display-only terminal info', () => {
  it('rewrites terminal content to the display-only id so it resolves after restart', () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'd1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
        _meta: {
          terminal_info: { terminal_id: 'zen-d1', cwd: '/tmp' },
        },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'd1',
        status: 'in_progress',
        content: [{ type: 'terminal', terminalId: 'real-uuid-1' }],
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'd1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 'real-uuid-1' },
          { type: 'content', content: { type: 'text', text: 'out' } },
        ],
        _meta: {
          terminal_output: { terminal_id: 'zen-d1', data: 'out' },
          terminal_exit: { terminal_id: 'zen-d1', exit_code: 0 },
        },
        rawOutput: { output: 'out' },
      },
    ];

    expect(prepare(events)).toEqual([
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'd1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
        _meta: {
          terminal_info: { terminal_id: 'zen-d1', cwd: '/tmp' },
        },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'd1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 'zen-d1' },
          { type: 'content', content: { type: 'text', text: 'out' } },
        ],
        _meta: {
          terminal_output: { terminal_id: 'zen-d1', data: 'out' },
          terminal_exit: { terminal_id: 'zen-d1', exit_code: 0 },
        },
        rawOutput: { output: 'out' },
      },
    ]);
  });

  it('uses the existing terminal_info id (new sessions) instead of synthesizing', () => {
    const events: ReplayEvent[] = [
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'n1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
        _meta: {
          terminal_info: { terminal_id: 'zen-n1', cwd: '/work' },
        },
      },
      {
        sessionUpdate: 'tool_call_update',
        toolCallId: 'n1',
        status: 'completed',
        content: [
          { type: 'terminal', terminalId: 'real-uuid-9' },
          { type: 'content', content: { type: 'text', text: 'n out' } },
        ],
        _meta: {
          terminal_output: { terminal_id: 'zen-n1', data: 'n out' },
          terminal_exit: { terminal_id: 'zen-n1', exit_code: 0 },
        },
        rawOutput: { output: 'n out', exitCode: 0 },
      },
    ];
    const out = prepare(events, '/tmp');
    expect(out[0]).toEqual(events[0]);
    const update = out[1] as ReplayEvent;
    expect(update.content).toContainEqual(
      expect.objectContaining({ type: 'terminal', terminalId: 'zen-n1' }),
    );
    expect(
      (update as unknown as { _meta?: { terminal_output?: { data?: unknown } } })._meta
        ?.terminal_output?.data,
    ).toBe('n out');
    // cwd passed to prepare() must not override the persisted one.
    expect(
      (out[0] as unknown as { _meta?: { terminal_info?: { cwd?: unknown } } })._meta?.terminal_info
        ?.cwd,
    ).toBe('/work');
  });
});
