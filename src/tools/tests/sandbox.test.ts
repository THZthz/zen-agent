import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('../../providers/index.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from '../../agent/index.js';
import { runLlmStep, type LlmStepResult } from '../../providers/index.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);

function makeAgentContext() {
  const notifications: Array<{ sessionId: string; update: acp.SessionUpdate }> = [];
  const request = vi.fn((method: string, _params?: unknown) => {
    switch (method) {
      case acp.methods.client.terminal.create:
        return Promise.resolve({ terminalId: 't1' });
      case acp.methods.client.terminal.waitForExit:
        return Promise.resolve({ exitCode: 0, signal: null });
      case acp.methods.client.terminal.output:
        return Promise.resolve({ output: 'done', truncated: false });
      case acp.methods.client.terminal.release:
        return Promise.resolve({});
      case acp.methods.client.terminal.kill:
        return Promise.resolve({});
      default:
        return Promise.reject(new Error(`unexpected client request: ${method}`));
    }
  });
  const notify = vi.fn(
    async (method: string, params: { sessionId: string; update: acp.SessionUpdate }) => {
      if (method === acp.methods.client.session.update) {
        notifications.push(params);
      }
    },
  );
  return {
    cx: { request, notify } as unknown as acp.AgentContext,
    notifications,
    request,
  };
}

async function setupAgent(cwd: string) {
  const agent = new ZenAgent();
  await agent.initialize({
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: { terminal: true },
  } as acp.InitializeRequest);
  const { cx, notifications, request } = makeAgentContext();
  const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
  return { agent, cx, notifications, request, sessionId: created.sessionId };
}

function bashStep(): LlmStepResult {
  return {
    text: '',
    reasoning: '',
    toolCalls: [{ id: 'c1', name: 'bash', input: { command: 'echo hi' } }],
    finishReason: 'tool-calls',
    usage: null,
  };
}

/** Queues one bash step followed by a final answer so the turn terminates. */
function queueBashThenAnswer(): void {
  mockedRunLlmStep.mockResolvedValueOnce(bashStep()).mockResolvedValueOnce({
    text: 'done',
    reasoning: '',
    toolCalls: [],
    finishReason: 'end_turn',
    usage: null,
  });
}

function agentMessages(notifications: Array<{ update: acp.SessionUpdate }>): string[] {
  return notifications
    .filter((n) => n.update.sessionUpdate === 'agent_message_chunk')
    .map((n) => {
      const c = n.update as { content?: { type?: string; text?: string } };
      return c.content?.text ?? '';
    });
}

/** The `-lc` script passed to terminal.create, or undefined. */
function createdScript(
  request: ReturnType<typeof makeAgentContext>['request'],
): string | undefined {
  const call = request.mock.calls.find((c) => c[0] === acp.methods.client.terminal.create);
  const params = call?.[1] as { args?: string[] } | undefined;
  return params?.args?.[1];
}

describe('/sandbox slash command', () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  it('reports OFF by default and does not wrap bash calls', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-sandbox-'));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);
      const status = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/sandbox' }] },
        cx,
      );
      expect(status.stopReason).toBe('end_turn');
      expect(agentMessages(notifications).join('\n')).toContain('Bash tool sandbox: OFF (off)');

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(createdScript(request)).not.toContain('bwrap');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('turns the sandbox on and wraps subsequent bash calls in bwrap', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-sandbox-'));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/sandbox on' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain(
        'Bash tool calls are now sandboxed',
      );
      const active = (
        agent as unknown as {
          sessions: Map<string, { session: { config: { sandbox: boolean } } }>;
        }
      ).sessions.get(sessionId)!;
      expect(active.session.config.sandbox).toBe(true);

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      const script = createdScript(request)!;
      expect(script).toContain('bwrap');
      expect(script).toMatch(/bwrap .*--ro-bind \/ \//);
      expect(script).toContain('--bind /tmp /tmp');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('turns the sandbox off and stops wrapping bash calls', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-sandbox-'));
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/sandbox on' }] }, cx);
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/sandbox off' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain('Bash tool sandbox disabled');

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(createdScript(request)).not.toContain('bwrap');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('rejects unknown arguments without changing state', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-sandbox-'));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
      const response = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/sandbox maybe' }] },
        cx,
      );
      expect(response.stopReason).toBe('end_turn');
      expect(agentMessages(notifications).join('\n')).toContain('Usage: /sandbox on | off');
      const active = (
        agent as unknown as {
          sessions: Map<string, { session: { config: { sandbox: boolean } } }>;
        }
      ).sessions.get(sessionId)!;
      expect(active.session.config.sandbox).toBe(false);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('reports ON and refuses to disable when ZEN_AGENT_SANDBOX=1 is set', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-sandbox-'));
    process.env.ZEN_AGENT_SANDBOX = '1';
    try {
      const { agent, cx, notifications, request, sessionId } = await setupAgent(cwd);

      const status = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/sandbox' }] },
        cx,
      );
      expect(agentMessages(notifications).join('\n')).toContain(
        'Bash tool sandbox: ON (enforced by ZEN_AGENT_SANDBOX=1)',
      );

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/sandbox off' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain('Cannot disable');

      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(createdScript(request)).toContain('bwrap');
      void status;
    } finally {
      delete process.env.ZEN_AGENT_SANDBOX;
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});

describe('/writable slash command', () => {
  beforeEach(() => {
    mockedRunLlmStep.mockReset();
  });

  /** The `-lc` script of the most recent terminal.create call. */
  function lastCreatedScript(
    request: ReturnType<typeof makeAgentContext>['request'],
  ): string | undefined {
    const calls = request.mock.calls.filter((c) => c[0] === acp.methods.client.terminal.create);
    const params = calls.at(-1)?.[1] as { args?: string[] } | undefined;
    return params?.args?.[1];
  }

  function sessionConfig(
    agent: ZenAgent,
    sessionId: string,
  ): { sandbox: boolean; writablePaths: string[] } {
    const active = (
      agent as unknown as {
        sessions: Map<
          string,
          { session: { config: { sandbox: boolean; writablePaths: string[] } } }
        >;
      }
    ).sessions.get(sessionId)!;
    return active.session.config;
  }

  it('reports the default writable paths by default', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-writable-'));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);

      const response = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/writable' }] },
        cx,
      );
      expect(response.stopReason).toBe('end_turn');
      expect(agentMessages(notifications).join('\n')).toContain(
        'Writable paths: /tmp, /var/tmp',
      );
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual(['/tmp', '/var/tmp']);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('add, del and clear manage the list and persist it', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-writable-'));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);

      await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/writable add /mnt/data, /mnt/secrets' }] },
        cx,
      );
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual([
        '/tmp',
        '/var/tmp',
        '/mnt/data',
        '/mnt/secrets',
      ]);

      // add dedupes; del removes only the listed entries.
      await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/writable add /mnt/data' }] },
        cx,
      );
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual([
        '/tmp',
        '/var/tmp',
        '/mnt/data',
        '/mnt/secrets',
      ]);

      await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/writable del /mnt/data, /mnt/missing' }] },
        cx,
      );
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual(['/tmp', '/var/tmp', '/mnt/secrets']);

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/writable clear' }] }, cx);
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual([]);
      expect(agentMessages(notifications).join('\n')).toContain(
        'Writable paths cleared for this session',
      );

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/writable' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain(
        'Writable paths: no paths (everything is read-only)',
      );
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('applies the writable paths in subsequent sandboxed bash calls', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-writable-'));
    try {
      const { agent, cx, request, sessionId } = await setupAgent(cwd);

      // Without sandboxing no bwrap is involved.
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/sandbox on' }] }, cx);
      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(lastCreatedScript(request)).toContain('bwrap');
      // Default writable list: /tmp and /var/tmp are bound read-write, root is ro.
      expect(lastCreatedScript(request)).toContain('--ro-bind / /');
      expect(lastCreatedScript(request)).toContain('--bind /tmp /tmp');
      expect(lastCreatedScript(request)).toContain('--bind /var/tmp /var/tmp');

      // Adding a path adds another writable bind.
      await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/writable add /mnt/data' }] },
        cx,
      );
      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(lastCreatedScript(request)).toContain('--bind /mnt/data /mnt/data');

      // Clearing drops the writable binds (only the default ro root remains).
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/writable clear' }] }, cx);
      queueBashThenAnswer();
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);
      expect(lastCreatedScript(request)).not.toContain('--bind /tmp /tmp');
      expect(lastCreatedScript(request)).not.toContain('--bind /var/tmp /var/tmp');
      expect(lastCreatedScript(request)).not.toContain('--bind /mnt/data /mnt/data');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('rejects unknown subcommands and empty add lists without changing state', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-writable-'));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/writable on' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain(
        'Usage: /writable [add <path>[,<path>...] | del <path>[,<path>...] | clear]',
      );
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual(['/tmp', '/var/tmp']);

      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/writable add , ,' }] }, cx);
      expect(agentMessages(notifications).join('\n')).toContain('No valid paths');
      expect(sessionConfig(agent, sessionId).writablePaths).toEqual(['/tmp', '/var/tmp']);
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
