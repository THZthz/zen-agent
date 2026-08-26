import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('./deepseek.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from './agent.js';
import { runLlmStep, type LlmStepResult } from './deepseek.js';
import { BASH_TOOL_SCHEMA } from './llm-client.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);

/** Snapshot of each LLM request at call time (options.tools/system). */
const recordedSteps: Array<{ tools?: unknown[]; system?: string }> = [];

function recordStep(result: LlmStepResult): void {
  mockedRunLlmStep.mockImplementationOnce(async (options) => {
    recordedSteps.push({ tools: options.tools, system: options.system });
    return result;
  });
}

function textStep(text: string): LlmStepResult {
  return {
    text,
    reasoning: '',
    toolCalls: [],
    finishReason: 'end_turn',
    usage: null,
  };
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

function agentMessages(notifications: Array<{ update: acp.SessionUpdate }>): string[] {
  return notifications
    .filter((n) => n.update.sessionUpdate === 'agent_message_chunk')
    .map((n) => {
      const c = n.update as { content?: { type?: string; text?: string } };
      return c.content?.text ?? '';
    });
}

type TestAgent = {
  sessions: Map<string, { session: { config: { toolsEnabled: boolean } } }>;
};

describe('/tools slash command', () => {
  let cwd: string;

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-agent-tools-'));
    mockedRunLlmStep.mockReset();
    recordedSteps.length = 0;
  });

  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it('reports ON by default and keeps sending tool schemas', async () => {
    const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
    const status = await agent.prompt(
      { sessionId, prompt: [{ type: 'text', text: '/tools' }] },
      cx,
    );
    expect(status.stopReason).toBe('end_turn');
    expect(agentMessages(notifications).join('\n')).toContain('Tools (bash, read_media): ON');

    recordStep(textStep('ok'));
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'hello' }] }, cx);
    expect(recordedSteps[0]?.tools).toEqual([BASH_TOOL_SCHEMA]);
    expect(recordedSteps[0]?.system).toContain('You have exactly one tool: bash');
  });

  it('turns tools off: no schemas sent, no system-prompt notice, state persists', async () => {
    const { agent, cx, notifications, sessionId } = await setupAgent(cwd);

    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools off' }] }, cx);
    expect(agentMessages(notifications).join('\n')).toContain(
      'Tools (bash, read_media) are now disabled',
    );
    const active = (agent as unknown as TestAgent).sessions.get(sessionId)!;
    expect(active.session.config.toolsEnabled).toBe(false);

    recordStep(textStep('ok'));
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'hello' }] }, cx);
    expect(recordedSteps[0]?.tools).toEqual([]);
    // The bash paragraph must not appear when tools are off.
    expect(recordedSteps[0]?.system).not.toContain('You have exactly one tool: bash');
    expect(recordedSteps[0]?.system).toContain('You are an experienced software engineer');
  });

  it('turns tools back on and restores tool schemas', async () => {
    const { agent, cx, notifications, sessionId } = await setupAgent(cwd);

    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools off' }] }, cx);
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools on' }] }, cx);
    expect(agentMessages(notifications).join('\n')).toContain(
      'Tools (bash, read_media) are now enabled',
    );
    const active = (agent as unknown as TestAgent).sessions.get(sessionId)!;
    expect(active.session.config.toolsEnabled).toBe(true);

    recordStep(textStep('ok'));
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'hello' }] }, cx);
    expect(recordedSteps[0]?.tools).toEqual([BASH_TOOL_SCHEMA]);
    expect(recordedSteps[0]?.system).toContain('You have exactly one tool: bash');
  });

  it('shows OFF in the status after disabling', async () => {
    const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools off' }] }, cx);
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools' }] }, cx);
    expect(agentMessages(notifications).join('\n')).toContain('Tools (bash, read_media): OFF');
  });

  it('rejects unknown arguments without changing state', async () => {
    const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
    const response = await agent.prompt(
      { sessionId, prompt: [{ type: 'text', text: '/tools maybe' }] },
      cx,
    );
    expect(response.stopReason).toBe('end_turn');
    expect(agentMessages(notifications).join('\n')).toContain('Usage: /tools on | off');
    const active = (agent as unknown as TestAgent).sessions.get(sessionId)!;
    expect(active.session.config.toolsEnabled).toBe(true);
  });

  it('refuses tool calls the model still emits while tools are off', async () => {
    const { agent, cx, request, sessionId } = await setupAgent(cwd);
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/tools off' }] }, cx);

    // Defensive path: model ignores the empty tool list and calls bash anyway.
    mockedRunLlmStep.mockResolvedValueOnce(bashStep()).mockResolvedValueOnce(textStep('done'));
    await agent.prompt({ sessionId, prompt: [{ type: 'text', text: 'run it' }] }, cx);

    const terminalCall = request.mock.calls.find(
      (c) => c[0] === acp.methods.client.terminal.create,
    );
    expect(terminalCall).toBeUndefined();

    const active = (
      agent as unknown as {
        sessions: Map<string, { session: { llmMessages: unknown[] } }>;
      }
    ).sessions.get(sessionId)!;
    const toolResult = active.session.llmMessages
      .filter((m) => (m as { role?: string }).role === 'tool')
      .flatMap((m) => (m as { content: Array<{ output?: { value?: string } }> }).content)[0];
    expect(toolResult?.output?.value).toContain('disabled');
    expect(toolResult?.output?.value).toContain('/tools on');
  });
});
