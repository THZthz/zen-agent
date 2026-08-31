import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as acp from '@agentclientprotocol/sdk';

vi.mock('./provider.js', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, runLlmStep: vi.fn() };
});

import { ZenAgent } from './agent.js';
import { runLlmStep, type LlmStepResult } from './provider.js';

const mockedRunLlmStep = vi.mocked(runLlmStep);

let homeDir: string;

beforeEach(() => {
  homeDir = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-home-'));
  process.env.HOME = homeDir;
  delete process.env.ZEN_AGENT_SHOW_SKILLS_CATALOG;
  mockedRunLlmStep.mockReset();
});

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
  };
}

async function setupAgent(cwd: string) {
  const agent = new ZenAgent();
  await agent.initialize({
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: { terminal: true },
  } as acp.InitializeRequest);
  const { cx, notifications } = makeAgentContext();
  const created = await agent.newSession({ cwd, mcpServers: [] } as acp.NewSessionRequest, cx);
  return { agent, cx, notifications, sessionId: created.sessionId };
}

function writeSkill(root: string, name: string, content: string): void {
  const dir = join(root, '.agents', 'skills', name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'SKILL.md'), content, 'utf8');
}

/** A final answer with no tool calls, so the turn ends without terminals. */
function answerStep(text = 'done'): LlmStepResult {
  return {
    text,
    reasoning: '',
    toolCalls: [],
    finishReason: 'end_turn',
    usage: null,
  };
}

const GRILL_ME = `---
name: grill-me
description: A relentless interview to sharpen a plan or design.
disable-model-invocation: true
---
Interview the user relentlessly about the given plan.`;

describe('skill slash commands', () => {
  it('invokes an installed skill via /skill-name regardless of ZEN_AGENT_SHOW_SKILLS_CATALOG', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-project-'));
    try {
      writeSkill(cwd, 'grill-me', GRILL_ME);
      const { agent, cx, sessionId } = await setupAgent(cwd);

      mockedRunLlmStep.mockResolvedValueOnce(answerStep("Let's grill."));
      const response = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/grill-me my plan' }] },
        cx,
      );
      expect(response.stopReason).toBe('end_turn');

      // The model turn started with the skill injected into a user message.
      const messages = mockedRunLlmStep.mock.calls[0]?.[1]?.messages ?? [];
      const skillMessage = messages.find(
        (m) => m.role === 'user' && typeof m.content === 'string' && m.content.includes('grill-me'),
      );
      expect(skillMessage?.content).toContain('<skill-invoked>');
      expect(skillMessage?.content).toContain('<skill-name>grill-me</skill-name>');
      expect(skillMessage?.content).toContain('<skill-argument>\nmy plan\n</skill-argument>');
      expect(skillMessage?.content).toContain('Interview the user relentlessly');
      expect(skillMessage?.content).toContain('</skill-invoked>');

      // The environment message stayed catalog-free (no skills section).
      const environment = messages.find(
        (m) => m.role === 'user' && 'name' in m && m.name === 'Environment',
      ) as { content?: string } | undefined;
      expect(environment?.content).toBeDefined();
      expect(environment?.content).not.toContain('## Skills');
      expect(environment?.content).not.toContain('grill-me');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('works with no argument and with a skill that lacks disable-model-invocation', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-project-'));
    try {
      writeSkill(
        cwd,
        'code-review',
        '---\nname: code-review\ndescription: Review the code.\n---\nBe a strict reviewer.',
      );
      const { agent, cx, sessionId } = await setupAgent(cwd);

      mockedRunLlmStep.mockResolvedValueOnce(answerStep('ok'));
      const response = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/code-review' }] },
        cx,
      );
      expect(response.stopReason).toBe('end_turn');

      const messages = mockedRunLlmStep.mock.calls[0]?.[1]?.messages ?? [];
      const skillMessage = messages.find(
        (m) =>
          m.role === 'user' && typeof m.content === 'string' && m.content.includes('code-review'),
      );
      expect(skillMessage?.content).toContain('Be a strict reviewer.');
      expect(skillMessage?.content).not.toContain('<skill-argument>');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('treats unknown /names as ordinary prompts instead of failed commands', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-project-'));
    try {
      const { agent, cx, notifications, sessionId } = await setupAgent(cwd);
      mockedRunLlmStep.mockResolvedValueOnce(answerStep('looking into /not-a-skill for you'));
      const response = await agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/not-a-skill explain this' }] },
        cx,
      );
      // Forwarded to the model as a normal user message...
      expect(response.stopReason).toBe('end_turn');
      expect(mockedRunLlmStep).toHaveBeenCalledTimes(1);
      const firstCall = mockedRunLlmStep.mock.calls[0]?.[1] as {
        messages: Array<{ role: string; content: unknown }>;
      };
      const lastUser = [...firstCall.messages].reverse().find((m) => m.role === 'user');
      expect(lastUser?.content).toContain('/not-a-skill explain this');
      // ...not answered with an "unknown command" bubble.
      const text = notifications
        .filter((n) => n.update.sessionUpdate === 'agent_message_chunk')
        .map((n) => (n.update as { content?: { text?: string } }).content?.text ?? '')
        .join('\n');
      expect(text).not.toContain('Unknown slash command');

      // Builtins and installed skills still take the command path.
      writeSkill(cwd, 'grill-me', GRILL_ME);
      mockedRunLlmStep.mockClear();
      mockedRunLlmStep.mockResolvedValueOnce({
        text: '',
        reasoning: '',
        toolCalls: [],
        finishReason: 'stop',
        usage: null,
      });
      await agent.prompt({ sessionId, prompt: [{ type: 'text', text: '/grill-me' }] }, cx);
      const skillMessages = mockedRunLlmStep.mock.calls[0]?.[1]?.messages.filter(
        (m) => m.role === 'user' && typeof m.content === 'string',
      );
      const skillInvocation = skillMessages?.at(-1)?.content as string;
      expect(skillInvocation).toContain('<skill-invoked>');
      expect(skillInvocation).toContain('Interview the user relentlessly');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('keeps skill turns cancellable until the slash command finishes', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-project-'));
    try {
      writeSkill(cwd, 'grill-me', GRILL_ME);
      const { agent, cx, sessionId } = await setupAgent(cwd);
      let resolveStep!: (result: LlmStepResult) => void;
      mockedRunLlmStep.mockReturnValueOnce(
        new Promise<LlmStepResult>((resolve) => {
          resolveStep = resolve;
        }),
      );

      const prompt = agent.prompt(
        { sessionId, prompt: [{ type: 'text', text: '/grill-me my plan' }] },
        cx,
      );
      await vi.waitFor(() => expect(mockedRunLlmStep).toHaveBeenCalledTimes(1));

      const active = (
        agent as unknown as {
          sessions: Map<
            string,
            { abortController: AbortController | null; gracefulCancel: boolean }
          >;
        }
      ).sessions.get(sessionId)!;
      expect(active.abortController).not.toBeNull();
      agent.cancel({ sessionId });
      expect(active.gracefulCancel).toBe(true);

      resolveStep(answerStep('cancelled cleanly'));
      await expect(prompt).resolves.toMatchObject({ stopReason: 'cancelled' });
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });

  it('advertises installed skills in available_commands_update', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'zen-agent-skills-slash-project-'));
    try {
      writeSkill(cwd, 'grill-me', GRILL_ME);
      const { notifications } = await setupAgent(cwd);
      await new Promise((resolve) => setTimeout(resolve, 50));

      const update = notifications.find(
        (n) => n.update.sessionUpdate === 'available_commands_update',
      );
      expect(update).toBeDefined();
      const commands =
        (update?.update as { availableCommands?: Array<{ name: string }> })?.availableCommands ??
        [];
      const names = commands.map((c) => c.name);
      expect(names).toContain('prompt');
      expect(names).toContain('sandbox');
      expect(names).toContain('grill-me');
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
});
