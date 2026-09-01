import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './core.js';
import type { TurnSurface } from './turn.js';
import { ENVIRONMENT_MESSAGE_NAME } from '../session/system-prompt.js';
import { getReadOnlyBindPaths } from '../tools/sandbox.js';
import { buildSystemPrompt } from '../session/system-prompt.js';
import { buildSkillInvocationPrompt, listSkills } from '../session/skills.js';

/**
 * Built-in slash-command handlers (`/prompt`, `/sandbox`, `/robind`, `/tools` and
 * installed-skill invocation) — see the ownership map in agent.ts. The
 * prompt entry point (agent-prompt.ts) owns parsing, the known-command gate
 * and every persistence boundary around a command; this module owns what a
 * command *does*: its user-visible replies, config mutations and, for
 * skills, the injected turn.
 */

/** Command-handling surface the prompt mixin relies on. */
export interface AgentCommandsSurface {
  handleSlashCommand(
    active: ActiveSession,
    cx: acp.AgentContext,
    command: { name: string; argument: string },
  ): Promise<acp.PromptResponse>;
}

export function withAgentCommands<T extends Constructor<ZenAgentCore & TurnSurface>>(
  Base: T,
): T & Constructor<AgentCommandsSurface> {
  class AgentCommands extends Base {
    constructor(...args: any[]) {
      super(...args);
    }

    async handleSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      command: { name: string; argument: string },
    ): Promise<acp.PromptResponse> {
      switch (command.name) {
        case 'prompt':
          return {
            stopReason: await this.handlePromptSlashCommand(active, cx, command.argument),
          };
        case 'sandbox':
          return {
            stopReason: await this.handleSandboxSlashCommand(active, cx, command.argument),
          };
        case 'robind':
          return {
            stopReason: await this.handleROBindSlashCommand(active, cx, command.argument),
          };
        case 'tools':
          return {
            stopReason: await this.handleToolsSlashCommand(active, cx, command.argument),
          };
        default: {
          const skillStop = await this.handleSkillSlashCommand(active, cx, command);
          if (skillStop) {
            return skillStop;
          }
          await this.emit(active, cx, {
            sessionUpdate: 'agent_message_chunk',
            content: {
              type: 'text',
              text: `Unknown slash command: /${command.name}`,
            },
          });
          return { stopReason: 'end_turn' };
        }
      }
    }

    /**
     * `/skill-name <argument>` invokes an installed Agent Skill: the skill's
     * `SKILL.md` is read and injected as the user message that starts the
     * next model turn, so the model follows the skill's instructions with its
     * bash tool. Works for every installed skill, regardless of
     * `ZEN_AGENT_SHOW_SKILLS_CATALOG` (that flag only controls the frozen
     * environment catalog, not slash commands). Returns null when no installed
     * skill matches the command name.
     */
    private async handleSkillSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      command: { name: string; argument: string },
    ): Promise<acp.PromptResponse | null> {
      const skills = await listSkills(active.session.cwd);
      const skill = skills.find((s) => s.name.toLowerCase() === command.name);
      if (!skill) {
        return null;
      }

      active.session.llmMessages.push({
        role: 'user',
        content: await buildSkillInvocationPrompt(skill, command.argument),
        name: ENVIRONMENT_MESSAGE_NAME,
      });
      await this.save(active);
      void this.logRuntime(active.session.cwd, 'info', 'skill invoked via slash command', {
        sessionId: active.session.sessionId,
        skill: skill.name,
        scope: skill.scope,
        disableModelInvocation: skill.disableModelInvocation,
      });

      const signal = active.abortController?.signal ?? new AbortController().signal;
      return this.runTurnTracked(active, cx, signal);
    }

    private async handlePromptSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      argument: string,
    ): Promise<acp.StopReason> {
      if (!argument) {
        // No argument = status: print the current system prompt. Always allowed.
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: buildSystemPrompt(active.session),
          },
        });
        return 'end_turn';
      }

      // The system prompt is part of the provider's cache prefix: once the
      // conversation has a real user message, changing it would break every
      // cache hit. Lock the set form like provider/model/thinking_effort.
      if (this.sessionHasStarted(active.session)) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: 'System prompt cannot be changed after the first message of a session.',
          },
        });
        return 'end_turn';
      }

      active.session.config.systemPrompt = argument;
      await this.save(active);
      void this.logRuntime(active.session.cwd, 'info', 'system prompt updated', {
        sessionId: active.session.sessionId,
      });

      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: 'System prompt updated for this session.',
        },
      });

      return 'end_turn';
    }

    /**
     * `/sandbox` toggles bwrap wrapping of bash tool calls for this session.
     *
     * Unlike `ZEN_AGENT_SANDBOX=1` (a global env policy set at startup), the
     * slash command changes the per-session `config.sandbox` flag at runtime
     * and persists it across restarts. The env policy still applies on top:
     * when `ZEN_AGENT_SANDBOX=1` the sandbox cannot be turned off per session.
     */
    private async handleSandboxSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      argument: string,
    ): Promise<acp.StopReason> {
      const normalized = argument.trim().toLowerCase();
      const envForced = process.env.ZEN_AGENT_SANDBOX === '1';
      const effective = this.sessionSandboxEnabled(active.session);

      if (normalized === '' || normalized === 'status') {
        const via =
          envForced && !active.session.config.sandbox
            ? ' (enforced by ZEN_AGENT_SANDBOX=1)'
            : active.session.config.sandbox
              ? ' (session)'
              : ' (off)';
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text:
              `Bash tool sandbox: ${effective ? 'ON' : 'OFF'}${via}\n` + 'Usage: /sandbox on | off',
          },
        });
        return 'end_turn';
      }

      const enabled =
        normalized === 'on' || normalized === '1' || normalized === 'true' || normalized === 'yes';
      const disabled =
        normalized === 'off' || normalized === '0' || normalized === 'false' || normalized === 'no';

      if (!enabled && !disabled) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Unknown /sandbox argument: ${argument}
Usage: /sandbox on | off`,
          },
        });
        return 'end_turn';
      }

      if (envForced) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: enabled
              ? 'Bash tool sandbox is already ON (enforced by ZEN_AGENT_SANDBOX=1).'
              : 'Cannot disable: ZEN_AGENT_SANDBOX=1 forces the bash tool sandbox on.',
          },
        });
        return 'end_turn';
      }

      active.session.config.sandbox = enabled;
      await this.save(active);
      void this.logRuntime(active.session.cwd, 'info', 'bash sandbox toggled', {
        sessionId: active.session.sessionId,
        sandbox: enabled,
      });

      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: enabled
            ? 'Bash tool calls are now sandboxed with bubblewrap for this session.'
            : 'Bash tool sandbox disabled for this session.',
        },
      });

      return 'end_turn';
    }

    /**
     * `/robind` toggles whether the additional read-only bind mounts from
     * `ZEN_AGENT_SANDBOX_RO_BIND` are applied inside the bash sandbox for
     * this session.
     *
     * The environment variable is only the path source: this command never
     * changes its value, it toggles the per-session `config.roBindEnabled`
     * flag at runtime and persists it across restarts. When
     * `ZEN_AGENT_SANDBOX_RO_BIND` is empty there is nothing to bind, so any
     * on|off attempt is refused with a warning and the flag is left
     * untouched; status stays available.
     */
    private async handleROBindSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      argument: string,
    ): Promise<acp.StopReason> {
      const normalized = argument.trim().toLowerCase();
      const configuredPaths = getReadOnlyBindPaths();
      const configured = configuredPaths.length > 0;

      if (normalized === '' || normalized === 'status') {
        const state = active.session.config.roBindEnabled ? 'ON' : 'OFF';
        const source = configured
          ? configuredPaths.join(', ')
          : 'not configured (ZEN_AGENT_SANDBOX_RO_BIND is empty)';
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Read-only binds: ${state} (${source})\n` + 'Usage: /robind on | off',
          },
        });
        return 'end_turn';
      }

      const enabled =
        normalized === 'on' || normalized === '1' || normalized === 'true' || normalized === 'yes';
      const disabled =
        normalized === 'off' || normalized === '0' || normalized === 'false' || normalized === 'no';

      if (!enabled && !disabled) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Unknown /robind argument: ${argument}\n` + 'Usage: /robind on | off',
          },
        });
        return 'end_turn';
      }

      if (!configured) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text:
              'ZEN_AGENT_SANDBOX_RO_BIND is empty; nothing to toggle. ' +
              'Set it to a comma-separated list of paths and try again.',
          },
        });
        return 'end_turn';
      }

      active.session.config.roBindEnabled = enabled;
      await this.save(active);
      void this.logRuntime(active.session.cwd, 'info', 'read-only binds toggled', {
        sessionId: active.session.sessionId,
        roBindEnabled: enabled,
        paths: configuredPaths,
      });

      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: enabled
            ? `Read-only bind mounts enabled for this session: ${configuredPaths.join(', ')}`
            : 'Read-only bind mounts disabled for this session.',
        },
      });

      return 'end_turn';
    }

    /**
     * `/tools` toggles every tool (bash + read_media) for this session.
     *
     * The flag lives on the per-session `config.toolsEnabled` and persists in
     * `state.json`, so a resumed session keeps its choice across restarts.
     * With tools off, `sessionToolSchemas` sends no tool schemas to the model
     * and `executeLlmToolCall` refuses any tool call that still arrives. The
     * session becomes chat-only: every environment snapshot/continuation is
     * dropped from the history, and nothing is injected on load/resume.
     * `on|off` is locked after the first user message (the tool list and
     * environment snapshot are part of the cache prefix); status stays open.
     */
    private async handleToolsSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      argument: string,
    ): Promise<acp.StopReason> {
      const normalized = argument.trim().toLowerCase();

      if (normalized === '' || normalized === 'status') {
        const state = active.session.config.toolsEnabled ? 'ON' : 'OFF';
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Tools (bash, read_media): ${state}
Usage: /tools on | off`,
          },
        });
        return 'end_turn';
      }

      const enabled =
        normalized === 'on' || normalized === '1' || normalized === 'true' || normalized === 'yes';
      const disabled =
        normalized === 'off' || normalized === '0' || normalized === 'false' || normalized === 'no';

      if (!enabled && !disabled) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Unknown /tools argument: ${argument}
Usage: /tools on | off`,
          },
        });
        return 'end_turn';
      }

      // The tool list and environment snapshot are part of the provider's
      // cache prefix: once the conversation has a real user message, toggling
      // them would break every cache hit. Lock on|off like
      // provider/model/thinking_effort; status stays open.
      if (this.sessionHasStarted(active.session)) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: 'Tools cannot be changed after the first message of a session.',
          },
        });
        return 'end_turn';
      }

      if (enabled === active.session.config.toolsEnabled) {
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: enabled
              ? 'Tools are already enabled for this session.'
              : 'Tools are already disabled for this session.',
          },
        });
        return 'end_turn';
      }

      active.session.config.toolsEnabled = enabled;
      if (enabled) {
        // Chat-only -> tools-on: restore the frozen environment snapshot so
        // the model has the working directory / git state to act on.
        await this.ensureEnvironmentMessage(active.session);
      } else {
        // Tools-on -> chat-only: drop every environment snapshot/continuation
        // so the conversation really is environment-free.
        this.removeEnvironmentMessages(active.session);
      }
      await this.save(active);
      void this.logRuntime(active.session.cwd, 'info', 'tools toggled', {
        sessionId: active.session.sessionId,
        toolsEnabled: enabled,
      });

      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: enabled
            ? 'Tools (bash, read_media) are now enabled for this session.'
            : 'Tools (bash, read_media) are now disabled for this session.',
        },
      });

      return 'end_turn';
    }
  }
  return AgentCommands;
}
