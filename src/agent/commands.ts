import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './core.js';
import type { TurnSurface } from './turn.js';
import { ENVIRONMENT_MESSAGE_NAME } from '../session/system-prompt.js';
import { buildSystemPrompt } from '../session/system-prompt.js';
import { buildSkillInvocationPrompt, listSkills } from '../session/skills.js';

/**
 * Built-in slash-command handlers (`/prompt`, `/sandbox`, `/writable`, `/tools` and
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
        case 'writable':
          return {
            stopReason: await this.handleWritableSlashCommand(active, cx, command.argument),
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
     * `/writable` manages the paths the bash sandbox keeps writable. The
     * sandbox is deny-by-default: the whole rootfs is mounted read-only and
     * only these paths are bind-mounted read-write. With no argument (or
     * `status`) it prints the current list; `add <path>[,<path>...]` appends
     * paths (trimmed, deduped, empty entries dropped); `del <path>[,<path>...]`
     * removes them (missing entries are ignored); `clear` empties the list.
     * The list is persisted per session in `config.writablePaths` and applied
     * as `--bind <path> <path>` inside the sandbox.
     */
    private async handleWritableSlashCommand(
      active: ActiveSession,
      cx: acp.AgentContext,
      argument: string,
    ): Promise<acp.StopReason> {
      const usage =
        'Usage: /writable [add <path>[,<path>...] | del <path>[,<path>...] | clear]';

      if (argument.trim() === '' || argument.trim().toLowerCase() === 'status') {
        const paths = active.session.config.writablePaths;
        const source =
          paths.length > 0 ? paths.join(', ') : 'no paths (everything is read-only)';
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Writable paths: ${source}\n${usage}`,
          },
        });
        return 'end_turn';
      }

      const [verbRaw, ...rest] = argument.trim().split(/\s+/);
      const verb = verbRaw!.toLowerCase();
      const paths = rest
        .join(' ')
        .split(',')
        .map((p) => p.trim())
        .filter(Boolean);

      if (verb === 'clear') {
        active.session.config.writablePaths = [];
        await this.save(active);
        void this.logRuntime(active.session.cwd, 'info', 'writable paths cleared', {
          sessionId: active.session.sessionId,
        });
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: 'Writable paths cleared for this session (everything is read-only).',
          },
        });
        return 'end_turn';
      }

      if (verb === 'add') {
        if (paths.length === 0) {
          await this.emit(active, cx, {
            sessionUpdate: 'agent_message_chunk',
            content: {
              type: 'text',
              text: `No valid paths in: ${argument}\n${usage}`,
            },
          });
          return 'end_turn';
        }
        const existing = active.session.config.writablePaths;
        for (const p of paths) {
          if (!existing.includes(p)) existing.push(p);
        }
        await this.save(active);
        void this.logRuntime(active.session.cwd, 'info', 'writable paths added', {
          sessionId: active.session.sessionId,
          paths,
        });
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text: `Writable paths added: ${paths.join(', ')}\nWritable paths: ${existing.join(', ')}`,
          },
        });
        return 'end_turn';
      }

      if (verb === 'del') {
        const existing = active.session.config.writablePaths;
        const removed = paths.filter((p) => existing.includes(p));
        active.session.config.writablePaths = existing.filter((p) => !paths.includes(p));
        if (removed.length > 0) {
          await this.save(active);
          void this.logRuntime(active.session.cwd, 'info', 'writable paths removed', {
            sessionId: active.session.sessionId,
            paths: removed,
          });
        }
        await this.emit(active, cx, {
          sessionUpdate: 'agent_message_chunk',
          content: {
            type: 'text',
            text:
              removed.length > 0
                ? `Writable paths removed: ${removed.join(', ')}`
                : `No writable paths matched: ${paths.join(', ')}`,
          },
        });
        return 'end_turn';
      }

      await this.emit(active, cx, {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: `Unknown /writable argument: ${argument}\n${usage}`,
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
