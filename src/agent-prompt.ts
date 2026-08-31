import * as acp from '@agentclientprotocol/sdk';
import type { ActiveSession, Constructor, ZenAgentCore } from './agent-core.js';
import type { TurnSurface } from './agent-turn.js';
import { promptBlocksToPromptContent } from './prompt-content.js';
import {
  buildSystemPrompt,
  getUserMessageName,
  ENVIRONMENT_MESSAGE_NAME,
} from './system-prompt.js';
import { buildSkillInvocationPrompt, listSkills } from './skills.js';
import { formatLlmError } from './llm-errors.js';
import { AVAILABLE_COMMANDS } from './agent-config.js';
import type { NamedUserMessage, StoredSession, UserContentPart } from './storage.js';

/**
 * ACP `session/prompt` entry point and the built-in slash commands
 * (`/prompt`, `/sandbox`, `/tools`, `/skill-name`). Depends on
 * withTurnExecution for the actual turn loop (TurnSurface); everything here
 * is about turning a client prompt into either a slash-command response or a
 * user message appended to the session history before the turn starts.
 */
/** Public ACP API contributed by withPromptExecution. */
export interface PromptExecutionSurface {
  prompt(params: acp.PromptRequest, cx: acp.AgentContext): Promise<acp.PromptResponse>;
}

export function withPromptExecution<T extends Constructor<ZenAgentCore & TurnSurface>>(
  Base: T,
): T & Constructor<PromptExecutionSurface> {
  class PromptExecution extends Base {
    constructor(...args: any[]) {
      super(...args);
    }
    async prompt(params: acp.PromptRequest, cx: acp.AgentContext): Promise<acp.PromptResponse> {
      // Abort the currently running turn before queuing the replacement. The
      // shared operation queue then prevents preprocessing, history updates and
      // final saves from overlapping with any prompt/lifecycle/config operation.
      this.abortActiveSession(params.sessionId);
      return this.withSessionOperation(params.sessionId, () => this.promptSerialized(params, cx));
    }

    private async promptSerialized(
      params: acp.PromptRequest,
      cx: acp.AgentContext,
    ): Promise<acp.PromptResponse> {
      const active = this.sessions.get(params.sessionId);
      if (!active) {
        throw new Error(`Session ${params.sessionId} not found`);
      }

      this.clearGracefulCancel(active);
      const controller = new AbortController();
      active.abortController = controller;

      try {
        const { text: userText, parts } = await promptBlocksToPromptContent(params.prompt);
        void this.logRuntime(active.session.cwd, 'info', 'prompt received', {
          sessionId: params.sessionId,
          text: userText,
        });
        const slashCommand = this.parseSlashCommand(userText);

        // Gate on KNOWN command names (builtins + installed skills): any other
        // text that merely starts with "/" — e.g. "/etc/hosts permissions?" or
        // a regex like "/foo.*bar/" — is an ordinary prompt for the model, not
        // a failed command invocation.
        if (slashCommand && (await this.isKnownSlashCommand(active.session, slashCommand.name))) {
          active.session.events.push({
            sessionUpdate: 'user_message_chunk',
            content: { type: 'text', text: userText },
          });
          await this.save(active);

          // Keep controller ownership and the final persistence boundary alive
          // for the complete command, including skill-backed LLM turns.
          return await this.handleSlashCommand(active, cx, slashCommand);
        }

        const mediaParts = parts.filter((part) => part.type !== 'text');

        // Transcript events show the original blocks (Zed renders attached
        // images/audio in the thread), while the LLM message carries whatever
        // the active model can actually consume. Pure-text prompts keep the
        // single combined chunk shape used before media support.
        if (mediaParts.length === 0) {
          active.session.events.push({
            sessionUpdate: 'user_message_chunk',
            content: { type: 'text', text: userText },
          });
        } else {
          for (const part of parts) {
            const block: acp.ContentBlock =
              part.type === 'text'
                ? { type: 'text', text: part.text }
                : part.type === 'image'
                  ? {
                      type: 'image',
                      data: part.data,
                      mimeType: part.mimeType,
                      ...(part.uri ? { uri: part.uri } : {}),
                    }
                  : { type: 'audio', data: part.data, mimeType: part.mimeType };
            active.session.events.push({
              sessionUpdate: 'user_message_chunk',
              content: block,
            });
          }
        }

        let llmContent: string | UserContentPart[];
        if (mediaParts.length === 0) {
          // Plain-string shape stays byte-compatible with existing history
          // and the provider's prefix cache.
          llmContent = userText;
        } else {
          const modalities = await this.mediaModalities(active);
          llmContent = [];
          const rawText = parts
            .filter(
              (part): part is Extract<UserContentPart, { type: 'text' }> => part.type === 'text',
            )
            .map((part) => part.text)
            .join('\n\n');
          if (rawText.length > 0) {
            llmContent.push({ type: 'text', text: rawText });
          }
          for (const part of mediaParts) {
            if (
              (part.type === 'image' && !modalities.image) ||
              (part.type === 'audio' && !modalities.audio)
            ) {
              // The model cannot consume this modality; degrade to a note so
              // the turn still runs (and the user learns why).
              llmContent.push({
                type: 'text',
                text:
                  part.type === 'image'
                    ? `[image attached${part.uri ? `: ${part.uri}` : ''} (${part.mimeType}) omitted: current model does not support image input]`
                    : `[audio attached (${part.mimeType}) omitted: current model does not support audio input]`,
              });
              continue;
            }
            llmContent.push(part);
          }
        }

        const userMessage: NamedUserMessage = {
          role: 'user',
          content: llmContent,
          name: await getUserMessageName(active.session.cwd),
        };
        active.session.llmMessages.push(userMessage);
        await this.save(active);

        // IMPORTANT: must `await` here. `return promise` inside try/finally runs
        // the finally block IMMEDIATELY (the returned promise is adopted
        // asynchronously outside the function), which would null
        // active.abortController while the turn is still running and break
        // graceful cancel (cancel() would see no controller).
        const response = await this.runTurnTracked(active, cx, controller.signal);
        return response;
      } catch (error) {
        if (controller.signal.aborted) {
          void this.logRuntime(active.session.cwd, 'warn', 'prompt cancelled', {
            sessionId: params.sessionId,
          });
          return { stopReason: 'cancelled' };
        }
        // Classify API failures (401/402/429/5xx/context overflow) into
        // actionable guidance for the user; the original error is kept as the
        // thrown instance so its stack and cause survive.
        const formatted = await formatLlmError(error, {
          provider: active.session.config.provider,
        });
        void this.logRuntime(active.session.cwd, 'error', 'prompt failed', {
          sessionId: params.sessionId,
          error: formatted,
        });
        if (error instanceof Error && formatted !== error.message) {
          error.message = formatted;
        }
        throw error;
      } finally {
        if (active.abortController === controller) {
          active.abortController = null;
        }
        this.clearGracefulCancel(active);
        active.session.updatedAt = new Date().toISOString();
        await this.save(active).catch((error) => {
          void this.logRuntime(active.session.cwd, 'error', 'final session save failed', {
            sessionId: params.sessionId,
            error: error instanceof Error ? error.message : String(error),
          });
        });
      }
    }

    private parseSlashCommand(text: string): {
      name: string;
      argument: string;
    } | null {
      const trimmed = text.trim();
      if (!trimmed.startsWith('/')) {
        return null;
      }
      const match = trimmed.match(/^\/(\S+)\s*([\s\S]*)$/);
      if (!match) {
        return { name: trimmed.slice(1).trim(), argument: '' };
      }
      return {
        name: match[1].toLowerCase(),
        argument: match[2].trim(),
      };
    }

    /**
     * Whether `name` resolves to a builtin command or an installed Agent Skill;
     * gates slash-command handling so unknown "/words" stay normal prompts.
     */
    private async isKnownSlashCommand(session: StoredSession, name: string): Promise<boolean> {
      if (AVAILABLE_COMMANDS.some((command) => command.name === name)) {
        return true;
      }
      const skills = await listSkills(session.cwd);
      return skills.some((skill) => skill.name.toLowerCase() === name);
    }

    private async handleSlashCommand(
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
  return PromptExecution;
}
