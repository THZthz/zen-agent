import * as acp from '@agentclientprotocol/sdk';
import { isAbsolute } from 'node:path';
import {
  createStoredSession,
  deleteStoredSession,
  readStoredSession,
  validateSessionId,
  writeSession,
  type StoredSession,
  type ThinkingEffort,
} from './storage.js';
import { findSessionCwd, listStoredSessions } from './session-index.js';
import { getDefaultModel } from './provider.js';
import {
  buildEnvironmentMessage,
  buildSessionContinuedMessage,
  ENVIRONMENT_MESSAGE_NAME,
  isEnvironmentMessage,
} from './system-prompt.js';
import { prepareReplayEvents, coalesceReplayEvents } from './replay.js';
import {
  AVAILABLE_COMMANDS,
  GRACEFUL_CANCEL_TIMEOUT_MS,
  modelConfigOption,
  providerConfigOption,
  thinkingConfigOption,
  THINKING_EFFORT_VALUES,
} from './agent-config.js';
import {
  getDefaultProviderId,
  isKnownProvider,
  requireProviderDefinition,
} from './provider-registry.js';
import { listSkills } from './skills.js';
import type { Constructor, ZenAgentCore } from './agent-core.js';

/**
 * ACP session lifecycle: create/load/resume/list/delete/close, config
 * options, graceful cancel and the available-command advertising that goes
 * with session creation. Everything here only touches ZenAgentCore state and
 * helpers — it never runs an LLM turn.
 */
/** Public ACP session-lifecycle API contributed by withSessionManagement. */
export interface SessionManagementSurface {
  newSession(params: acp.NewSessionRequest, cx: acp.AgentContext): Promise<acp.NewSessionResponse>;
  loadSession(
    params: acp.LoadSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.LoadSessionResponse>;
  resumeSession(
    params: acp.ResumeSessionRequest,
    cx: acp.AgentContext,
  ): Promise<acp.ResumeSessionResponse>;
  listSessions(params: acp.ListSessionsRequest): Promise<acp.ListSessionsResponse>;
  deleteSession(params: acp.DeleteSessionRequest): Promise<acp.DeleteSessionResponse>;
  closeSession(params: acp.CloseSessionRequest): Promise<acp.CloseSessionResponse>;
  setSessionConfigOption(
    params: acp.SetSessionConfigOptionRequest,
  ): Promise<acp.SetSessionConfigOptionResponse>;
  cancel(params: acp.CancelNotification): void;
}

export function withSessionManagement<T extends Constructor<ZenAgentCore>>(
  Base: T,
): T & Constructor<SessionManagementSurface> {
  class SessionManagement extends Base {
    constructor(...args: any[]) {
      super(...args);
    }

    /** Abort, hide and settle any active turn before replacing/removing it. */
    private async retireActiveSession(sessionId: string): Promise<void> {
      const previous = this.sessions.get(sessionId);
      this.abortActiveSession(sessionId);
      if (previous && this.sessions.get(sessionId) === previous) {
        this.sessions.delete(sessionId);
      }
      await this.settlePreviousTurn(previous);
    }

    async newSession(
      params: acp.NewSessionRequest,
      cx: acp.AgentContext,
    ): Promise<acp.NewSessionResponse> {
      if (!isAbsolute(params.cwd)) {
        throw new Error('cwd must be an absolute path');
      }
      const session = await createStoredSession(params.cwd, getDefaultProviderId());
      // Freeze the environment snapshot into the persisted conversation at
      // session creation. It sits right after the system prompt, so it must
      // stay byte-identical for the provider's context cache to keep hitting
      // across steps and restarts (a per-request regenerated message — e.g.
      // with a changing git status — would break the whole cached prefix).
      // With /tools off the model has no tools to act on the environment, so
      // the snapshot is omitted entirely (chat-only session).
      if (session.config.toolsEnabled !== false) {
        session.llmMessages.push({
          role: 'user',
          name: ENVIRONMENT_MESSAGE_NAME,
          content: await buildEnvironmentMessage(session),
        });
      }
      await writeSession(session);
      this.sessions.set(session.sessionId, this.makeActiveSession(session));
      void this.logRuntime(params.cwd, 'info', 'session created', {
        sessionId: session.sessionId,
      });
      this.scheduleAvailableCommands(session.sessionId, cx);
      return {
        sessionId: session.sessionId,
        configOptions: await this.getConfigOptions(session),
      };
    }

    async loadSession(
      params: acp.LoadSessionRequest,
      cx: acp.AgentContext,
    ): Promise<acp.LoadSessionResponse> {
      this.abortActiveSession(params.sessionId);
      return this.withSessionOperation(params.sessionId, async () => {
        // The previous queued operation has fully settled before this read. The
        // explicit retirement also covers legacy/directly injected active turns.
        await this.retireActiveSession(params.sessionId);
        const { session, droppedEntries } = await readStoredSession(params.cwd, params.sessionId);
        await this.prepareResumedSession(session);
        this.sessions.set(params.sessionId, this.makeActiveSession(session));
        void this.logRuntime(params.cwd, 'info', 'session loaded', {
          sessionId: session.sessionId,
          droppedEntries,
        });

        const replayEvents = coalesceReplayEvents(prepareReplayEvents(session.events, session.cwd));
        for (const update of replayEvents) {
          await cx.notify(acp.methods.client.session.update, {
            sessionId: session.sessionId,
            update,
          });
        }
        this.scheduleAvailableCommands(session.sessionId, cx);

        return {
          configOptions: await this.getConfigOptions(session),
        };
      });
    }

    async resumeSession(
      params: acp.ResumeSessionRequest,
      cx: acp.AgentContext,
    ): Promise<acp.ResumeSessionResponse> {
      this.abortActiveSession(params.sessionId);
      return this.withSessionOperation(params.sessionId, async () => {
        // See loadSession: the old operation's final save precedes this read.
        await this.retireActiveSession(params.sessionId);
        const { session, droppedEntries } = await readStoredSession(params.cwd, params.sessionId);
        await this.prepareResumedSession(session);
        this.sessions.set(params.sessionId, this.makeActiveSession(session));
        void this.logRuntime(params.cwd, 'info', 'session resumed', {
          sessionId: session.sessionId,
          droppedEntries,
        });
        this.scheduleAvailableCommands(session.sessionId, cx);
        return {
          configOptions: await this.getConfigOptions(session),
        };
      });
    }

    async listSessions(params: acp.ListSessionsRequest): Promise<acp.ListSessionsResponse> {
      const sessions = await listStoredSessions(params.cwd ?? undefined);
      return { sessions };
    }

    async deleteSession(params: acp.DeleteSessionRequest): Promise<acp.DeleteSessionResponse> {
      this.abortActiveSession(params.sessionId);
      return this.withSessionOperation(params.sessionId, async () => {
        const active = this.sessions.get(params.sessionId);
        const cwd = active?.session.cwd ?? (await findSessionCwd(params.sessionId));
        if (!cwd) {
          throw new Error(`Session ${params.sessionId} not found`);
        }
        // The queued prompt has fully settled, so deletion cannot be followed
        // by a late final save that recreates state.json.
        await this.retireActiveSession(params.sessionId);
        await deleteStoredSession(cwd, params.sessionId);
        void this.logRuntime(cwd, 'info', 'session deleted', {
          sessionId: params.sessionId,
        });
        return {};
      });
    }

    async closeSession(params: acp.CloseSessionRequest): Promise<acp.CloseSessionResponse> {
      this.abortActiveSession(params.sessionId);
      return this.withSessionOperation(params.sessionId, async () => {
        await this.retireActiveSession(params.sessionId);
        return {};
      });
    }

    async setSessionConfigOption(
      params: acp.SetSessionConfigOptionRequest,
    ): Promise<acp.SetSessionConfigOptionResponse> {
      validateSessionId(params.sessionId);
      return this.withSessionOperation(params.sessionId, () =>
        this.setSessionConfigOptionSerialized(params),
      );
    }

    private async setSessionConfigOptionSerialized(
      params: acp.SetSessionConfigOptionRequest,
    ): Promise<acp.SetSessionConfigOptionResponse> {
      const active = this.sessions.get(params.sessionId);
      if (!active) {
        throw new Error(`Session ${params.sessionId} not found`);
      }

      const value = String(params.value);

      // Provider, model and thinking effort are fixed once the user sent their
      // first message: changing them mid-conversation would mix model
      // behaviors and billing currencies within one thread.
      if (this.sessionHasStarted(active.session)) {
        throw new Error(
          `${params.configId} cannot be changed after the first message of a session`,
        );
      }

      switch (params.configId) {
        case 'provider': {
          if (!isKnownProvider(value)) {
            throw new Error(
              `Unknown provider: ${value}. Configure it via ZEN_AGENT_PROVIDERS or ZEN_AGENT_PROVIDERS_FILE.`,
            );
          }
          if (value !== active.session.config.provider) {
            active.session.config.provider = value;
            // The previous provider's model likely does not exist on the new
            // one; reset to the provider's default so the selector stays valid.
            active.session.config.model = getDefaultModel(value);
          }
          break;
        }
        case 'model': {
          const def = requireProviderDefinition(active.session.config.provider);
          if (def.discovery.enabled) {
            // Discovery providers accept any model slug; only the fetched
            // catalog is offered in the selector.
            if (value.trim().length === 0) {
              throw new Error('Model must not be empty');
            }
          } else if (!def.staticModels.some((opt) => opt.value === value)) {
            throw new Error(`Unknown model: ${value}`);
          }
          active.session.config.model = value;
          break;
        }
        case 'thinking_effort': {
          if (!THINKING_EFFORT_VALUES.includes(value as ThinkingEffort)) {
            throw new Error(`Unknown thinking effort: ${value}`);
          }
          active.session.config.thinkingEffort = value as ThinkingEffort;
          break;
        }
        default:
          throw new Error(`Unknown config option: ${params.configId}`);
      }

      await this.save(active);
      return { configOptions: await this.getConfigOptions(active.session) };
    }

    /**
     * Graceful cancel (ACP `session/cancel`).
     *
     * Zed sends this notification for ALL interruption paths — the Stop button
     * (`thread_view.rs::cancel_generation`), force-send while generating
     * (`thread_view.rs::stop_current_and_send_new_message`) and "steer" queued
     * messages (`thread_view.rs::dispatch_queued_entry`) — and it carries no
     * reason field, so they are indistinguishable here.
     *
     * We therefore never hard-abort on cancel: the running turn keeps executing
     * until the current LLM step (thinking/answering) or bash tool call
     * completes, then responds `stopReason: "cancelled"`. Zed awaits that
     * response before delivering the follow-up message, which is what makes the
     * new message appear after the interrupted unit of work rather than
     * mid-way through it.
     *
     * A hard abort still happens on `session/close` / `session/delete`, and
     * optionally after `ZEN_AGENT_GRACEFUL_CANCEL_TIMEOUT_MS` as an escape
     * hatch for runaway commands.
     */
    cancel(params: acp.CancelNotification): void {
      const active = this.sessions.get(params.sessionId);
      if (!active || active.gracefulCancel) {
        return;
      }
      if (!active.abortController) {
        // No turn is running; nothing to stop. The flag is not set so a later
        // prompt starts fresh.
        return;
      }
      active.gracefulCancel = true;
      void this.logRuntime(active.session.cwd, 'info', 'graceful cancel requested', {
        sessionId: params.sessionId,
      });
      if (GRACEFUL_CANCEL_TIMEOUT_MS > 0) {
        active.cancelTimer = setTimeout(() => {
          active.cancelTimer = null;
          void this.logRuntime(
            active.session.cwd,
            'warn',
            'graceful cancel timed out; hard abort',
            {
              sessionId: params.sessionId,
              timeoutMs: GRACEFUL_CANCEL_TIMEOUT_MS,
            },
          );
          active.abortController?.abort();
        }, GRACEFUL_CANCEL_TIMEOUT_MS);
        active.cancelTimer.unref?.();
      }
    }

    /**
     * Prepare a stored session for continuation after a restart.
     *
     * 1. Backfill: sessions created before environment messages existed have
     *    none in their history; prepend a frozen snapshot so the cache prefix
     *    is stable (system + environment + history).
     * 2. Notify: append a fresh environment notification at the END of the
     *    conversation so the model knows the session was continued. Being
     *    appended after all history, it does not disturb the cached prefix —
     *    the system prompt, frozen environment message and persisted history
     *    stay byte-identical, so the provider's context cache keeps hitting.
     */
    private async prepareResumedSession(session: StoredSession): Promise<void> {
      // With /tools off the session is chat-only: no environment snapshot is
      // injected (see newSession), so there is nothing to backfill and no
      // continuation notice to append either. Any environment message left in
      // the history by an older build (or by a tools-on creation before the
      // toggle) is stripped so the invariant holds across restarts.
      if (session.config.toolsEnabled !== false) {
        if (session.llmMessages.length === 0 || !isEnvironmentMessage(session.llmMessages[0]!)) {
          session.llmMessages.unshift({
            role: 'user',
            name: ENVIRONMENT_MESSAGE_NAME,
            content: await buildEnvironmentMessage(session),
          });
        }
        session.llmMessages.push({
          role: 'user',
          name: ENVIRONMENT_MESSAGE_NAME,
          content: await buildSessionContinuedMessage(session),
        });
      } else {
        this.removeEnvironmentMessages(session);
      }
      await writeSession(session);
    }

    private async getConfigOptions(session: StoredSession): Promise<acp.SessionConfigOption[]> {
      return [
        {
          ...providerConfigOption(),
          currentValue: session.config.provider,
        } as acp.SessionConfigOption,
        {
          ...(await modelConfigOption(session.config.provider)),
          currentValue: session.config.model,
        } as acp.SessionConfigOption,
        (await thinkingConfigOption(session)) as acp.SessionConfigOption,
      ];
    }

    private scheduleAvailableCommands(sessionId: string, cx: acp.AgentContext): void {
      setTimeout(() => {
        void this.sendAvailableCommands(sessionId, cx).catch((error) => {
          console.error('Failed to send available commands:', error);
        });
      }, 0);
    }

    private async sendAvailableCommands(sessionId: string, cx: acp.AgentContext): Promise<void> {
      const active = this.sessions.get(sessionId);
      const availableCommands = active
        ? await this.buildAvailableCommands(active.session)
        : AVAILABLE_COMMANDS;
      await cx.notify(acp.methods.client.session.update, {
        sessionId,
        update: {
          sessionUpdate: 'available_commands_update',
          availableCommands,
        },
      });
    }

    /**
     * Built-in slash commands plus one `/skill-name` command per installed
     * Agent Skill (discovered live from the session cwd, independent of the
     * `ZEN_AGENT_SHOW_SKILLS_CATALOG` environment flag).
     */
    private async buildAvailableCommands(session: StoredSession): Promise<acp.AvailableCommand[]> {
      const commands: acp.AvailableCommand[] = [...AVAILABLE_COMMANDS];
      for (const skill of await listSkills(session.cwd)) {
        commands.push({
          name: skill.name,
          description: skill.description || `Run the "${skill.name}" skill (installed Agent Skill)`,
          input: {
            hint: skill.description || `what to pass to the ${skill.name} skill`,
          },
        });
      }
      return commands;
    }
  }
  return SessionManagement;
}
