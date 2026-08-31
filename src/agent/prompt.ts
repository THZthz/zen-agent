import * as acp from '@agentclientprotocol/sdk';
import type { Constructor, ZenAgentCore } from './core.js';
import type { TurnSurface } from './turn.js';
import type { AgentCommandsSurface } from './commands.js';
import { promptBlocksToPromptContent } from './prompt-content.js';
import { getUserMessageName } from '../session/system-prompt.js';
import { listSkills } from '../session/skills.js';
import { formatLlmError } from '../providers/llm-errors.js';
import { AVAILABLE_COMMANDS } from './config.js';
import type { NamedUserMessage, StoredSession, UserContentPart } from '../session/storage.js';

/**
 * ACP `session/prompt` entry point: turns a client prompt into either a
 * slash-command dispatch (handlers live in agent-commands.ts) or a user
 * message appended to the session history, then starts the turn. This file
 * owns the prompt parse/preprocessing, the known-command gate and the
 * controller ownership + persistence boundaries around all of it. Depends on
 * withTurnExecution for the actual turn loop (TurnSurface) and
 * withAgentCommands for the command handlers.
 */
/** Public ACP API contributed by withPromptExecution. */
export interface PromptExecutionSurface {
  prompt(params: acp.PromptRequest, cx: acp.AgentContext): Promise<acp.PromptResponse>;
}

export function withPromptExecution<
  T extends Constructor<ZenAgentCore & TurnSurface & AgentCommandsSurface>,
>(Base: T): T & Constructor<PromptExecutionSurface> {
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
  }
  return PromptExecution;
}
