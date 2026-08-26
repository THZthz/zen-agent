import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  createStoredSession,
  readStoredSession,
  sessionPath,
  type StoredSession,
} from './storage.js';

describe('readStoredSession validation', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-validate-'));
    const dataHome = join(cwd, 'xdg');
    process.env.XDG_DATA_HOME = dataHome;
  });

  afterEach(() => {
    delete process.env.XDG_DATA_HOME;
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  async function writeState(sessionId: string, contents: string): Promise<void> {
    mkdirSync(join(cwd, '.sessions', sessionId), { recursive: true });
    writeFileSync(sessionPath(cwd, sessionId), contents, 'utf8');
  }

  it("rejects invalid JSON with a clean 'corrupted' error", async () => {
    await writeState('sess_bad', '{"sessionId": "sess_bad", "cwd":');
    await expect(readStoredSession(cwd, 'sess_bad')).rejects.toThrow(/corrupted: not valid JSON/);
  });

  it('rejects non-object and mismatched-identity files with clean errors', async () => {
    await writeState('sess_arr', '[1,2,3]');
    await expect(readStoredSession(cwd, 'sess_arr')).rejects.toThrow(/not a session object/);

    await writeState('sess_id', JSON.stringify({ sessionId: 'sess_other', cwd }));
    await expect(readStoredSession(cwd, 'sess_id')).rejects.toThrow(/invalid sessionId/);

    await writeState('sess_cwd', JSON.stringify({ sessionId: 'sess_cwd', cwd: '/elsewhere' }));
    await expect(readStoredSession(cwd, 'sess_cwd')).rejects.toThrow(/belongs to/);
  });

  it('backfills missing fields from older or damaged files instead of throwing', async () => {
    // A pre-providers, pre-sandbox legacy file with partial usage.
    await writeState(
      'sess_legacy',
      JSON.stringify({
        sessionId: 'sess_legacy',
        cwd,
        createdAt: '2026-01-01T00:00:00.000Z',
        llmMessages: [{ role: 'user', content: 'hi' }],
        config: { systemPrompt: '' },
        usage: { turns: 3, inputTokens: 100, costYuan: 3.5 },
      }),
    );
    const session = await readStoredSession(cwd, 'sess_legacy');
    expect(session.config.provider).toBe('deepseek');
    expect(session.config.model).toBe('deepseek-v4-flash');
    expect(session.config.thinkingEffort).toBe('off');
    expect(session.config.sandbox).toBe(false);
    // Older sessions predate the tools flag; absent means enabled.
    expect(session.config.toolsEnabled).toBe(true);
    expect(session.events).toEqual([]);
    expect(session.turnStats).toEqual([]);
    expect(session.title).toBeNull();
    expect(session.usage.turns).toBe(3);
    expect(session.usage.inputTokens).toBe(100);
    expect(session.usage.cost).toBe(3.5); // legacy costYuan key mapped
    expect(typeof session.updatedAt).toBe('string');

    // The normalized session round-trips through the real writer.
    await import('./storage.js').then(({ writeSession }) => writeSession(session));
    const reloaded = await readStoredSession(cwd, 'sess_legacy');
    expect(reloaded.config.sandbox).toBe(false);
    expect(reloaded.config.toolsEnabled).toBe(true);
  });

  it('keeps a disabled tools flag (toolsEnabled: false) across a round trip', async () => {
    await writeState(
      'sess_tools_off',
      JSON.stringify({
        sessionId: 'sess_tools_off',
        cwd,
        createdAt: '2026-08-20T00:00:00.000Z',
        llmMessages: [],
        config: { systemPrompt: '', sandbox: false, toolsEnabled: false },
        usage: {},
      }),
    );
    const session = await readStoredSession(cwd, 'sess_tools_off');
    expect(session.config.toolsEnabled).toBe(false);

    await import('./storage.js').then(({ writeSession }) => writeSession(session));
    const reloaded = await readStoredSession(cwd, 'sess_tools_off');
    expect(reloaded.config.toolsEnabled).toBe(false);
  });

  it('accepts the full thinking-effort vocabulary and rejects unknown values', async () => {
    await writeState(
      'sess_xhigh',
      JSON.stringify({
        sessionId: 'sess_xhigh',
        cwd,
        createdAt: '2026-01-01T00:00:00.000Z',
        llmMessages: [{ role: 'user', content: 'hi' }],
        config: { systemPrompt: '', thinkingEffort: 'xhigh' },
      }),
    );
    const session = await readStoredSession(cwd, 'sess_xhigh');
    expect(session.config.thinkingEffort).toBe('xhigh');

    await writeState(
      'sess_bad_effort',
      JSON.stringify({
        sessionId: 'sess_bad_effort',
        cwd,
        createdAt: '2026-01-01T00:00:00.000Z',
        llmMessages: [{ role: 'user', content: 'hi' }],
        config: { systemPrompt: '', thinkingEffort: 'turbo' },
      }),
    );
    const bad = await readStoredSession(cwd, 'sess_bad_effort');
    expect(bad.config.thinkingEffort).toBe('off');
  });

  it('keeps loading healthy sessions unchanged', async () => {
    const session: StoredSession = await createStoredSession(cwd);
    created.push(session.cwd);
    const reloaded = await readStoredSession(cwd, session.sessionId);
    expect(reloaded.sessionId).toBe(session.sessionId);
    expect(reloaded.config.provider).toBe('deepseek');
    expect(reloaded.config.sandbox).toBe(false);
    expect(reloaded.config.toolsEnabled).toBe(true);
    expect(reloaded.usage.turns).toBe(0);
  });
});
