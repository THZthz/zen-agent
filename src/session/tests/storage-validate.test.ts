import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { SessionUpdate } from '@agentclientprotocol/sdk';
import {
  createStoredSession,
  deleteStoredSession,
  findSessionCwd,
  normalizeEvents,
  normalizeLlmMessages,
  readStoredSession,
  validateSessionId,
  writeSession,
  type StoredSession,
} from '../storage.js';
import { openDb } from '../db.js';
import { emptyTurnStats } from '../../agent/stats.js';

describe('readStoredSession validation', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-validate-'));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  /** Insert a raw session row whose JSON columns mirror a former state.json. */
  function insertRaw(sessionId: string, raw: Record<string, unknown>): void {
    openDb()
      .prepare(
        `INSERT INTO sessions
           (session_id, cwd, created_at, updated_at, title, config, usage,
            events, llm_messages, turn_stats, cache_diagnostics)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        sessionId,
        typeof raw.cwd === 'string' ? raw.cwd : cwd,
        typeof raw.createdAt === 'string' ? raw.createdAt : '2026-01-01T00:00:00.000Z',
        typeof raw.updatedAt === 'string' ? raw.updatedAt : '2026-01-01T00:00:00.000Z',
        typeof raw.title === 'string' ? raw.title : null,
        JSON.stringify(raw.config ?? {}),
        JSON.stringify(raw.usage ?? {}),
        JSON.stringify(raw.events ?? []),
        JSON.stringify(raw.llmMessages ?? []),
        JSON.stringify(raw.turnStats ?? []),
        JSON.stringify(raw.cacheDiagnostics ?? []),
      );
  }

  it('accepts generated ids and compatible safe manual ids', async () => {
    const generated = await createStoredSession(cwd);
    created.push(generated.cwd);
    expect(generated.sessionId).toMatch(/^sess_[0-9a-f]{24}$/);
    expect(validateSessionId('sess_manual')).toBe('sess_manual');
  });

  it('rejects invalid ids before touching the database', async () => {
    for (const sessionId of ['', '.', '..', '../outside', 'nested/session', 'nested\\session']) {
      await expect(readStoredSession(cwd, sessionId)).rejects.toThrow(/Invalid session ID/);
      await expect(deleteStoredSession(sessionId)).rejects.toThrow(/Invalid session ID/);
      await expect(findSessionCwd(sessionId)).rejects.toThrow(/Invalid session ID/);
    }
  });

  it("rejects corrupt JSON columns with a clean 'corrupted' error", async () => {
    openDb()
      .prepare(
        `INSERT INTO sessions
           (session_id, cwd, created_at, updated_at, title, config, usage,
            events, llm_messages, turn_stats, cache_diagnostics)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'sess_bad',
        cwd,
        '',
        '',
        null,
        '{}',
        '{}',
        '{"sessionId": "sess_bad", "cwd:',
        '[]',
        '[]',
        '[]',
      );
    await expect(readStoredSession(cwd, 'sess_bad')).rejects.toThrow(
      /corrupted: events is not valid JSON/,
    );
  });

  it('rejects a session that belongs to another project', async () => {
    const session = await createStoredSession(cwd);
    created.push(session.cwd);
    const otherDir = mkdtempSync(join(tmpdir(), 'zen-storage-validate-other-'));
    created.push(otherDir);
    const elsewhere = await createStoredSession(otherDir);
    await expect(readStoredSession(cwd, elsewhere.sessionId)).rejects.toThrow(/belongs to/);
  });

  it('backfills missing fields from older or damaged rows instead of throwing', async () => {
    // A pre-providers, pre-sandbox legacy row with partial usage.
    insertRaw('sess_legacy', {
      llmMessages: [{ role: 'user', content: 'hi' }],
      config: { systemPrompt: '' },
      usage: { turns: 3, inputTokens: 100, costYuan: 3.5 },
    });
    const session = (await readStoredSession(cwd, 'sess_legacy')).session;
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
    await writeSession(session);
    const reloaded = (await readStoredSession(cwd, 'sess_legacy')).session;
    expect(reloaded.config.sandbox).toBe(false);
    expect(reloaded.config.toolsEnabled).toBe(true);
  });

  it('keeps a disabled tools flag (toolsEnabled: false) across a round trip', async () => {
    insertRaw('sess_tools_off', {
      config: { systemPrompt: '', sandbox: false, toolsEnabled: false },
    });
    const session = (await readStoredSession(cwd, 'sess_tools_off')).session;
    expect(session.config.toolsEnabled).toBe(false);

    await writeSession(session);
    const reloaded = (await readStoredSession(cwd, 'sess_tools_off')).session;
    expect(reloaded.config.toolsEnabled).toBe(false);
  });

  it('accepts the full thinking-effort vocabulary and rejects unknown values', async () => {
    insertRaw('sess_xhigh', {
      llmMessages: [{ role: 'user', content: 'hi' }],
      config: { systemPrompt: '', thinkingEffort: 'xhigh' },
    });
    const session = (await readStoredSession(cwd, 'sess_xhigh')).session;
    expect(session.config.thinkingEffort).toBe('xhigh');

    insertRaw('sess_bad_effort', {
      llmMessages: [{ role: 'user', content: 'hi' }],
      config: { systemPrompt: '', thinkingEffort: 'turbo' },
    });
    const bad = (await readStoredSession(cwd, 'sess_bad_effort')).session;
    expect(bad.config.thinkingEffort).toBe('off');
  });

  it('keeps loading healthy sessions unchanged', async () => {
    const session: StoredSession = await createStoredSession(cwd);
    created.push(session.cwd);
    const reloaded = (await readStoredSession(cwd, session.sessionId)).session;
    expect(reloaded.sessionId).toBe(session.sessionId);
    expect(session.sessionId).toMatch(/^sess_[0-9a-f]{24}$/);
    expect(reloaded.config.provider).toBe('deepseek');
    expect(reloaded.config.sandbox).toBe(false);
    expect(reloaded.config.toolsEnabled).toBe(true);
    expect(reloaded.usage.turns).toBe(0);
  });
});

describe('normalizeEvents', () => {
  it('keeps only objects with a non-empty sessionUpdate discriminator', () => {
    const { items, dropped } = normalizeEvents([
      null,
      'garbage',
      { sessionUpdate: '' },
      { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'kept' } },
      // Unknown (e.g. newer) kinds are syntactically fine and keep loading.
      { sessionUpdate: 'some_future_kind', x: 1 },
    ]);
    expect(items).toEqual([
      { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'kept' } },
      { sessionUpdate: 'some_future_kind', x: 1 },
    ]);
    expect(dropped).toBe(3);
  });

  it('degrades a non-array field to empty without counting drops', () => {
    expect(normalizeEvents('nope')).toEqual({ items: [], dropped: 0 });
    expect(normalizeEvents(undefined)).toEqual({ items: [], dropped: 0 });
  });
});

describe('normalizeLlmMessages', () => {
  it('keeps only messages with a usable role/content shape', () => {
    const { items, dropped } = normalizeLlmMessages([
      null,
      { role: 'bogus' },
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: [{ type: 'text', text: 'kept' }, 42, { type: 'reasoning' }] },
      { role: 'tool', content: 'not-an-array' },
      { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', output: 'ok' }] },
    ]);
    expect(items).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: [{ type: 'text', text: 'kept' }] },
      { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', output: 'ok' }] },
    ]);
    expect(dropped).toBe(3);
  });

  it('filters invalid parts from user messages and drops a non-string name', () => {
    const { items, dropped } = normalizeLlmMessages([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'kept' },
          { type: 'mystery', data: 'x' },
          { type: 'image', mimeType: 'image/png', data: 'aGk=' },
          { type: 'audio', mimeType: 'audio/wav', data: 'aGk=' },
        ],
      },
      { role: 'user', content: [{ type: 'text', text: 5 }] },
      { role: 'user', name: 9, content: 'named' },
    ]);
    expect(items).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'kept' },
          { type: 'image', mimeType: 'image/png', data: 'aGk=' },
          { type: 'audio', mimeType: 'audio/wav', data: 'aGk=' },
        ],
      },
      { role: 'user', content: 'named' },
    ]);
    expect(dropped).toBe(1);
  });

  it('drops a tool message unless every part is a tool-result with a toolCallId', () => {
    const { items, dropped } = normalizeLlmMessages([
      { role: 'tool', content: [{ type: 'tool-result', output: 'no-id' }] },
      { role: 'tool', content: [{ type: 'text', text: 'wrong type' }] },
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'a', output: 'ok' },
          { type: 'tool-result', toolCallId: 'b', toolName: 'bash', output: { out: 1 } },
        ],
      },
    ]);
    expect(items).toEqual([
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'a', output: 'ok' },
          { type: 'tool-result', toolCallId: 'b', toolName: 'bash', output: { out: 1 } },
        ],
      },
    ]);
    expect(dropped).toBe(2);
  });
});

describe('malformed persisted elements', () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), 'zen-storage-elements-'));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  function insertRaw(sessionId: string, raw: Record<string, unknown>): void {
    openDb()
      .prepare(
        `INSERT INTO sessions
           (session_id, cwd, created_at, updated_at, title, config, usage,
            events, llm_messages, turn_stats, cache_diagnostics)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        sessionId,
        typeof raw.cwd === 'string' ? raw.cwd : cwd,
        typeof raw.createdAt === 'string' ? raw.createdAt : '2026-01-01T00:00:00.000Z',
        typeof raw.updatedAt === 'string' ? raw.updatedAt : '2026-01-01T00:00:00.000Z',
        typeof raw.title === 'string' ? raw.title : null,
        JSON.stringify(raw.config ?? {}),
        JSON.stringify(raw.usage ?? {}),
        JSON.stringify(raw.events ?? []),
        JSON.stringify(raw.llmMessages ?? []),
        JSON.stringify(raw.turnStats ?? []),
        JSON.stringify(raw.cacheDiagnostics ?? []),
      );
  }

  it('loads a session with malformed elements, keeping only usable entries', async () => {
    insertRaw('sess_malformed', {
      events: [
        null,
        'garbage',
        { sessionUpdate: '' },
        { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'kept' } },
      ],
      llmMessages: [
        null,
        7,
        { role: 'bogus' },
        { role: 'user', content: 42 },
        { role: 'user', content: [{ type: 'text', text: 5 }, { type: 'image' }] },
        { role: 'user', name: 9, content: 'named' },
        { role: 'assistant', content: 'nope' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'kept' },
            { type: 'tool-call', toolName: 'bash', input: {} },
          ],
        },
        { role: 'assistant', content: [{ type: 'reasoning' }] },
        { role: 'tool', content: 'not-an-array' },
        { role: 'tool', content: [{ type: 'tool-result', output: 'no-id' }] },
        { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', output: 'kept' }] },
      ],
      turnStats: [null, 'x', { ...emptyTurnStats(), steps: 1 }],
      cacheDiagnostics: [null, { ts: 1 }],
      config: { systemPrompt: '' },
      usage: {},
    });
    const { session, droppedEntries } = await readStoredSession(cwd, 'sess_malformed');

    expect(session.events).toEqual([
      { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'kept' } },
    ]);
    expect(session.llmMessages).toEqual([
      { role: 'user', content: 'named' },
      { role: 'assistant', content: [{ type: 'text', text: 'kept' }] },
      { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', output: 'kept' }] },
    ]);
    expect(session.turnStats).toEqual([{ ...emptyTurnStats(), steps: 1 }]);
    expect(session.cacheDiagnostics).toEqual([{ ts: 1 }]);
    // events: 3; messages: null, 7, bogus, user 42, all-parts-invalid,
    // assistant non-array, assistant all-invalid, tool non-array,
    // tool no-id = 9; turnStats: 2; cacheDiagnostics: 1.
    expect(droppedEntries).toBe(15);
  });

  it('round-trips a healthy session unchanged through write -> read -> write', async () => {
    const session = await createStoredSession(cwd);
    created.push(session.cwd);
    session.events.push(
      {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: 'hi' },
      } as unknown as SessionUpdate,
      {
        sessionUpdate: 'tool_call',
        toolCallId: 'c1',
        title: '$ ls',
        kind: 'execute',
        status: 'pending',
        rawInput: { command: 'ls' },
        _meta: { terminal_info: { terminal_id: 'zen-c1' } },
      } as unknown as SessionUpdate,
    );
    session.llmMessages.push(
      { role: 'user', name: 'environment', content: 'env snapshot' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'look' },
          { type: 'image', mimeType: 'image/png', data: 'aGk=' },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'thinking', reasoningSignature: 'sig' },
          { type: 'text', text: 'doing' },
          { type: 'tool-call', toolCallId: 'c1', toolName: 'bash', input: { command: 'ls' } },
        ],
      },
      {
        role: 'tool',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'bash', output: 'out' }],
      },
    );
    session.turnStats.push({ ...emptyTurnStats(), steps: 2, cost: 0.5 });
    session.cacheDiagnostics?.push({
      ts: 1_700_000_000_000,
      turn: 1,
      model: 'deepseek-v4-flash',
      inputTokens: 100,
      cachedTokens: 90,
      cacheMissTokens: 10,
      cacheHitRate: 0.9,
      savedCost: 0.01,
      prefixHash: 'p',
      systemHash: 's',
      toolSpecsHash: 't',
      envHash: 'e',
      toolCount: 1,
      toolNames: ['bash'],
      missReason: 'no-miss',
      missReasonDetail: '',
    });
    await writeSession(session);

    const { session: reloaded, droppedEntries } = await readStoredSession(cwd, session.sessionId);
    expect(droppedEntries).toBe(0);
    await writeSession(reloaded);
    const reread = (await readStoredSession(cwd, session.sessionId)).session;

    // Structural identity guards the provider cache prefix: normalization
    // must not churn healthy persisted content.
    expect(reread).toEqual(session);
  });
});
