import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as acp from '@agentclientprotocol/sdk';
import { promptBlocksToPromptContent, promptBlocksToText } from '../prompt-content.js';

const PNG = Buffer.from([0x89, 0x50]).toString('base64');
const WAV = Buffer.from([0x52, 0x49]).toString('base64');

describe('promptBlocksToPromptContent', () => {
  it('converts text/image/audio blocks into parts plus a text view', async () => {
    const { text, parts } = await promptBlocksToPromptContent([
      { type: 'text', text: 'what is this?' },
      { type: 'image', data: PNG, mimeType: 'image/png', uri: 'file:///p/shot.png' },
      { type: 'audio', data: WAV, mimeType: 'audio/wav' },
    ]);
    expect(text).toContain('what is this?');
    expect(text).toContain('[image attached: file:///p/shot.png (image/png)]');
    expect(text).toContain('[audio attached (audio/wav)]');
    expect(parts).toEqual([
      { type: 'text', text: 'what is this?' },
      { type: 'image', mimeType: 'image/png', data: PNG, uri: 'file:///p/shot.png' },
      { type: 'audio', mimeType: 'audio/wav', data: WAV },
    ]);
  });

  it('degrades oversized media blocks to placeholder notes', async () => {
    const previous = process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
    process.env.ZEN_AGENT_MAX_MEDIA_BYTES = '1';
    try {
      const { parts } = await promptBlocksToPromptContent([
        { type: 'image', data: PNG, mimeType: 'image/png' },
      ]);
      expect(parts).toHaveLength(1);
      expect(parts[0]).toMatchObject({ type: 'text' });
      expect((parts[0] as { text: string }).text).toContain('ZEN_AGENT_MAX_MEDIA_BYTES');
    } finally {
      if (previous === undefined) delete process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
      else process.env.ZEN_AGENT_MAX_MEDIA_BYTES = previous;
    }
  });

  it('still throws on unsupported block types', async () => {
    await expect(promptBlocksToPromptContent([{ type: 'mystery' } as never])).rejects.toThrow(
      /Unsupported content block/,
    );
  });

  it('keeps the legacy text-only helper working', async () => {
    const text = await promptBlocksToText([{ type: 'text', text: 'hello' }]);
    expect(text).toBe('hello');
  });
});

describe('readResourceLink caps and sniffs file:// links', () => {
  const { mkdtempSync, rmSync, writeFileSync } = require('node:fs');
  const { tmpdir } = require('node:os');
  const { join } = require('node:path');

  let dir: string;
  const previousEnv = process.env.ZEN_AGENT_MAX_RESOURCE_BYTES;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'zen-resource-link-'));
  });

  afterEach(() => {
    if (previousEnv === undefined) delete process.env.ZEN_AGENT_MAX_RESOURCE_BYTES;
    else process.env.ZEN_AGENT_MAX_RESOURCE_BYTES = previousEnv;
    rmSync(dir, { recursive: true, force: true });
  });

  function link(name: string): acp.ContentBlock {
    return { type: 'resource_link', uri: `file://${join(dir, name)}`, name };
  }

  it('reads small text files in full', async () => {
    writeFileSync(join(dir, 'small.txt'), 'hello world');
    const { parts, text } = await promptBlocksToPromptContent([link('small.txt')]);
    expect(parts[0]).toEqual({ type: 'text', text: expect.stringContaining('hello world') });
    expect(text).toContain('File:');
  });

  it('truncates large files instead of blowing the context', async () => {
    process.env.ZEN_AGENT_MAX_RESOURCE_BYTES = '64';
    writeFileSync(join(dir, 'big.log'), `${'a'.repeat(100)}\nEND-OF-FILE\n`);
    const { parts } = await promptBlocksToPromptContent([link('big.log')]);
    const content = (parts[0] as { type: string; text: string }).text;
    expect(content).toContain('a'.repeat(64));
    // Nothing beyond the cap leaks into the context.
    expect(content).not.toContain('END-OF-FILE');
    expect(content).toMatch(/\[File truncated: showing 64 of \d+ bytes/);
  });

  it('omits binary files with a note instead of mojibake', async () => {
    writeFileSync(join(dir, 'img.png'), Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0x0d]));
    const { parts } = await promptBlocksToPromptContent([link('img.png')]);
    const content = (parts[0] as { type: string; text: string }).text;
    expect(content).toContain('binary content is not readable as text');
  });

  it('falls back to the link name for unreadable paths', async () => {
    const block: acp.ContentBlock = {
      type: 'resource_link',
      uri: 'file:///definitely/missing/path.txt',
      name: 'missing.txt',
    };
    const { text } = await promptBlocksToPromptContent([block]);
    expect(text).toBe('missing.txt');
  });
});
