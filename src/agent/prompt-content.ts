import * as acp from '@agentclientprotocol/sdk';
import { open } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import type { UserContentPart } from '../session/storage.js';
import { maxMediaBytes } from '../tools/media-limit.js';

/** Base64 length to decoded-byte estimate (no padding round-trip needed). */
function base64Bytes(data: string): number {
  return Math.floor((data.length * 3) / 4);
}

/**
 * Upper bound on bytes read from a `file://` resource link. Without it, a
 * linked multi-GB log would be read fully into memory and pushed verbatim
 * into the LLM context (media blocks have an equivalent ceiling in
 * media-limit.ts). Override with ZEN_AGENT_MAX_RESOURCE_BYTES.
 */
export const DEFAULT_MAX_RESOURCE_BYTES = 262_144;

function maxResourceBytes(): number {
  const raw = process.env.ZEN_AGENT_MAX_RESOURCE_BYTES;
  if (!raw) {
    return DEFAULT_MAX_RESOURCE_BYTES;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_RESOURCE_BYTES;
}

/** A NUL byte in the head of the file is the classic text-vs-binary sniff. */
function looksBinary(data: Buffer): boolean {
  return data.subarray(0, 8_192).includes(0);
}

export interface PromptContent {
  /**
   * Plain-text view of the prompt (text, resolved resource links and
   * placeholder notes for media). Used for slash-command detection, logs and
   * transcript events.
   */
  text: string;
  /**
   * Full content for the LLM message: text plus supported media parts.
   * Empty when the prompt was pure text (the caller then stores a plain
   * string content, keeping state.json byte-compatible with old sessions).
   */
  parts: UserContentPart[];
}

/**
 * Convert ACP prompt blocks into our user-message representation.
 *
 * Image/audio blocks become UserContentParts; whether they are actually sent
 * as media or degraded to placeholder text is decided by the caller based on
 * the active model's input modalities (see ZenAgent.prompt), so this function
 * always yields the full-fidelity parts.
 */
export async function promptBlocksToPromptContent(
  blocks: acp.ContentBlock[],
): Promise<PromptContent> {
  const parts: UserContentPart[] = [];

  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        parts.push({ type: 'text', text: block.text });
        break;
      case 'image': {
        if (base64Bytes(block.data) > maxMediaBytes()) {
          parts.push(oversizedMediaNote('image', block.mimeType, block.data.length));
          break;
        }
        parts.push({
          type: 'image',
          mimeType: block.mimeType,
          data: block.data,
          ...(block.uri ? { uri: block.uri } : {}),
        });
        break;
      }
      case 'audio': {
        if (base64Bytes(block.data) > maxMediaBytes()) {
          parts.push(oversizedMediaNote('audio', block.mimeType, block.data.length));
          break;
        }
        parts.push({
          type: 'audio',
          mimeType: block.mimeType,
          data: block.data,
        });
        break;
      }
      case 'resource_link':
        parts.push({ type: 'text', text: await readResourceLink(block) });
        break;
      case 'resource': {
        const resource = block.resource;
        if ('text' in resource && typeof resource.text === 'string') {
          parts.push({ type: 'text', text: resource.text });
        } else if ('blob' in resource && typeof resource.blob === 'string') {
          parts.push({
            type: 'text',
            text: `[Embedded binary resource ${resource.uri} (base64, ${resource.blob.length} chars)]`,
          });
        } else {
          parts.push({ type: 'text', text: `[Embedded resource ${resource.uri}]` });
        }
        break;
      }
      default:
        throw new Error(`Unsupported content block: ${(block as { type: string }).type}`);
    }
  }

  // The plain-text view joins text parts and summarizes media parts.
  const text = parts
    .map((part) => {
      switch (part.type) {
        case 'text':
          return part.text;
        case 'image':
          return `[image attached${part.uri ? `: ${part.uri}` : ''} (${part.mimeType})]`;
        case 'audio':
          return `[audio attached (${part.mimeType})]`;
      }
    })
    .join('\n\n');

  return { text, parts };
}

/** Legacy helper: text-only view of a prompt (slash commands, logs). */
export async function promptBlocksToText(blocks: acp.ContentBlock[]): Promise<string> {
  return (await promptBlocksToPromptContent(blocks)).text;
}

function oversizedMediaNote(
  kind: 'image' | 'audio',
  mimeType: string,
  base64Length: number,
): UserContentPart {
  return {
    type: 'text',
    text: `[${kind} attached (${mimeType}, base64 ${base64Length} chars) omitted: exceeds ZEN_AGENT_MAX_MEDIA_BYTES]`,
  };
}

async function readResourceLink(block: {
  type: 'resource_link';
  uri: string;
  name?: string;
  mimeType?: string | null;
}): Promise<string> {
  if (!block.uri.startsWith('file://')) {
    return block.name ?? block.uri;
  }

  let handle: Awaited<ReturnType<typeof open>> | null = null;
  try {
    const path = fileURLToPath(block.uri);
    // Read at most limit+1 bytes directly: a huge file must neither land in
    // memory nor in the context in full.
    const limit = maxResourceBytes();
    handle = await open(path, 'r');
    const buf = Buffer.alloc(limit + 1);
    const { bytesRead } = await handle.read(buf, 0, limit + 1, 0);
    const data = buf.subarray(0, bytesRead);

    if (looksBinary(data)) {
      return `[File: ${path} omitted: binary content is not readable as text]`;
    }
    if (bytesRead > limit) {
      const totalBytes = await handle.stat().then(
        (st) => st.size,
        () => bytesRead,
      );
      return (
        `File: ${path}\n${data.subarray(0, limit).toString('utf8')}\n\n` +
        `[File truncated: showing ${limit} of ${totalBytes} bytes (ZEN_AGENT_MAX_RESOURCE_BYTES). Read the rest with bash.]`
      );
    }
    return `File: ${path}\n${data.toString('utf8')}`;
  } catch {
    return block.name ?? block.uri;
  } finally {
    await handle?.close().catch(() => {});
  }
}
