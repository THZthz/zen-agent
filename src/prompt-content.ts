import * as acp from "@agentclientprotocol/sdk";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import type { UserContentPart } from "./storage.js";

/**
 * Upper bound on a single media block's base64 payload, in decoded bytes.
 * Blocks above the limit are degraded to placeholder text instead of being
 * forwarded to the LLM (protects request size and state.json). Override
 * with ZEN_AGENT_MAX_MEDIA_BYTES.
 */
const DEFAULT_MAX_MEDIA_BYTES = 10_000_000;

function maxMediaBytes(): number {
  const raw = process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
  if (!raw) return DEFAULT_MAX_MEDIA_BYTES;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_MEDIA_BYTES;
}

/** Base64 length to decoded-byte estimate (no padding round-trip needed). */
function base64Bytes(data: string): number {
  return Math.floor((data.length * 3) / 4);
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
      case "text":
        parts.push({ type: "text", text: block.text });
        break;
      case "image": {
        if (base64Bytes(block.data) > maxMediaBytes()) {
          parts.push(oversizedMediaNote("image", block.mimeType, block.data.length));
          break;
        }
        parts.push({
          type: "image",
          mimeType: block.mimeType,
          data: block.data,
          ...(block.uri ? { uri: block.uri } : {}),
        });
        break;
      }
      case "audio": {
        if (base64Bytes(block.data) > maxMediaBytes()) {
          parts.push(oversizedMediaNote("audio", block.mimeType, block.data.length));
          break;
        }
        parts.push({
          type: "audio",
          mimeType: block.mimeType,
          data: block.data,
        });
        break;
      }
      case "resource_link":
        parts.push({ type: "text", text: await readResourceLink(block) });
        break;
      case "resource": {
        const resource = block.resource;
        if ("text" in resource && typeof resource.text === "string") {
          parts.push({ type: "text", text: resource.text });
        } else if ("blob" in resource && typeof resource.blob === "string") {
          parts.push({
            type: "text",
            text: `[Embedded binary resource ${resource.uri} (base64, ${resource.blob.length} chars)]`,
          });
        } else {
          parts.push({ type: "text", text: `[Embedded resource ${resource.uri}]` });
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
        case "text":
          return part.text;
        case "image":
          return `[image attached${part.uri ? `: ${part.uri}` : ""} (${part.mimeType})]`;
        case "audio":
          return `[audio attached (${part.mimeType})]`;
      }
    })
    .join("\n\n");

  return { text, parts };
}

/** Legacy helper: text-only view of a prompt (slash commands, logs). */
export async function promptBlocksToText(blocks: acp.ContentBlock[]): Promise<string> {
  return (await promptBlocksToPromptContent(blocks)).text;
}

function oversizedMediaNote(
  kind: "image" | "audio",
  mimeType: string,
  base64Length: number,
): UserContentPart {
  return {
    type: "text",
    text: `[${kind} attached (${mimeType}, base64 ${base64Length} chars) omitted: exceeds ZEN_AGENT_MAX_MEDIA_BYTES]`,
  };
}

async function readResourceLink(block: {
  type: "resource_link";
  uri: string;
  name?: string;
  mimeType?: string | null;
}): Promise<string> {
  if (!block.uri.startsWith("file://")) {
    return block.name ?? block.uri;
  }

  try {
    const path = fileURLToPath(block.uri);
    const content = await readFile(path, "utf8");
    return `File: ${path}\n${content}`;
  } catch {
    return block.name ?? block.uri;
  }
}
