import * as acp from "@agentclientprotocol/sdk";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

export async function promptBlocksToText(
  blocks: acp.ContentBlock[],
): Promise<string> {
  const parts: string[] = [];

  for (const block of blocks) {
    switch (block.type) {
      case "text":
        parts.push(block.text);
        break;
      case "resource_link": {
        const text = await readResourceLink(block);
        parts.push(text);
        break;
      }
      case "resource": {
        const resource = block.resource;
        if ("text" in resource && typeof resource.text === "string") {
          parts.push(resource.text);
        } else if ("blob" in resource && typeof resource.blob === "string") {
          parts.push(
            `[Embedded binary resource ${resource.uri} (base64, ${resource.blob.length} chars)]`,
          );
        } else {
          parts.push(`[Embedded resource ${resource.uri}]`);
        }
        break;
      }
      case "image":
      case "audio":
        throw new Error(
          `${block.type} content is not supported by zen-agent yet`,
        );
      default:
        throw new Error(`Unsupported content block: ${(block as { type: string }).type}`);
    }
  }

  return parts.join("\n\n");
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
