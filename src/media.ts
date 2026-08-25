import { readFile, stat } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import { maxMediaBytes } from "./media-limit.js";

/**
 * The model-facing read_media tool: turns a local file path into an actual
 * image/audio payload injected into the conversation. Without it the model
 * can list or copy media files via bash but can never perceive their content,
 * and would have to ask the user to describe them.
 */

/** Extensions accepted by read_media, mapped to their MIME type. */
const MEDIA_MIME_TYPES: Record<string, { mimeType: string; modality: "image" | "audio" }> = {
  png: { mimeType: "image/png", modality: "image" },
  jpg: { mimeType: "image/jpeg", modality: "image" },
  jpeg: { mimeType: "image/jpeg", modality: "image" },
  webp: { mimeType: "image/webp", modality: "image" },
  gif: { mimeType: "image/gif", modality: "image" },
  bmp: { mimeType: "image/bmp", modality: "image" },
  wav: { mimeType: "audio/wav", modality: "audio" },
  mp3: { mimeType: "audio/mpeg", modality: "audio" },
};

export interface ResolvedMedia {
  /** Absolute path as shown to the model/user. */
  path: string;
  modality: "image" | "audio";
  mimeType: string;
  /** Base64-encoded file contents. */
  data: string;
  decodedBytes: number;
}

/** Resolve + validate a user/model-supplied media path against the session cwd. */
export async function resolveMedia(
  cwd: string,
  rawPath: string,
  allowedModalities: ReadonlyArray<"image" | "audio">,
): Promise<ResolvedMedia> {
  const trimmed = rawPath.trim();
  if (trimmed.length === 0) {
    throw new Error("path must be a non-empty string");
  }
  const absolute = isAbsolute(trimmed) ? trimmed : join(cwd, trimmed);
  const extension = absolute.split(".").pop()?.toLowerCase() ?? "";
  const meta = MEDIA_MIME_TYPES[extension];
  if (!meta) {
    throw new Error(
      `unsupported media type ".${extension}" (expected one of ${Object.keys(MEDIA_MIME_TYPES).join(", ")})`,
    );
  }
  if (!allowedModalities.includes(meta.modality)) {
    throw new Error(`current model does not support ${meta.modality} input`);
  }

  const info = await stat(absolute).catch(() => null);
  if (!info || !info.isFile()) {
    throw new Error(`not a file: ${absolute}`);
  }
  if (info.size > maxMediaBytes()) {
    throw new Error(
      `file exceeds the media size limit (${info.size} > ${maxMediaBytes()} bytes, ZEN_AGENT_MAX_MEDIA_BYTES)`,
    );
  }

  const data = await readFile(absolute);
  return {
    path: absolute,
    modality: meta.modality,
    mimeType: meta.mimeType,
    data: data.toString("base64"),
    decodedBytes: data.length,
  };
}
