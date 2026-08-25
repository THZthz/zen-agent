import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveMedia } from "./media.js";

let dir: string;
afterEach(() => {
  if (dir) rmSync(dir, { recursive: true, force: true });
  delete process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
});

function setup(): string {
  dir = mkdtempSync(join(tmpdir(), "zen-media-test-"));
  writeFileSync(join(dir, "shot.png"), Buffer.from([0x89, 0x50, 0x4e, 0x47]));
  writeFileSync(join(dir, "clip.wav"), Buffer.from([0x52, 0x49, 0x46, 0x46]));
  return dir;
}

describe("resolveMedia", () => {
  it("resolves relative paths against cwd and maps extensions to MIME types", async () => {
    const cwd = setup();
    const image = await resolveMedia(cwd, "shot.png", ["image"]);
    expect(image.mimeType).toBe("image/png");
    expect(image.modality).toBe("image");
    expect(image.decodedBytes).toBe(4);
    expect(image.data).toBe(Buffer.from([0x89, 0x50, 0x4e, 0x47]).toString("base64"));
    expect(image.path).toBe(join(cwd, "shot.png"));

    const audio = await resolveMedia(cwd, join(cwd, "clip.wav"), ["image", "audio"]);
    expect(audio.mimeType).toBe("audio/wav");
    expect(audio.modality).toBe("audio");
  });

  it("rejects unknown extensions, directories, and missing files", async () => {
    const cwd = setup();
    await expect(resolveMedia(cwd, "notes.txt", ["image"])).rejects.toThrow(/unsupported media type/);
    await expect(resolveMedia(cwd, ".", ["image"])).rejects.toThrow(); // stat().isFile() false
    await expect(resolveMedia(cwd, "missing.png", ["image"])).rejects.toThrow(/not a file/);
  });

  it("enforces the requested modality allow-list", async () => {
    const cwd = setup();
    await expect(resolveMedia(cwd, "shot.png", [])).rejects.toThrow(/does not support image input/);
    await expect(resolveMedia(cwd, "clip.wav", ["image"])).rejects.toThrow(/does not support audio input/);
    await expect(resolveMedia(cwd, "shot.png", ["image"])).resolves.toBeTruthy();
  });

  it("honors ZEN_AGENT_MAX_MEDIA_BYTES", async () => {
    const cwd = setup();
    process.env.ZEN_AGENT_MAX_MEDIA_BYTES = "2";
    await expect(resolveMedia(cwd, "shot.png", ["image"])).rejects.toThrow(/media size limit/);
  });
});
