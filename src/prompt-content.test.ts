import { describe, expect, it } from "vitest";
import { promptBlocksToPromptContent, promptBlocksToText } from "./prompt-content.js";

const PNG = Buffer.from([0x89, 0x50]).toString("base64");
const WAV = Buffer.from([0x52, 0x49]).toString("base64");

describe("promptBlocksToPromptContent", () => {
  it("converts text/image/audio blocks into parts plus a text view", async () => {
    const { text, parts } = await promptBlocksToPromptContent([
      { type: "text", text: "what is this?" },
      { type: "image", data: PNG, mimeType: "image/png", uri: "file:///p/shot.png" },
      { type: "audio", data: WAV, mimeType: "audio/wav" },
    ]);
    expect(text).toContain("what is this?");
    expect(text).toContain("[image attached: file:///p/shot.png (image/png)]");
    expect(text).toContain("[audio attached (audio/wav)]");
    expect(parts).toEqual([
      { type: "text", text: "what is this?" },
      { type: "image", mimeType: "image/png", data: PNG, uri: "file:///p/shot.png" },
      { type: "audio", mimeType: "audio/wav", data: WAV },
    ]);
  });

  it("degrades oversized media blocks to placeholder notes", async () => {
    const previous = process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
    process.env.ZEN_AGENT_MAX_MEDIA_BYTES = "1";
    try {
      const { parts } = await promptBlocksToPromptContent([
        { type: "image", data: PNG, mimeType: "image/png" },
      ]);
      expect(parts).toHaveLength(1);
      expect(parts[0]).toMatchObject({ type: "text" });
      expect((parts[0] as { text: string }).text).toContain("ZEN_AGENT_MAX_MEDIA_BYTES");
    } finally {
      if (previous === undefined) delete process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
      else process.env.ZEN_AGENT_MAX_MEDIA_BYTES = previous;
    }
  });

  it("still throws on unsupported block types", async () => {
    await expect(
      promptBlocksToPromptContent([{ type: "mystery" } as never]),
    ).rejects.toThrow(/Unsupported content block/);
  });

  it("keeps the legacy text-only helper working", async () => {
    const text = await promptBlocksToText([{ type: "text", text: "hello" }]);
    expect(text).toBe("hello");
  });
});
