import { describe, expect, it } from "vitest";
import { userPartsToOpenAi, toOpenAiMessages } from "./llm-client.js";
import type { NamedUserMessage } from "./storage.js";

describe("userPartsToOpenAi", () => {
  it("maps text parts unchanged", () => {
    expect(userPartsToOpenAi([{ type: "text", text: "hi" }])).toEqual([
      { type: "text", text: "hi" },
    ]);
  });

  it("maps images to image_url data URIs", () => {
    expect(
      userPartsToOpenAi([{ type: "image", mimeType: "image/png", data: "QUJD" }]),
    ).toEqual([{ type: "image_url", image_url: { url: "data:image/png;base64,QUJD" } }]);
  });

  it("maps wav/mp3 audio to input_audio with a format field", () => {
    expect(
      userPartsToOpenAi([
        { type: "audio", mimeType: "audio/wav", data: "UklGRg==" },
        { type: "audio", mimeType: "audio/mpeg", data: "SUQz" },
      ]),
    ).toEqual([
      { type: "input_audio", input_audio: { data: "UklGRg==", format: "wav" } },
      { type: "input_audio", input_audio: { data: "SUQz", format: "mp3" } },
    ]);
  });

  it("degrades unsupported audio containers to placeholder text", () => {
    const parts = userPartsToOpenAi([{ type: "audio", mimeType: "audio/ogg", data: "T0dn" }]);
    expect(parts).toHaveLength(1);
    expect(parts[0]).toMatchObject({ type: "text" });
    expect((parts[0] as { text: string }).text).toContain("unsupported format");
  });
});

describe("toOpenAiMessages multi-part user content", () => {
  it("emits an array content when the stored message is multi-part", () => {
    const message: NamedUserMessage = {
      role: "user",
      content: [
        { type: "text", text: "look at this" },
        { type: "image", mimeType: "image/jpeg", data: "QUJD" },
      ],
    };
    const wire = toOpenAiMessages([message], "reasoning_content");
    expect(wire).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "look at this" },
          { type: "image_url", image_url: { url: "data:image/jpeg;base64,QUJD" } },
        ],
      },
    ]);
  });

  it("keeps plain-string user messages as strings (cache-compatible)", () => {
    const wire = toOpenAiMessages([{ role: "user", content: "plain" }], "reasoning_content");
    expect(wire).toEqual([{ role: "user", content: "plain" }]);
  });
});
