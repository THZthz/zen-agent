import { describe, expect, it } from "vitest";
import {
  DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
  truncateTerminalOutput,
} from "./tool-execution.js";

describe("truncateTerminalOutput", () => {
  it("returns short text unchanged", () => {
    expect(truncateTerminalOutput("hello", 10)).toEqual({
      text: "hello",
      truncated: false,
      originalBytes: 5,
      keptBytes: 5,
    });
  });

  it("keeps the tail when the text exceeds the byte budget", () => {
    const result = truncateTerminalOutput("abcdefghijklmnopqrstuvwxyz", 10);
    expect(result.truncated).toBe(true);
    expect(result.text).toBe("qrstuvwxyz");
    expect(result.originalBytes).toBe(26);
    expect(result.keptBytes).toBe(10);
  });

  it("keeps the end of the output (where errors and exit codes appear)", () => {
    const result = truncateTerminalOutput("head\n" + "x".repeat(200) + "\nexit code 1", 20);
    expect(result.text.endsWith("exit code 1")).toBe(true);
  });

  it("never splits a UTF-8 multi-byte sequence", () => {
    const input = "a".repeat(20) + "🎉".repeat(10); // each 🎉 is 4 bytes
    const result = truncateTerminalOutput(input, 12);
    expect(result.truncated).toBe(true);
    expect(result.text).toBe("🎉🎉🎉");
    expect(Buffer.byteLength(result.text, "utf8")).toBeLessThanOrEqual(12);
    expect(result.text).not.toContain("\uFFFD"); // no replacement chars
  });

  it("handles a cut landing inside a multi-byte sequence by keeping fewer bytes", () => {
    const input = "a".repeat(10) + "🎉".repeat(10); // 10 + 40 = 50 bytes
    const result = truncateTerminalOutput(input, 15);
    // Byte 35 is the 7th emoji's lead; kept = 15 bytes exactly.
    // Cut at 35: emojis start at byte 10, each 4 bytes: emoji #7 = bytes 34-37.
    // Byte 35 is a continuation -> skip to 36 (also continuation) -> 37? no:
    //   #7: 34(lead),35,36,37(cont); #8 starts at 38. start=35 -> advance to 38.
    expect(Buffer.byteLength(result.text, "utf8")).toBeLessThanOrEqual(15);
    expect(result.text).toBe("🎉🎉🎉"); // 12 bytes, the last 3 emoji
    expect(result.text).not.toContain("\uFFFD");
  });

  it("exposes the default byte limit constant", () => {
    expect(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT).toBe(50_000);
  });
});
