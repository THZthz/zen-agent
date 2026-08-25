import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { existsSync, mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStoredSession, deleteStoredSession } from "./storage.js";

describe("deleteStoredSession", () => {
  let cwd: string;
  const created: string[] = [];

  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "zen-storage-delete-"));
  });

  afterEach(() => {
    for (const dir of created) rmSync(dir, { recursive: true, force: true });
    created.length = 0;
  });

  it("removes the whole session tree, not just state.json", async () => {
    const indexDirBefore = process.env.XDG_DATA_HOME;
    process.env.XDG_DATA_HOME = join(cwd, "xdg");
    try {
      const session = await createStoredSession(cwd);
      created.push(session.cwd);

      // Artifacts a real session accumulates.
      const terminals = join(cwd, ".sessions", session.sessionId, "terminals");
      mkdirSync(terminals, { recursive: true });
      writeFileSync(join(terminals, "output-1-c1.log"), "out");
      writeFileSync(
        join(cwd, ".sessions", session.sessionId, "llm.jsonl"),
        "{}\n",
      );

      expect(existsSync(join(cwd, ".sessions", session.sessionId))).toBe(true);

      await deleteStoredSession(cwd, session.sessionId);

      expect(existsSync(join(cwd, ".sessions", session.sessionId))).toBe(false);
      const index = JSON.parse(
        await readFileOrNull(join(process.env.XDG_DATA_HOME!, "zen-agent", "index.json")),
      );
      expect(index[session.sessionId]).toBeUndefined();
    } finally {
      if (indexDirBefore === undefined) delete process.env.XDG_DATA_HOME;
      else process.env.XDG_DATA_HOME = indexDirBefore;
    }
  });
});

async function readFileOrNull(path: string): Promise<string> {
  const { readFile } = await import("node:fs/promises");
  try {
    return await readFile(path, "utf8");
  } catch {
    return "{}";
  }
}
