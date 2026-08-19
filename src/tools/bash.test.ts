import { describe, expect, it } from "vitest";
import { executeBash } from "./bash.js";

describe("executeBash", () => {
  it("executes a command and captures output", async () => {
    const result = await executeBash("printf 'hello'", process.cwd());
    expect(result.output).toBe("hello");
    expect(result.exitCode).toBe(0);
    expect(result.cancelled).toBe(false);
  });

  it("reports non-zero exit codes", async () => {
    const result = await executeBash("echo oops >&2; exit 3", process.cwd());
    expect(result.output).toContain("oops");
    expect(result.exitCode).toBe(3);
  });

  it("cancels a running process", async () => {
    const controller = new AbortController();
    const promise = executeBash("sleep 10", process.cwd(), controller.signal);
    setTimeout(() => controller.abort(), 50);
    const result = await promise;
    expect(result.cancelled).toBe(true);
    expect(result.exitCode).toBe(null);
  });
});
