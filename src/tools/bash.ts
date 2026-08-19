import { spawn } from "node:child_process";

export interface BashResult {
  output: string;
  exitCode: number | null;
  cancelled: boolean;
  truncated: boolean;
}

const MAX_OUTPUT_CHARS = 30_000;
const TRUNCATION_SUFFIX = "\n\n[truncated]";

function stripAnsi(value: string): string {
  // Matches ANSI CSI and OSC escape sequences commonly emitted by CLIs.
  // eslint-disable-next-line no-control-regex
  return value.replace(/\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))/g, "");
}

function sanitizeBinaryOutput(value: string): string {
  let result = "";
  for (const char of value) {
    const code = char.codePointAt(0)!;
    if (
      code === 0x0a ||
      code === 0x0d ||
      code === 0x09 ||
      (code >= 0x20 && code <= 0x10ffff)
    ) {
      result += char;
    } else {
      result += "\uFFFD";
    }
  }
  return result.replace(/\r/g, "");
}

export function executeBash(
  command: string,
  cwd: string,
  signal?: AbortSignal,
): Promise<BashResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("/bin/bash", ["-lc", command], {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
      windowsHide: true,
    });

    const chunks: Buffer[] = [];
    let outputBytes = 0;
    let cancelled = false;

    const decoder = new TextDecoder();

    const onData = (chunk: Buffer) => {
      chunks.push(chunk);
      outputBytes += chunk.length;
    };

    child.stdout?.on("data", onData);
    child.stderr?.on("data", onData);

    const onAbort = () => {
      cancelled = true;
      if (child.pid) {
        try {
          process.kill(-child.pid, "SIGTERM");
        } catch {
          // Process group may already be gone.
        }
      }
    };

    if (signal) {
      if (signal.aborted) {
        onAbort();
      } else {
        signal.addEventListener("abort", onAbort, { once: true });
      }
    }

    child.on("error", (error) => {
      if (signal) signal.removeEventListener("abort", onAbort);
      if (cancelled) {
        resolve({
          output: decodeOutput(chunks, decoder),
          exitCode: null,
          cancelled: true,
          truncated: false,
        });
      } else {
        reject(error);
      }
    });

    child.on("close", (exitCode) => {
      if (signal) signal.removeEventListener("abort", onAbort);

      const fullOutput = decodeOutput(chunks, decoder);
      const truncated = fullOutput.length > MAX_OUTPUT_CHARS;
      const output = truncated
        ? fullOutput.slice(0, MAX_OUTPUT_CHARS) + TRUNCATION_SUFFIX
        : fullOutput;

      resolve({
        output,
        exitCode,
        cancelled,
        truncated,
      });
    });
  });
}

function decodeOutput(chunks: Buffer[], decoder: TextDecoder): string {
  let text = "";
  for (const chunk of chunks) {
    text += decoder.decode(chunk, { stream: true });
  }
  text += decoder.decode();
  return sanitizeBinaryOutput(stripAnsi(text));
}
