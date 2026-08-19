import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

export interface LogDetails {
  [key: string]: unknown;
}

export async function appendJsonLine(
  filePath: string,
  data: unknown,
): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await appendFile(filePath, `${JSON.stringify(data)}\n`, "utf8");
}

export function makeLogEntry(
  level: "debug" | "info" | "warn" | "error",
  message: string,
  details?: LogDetails,
): Record<string, unknown> {
  return {
    timestamp: new Date().toISOString(),
    level,
    message,
    ...(details ?? {}),
  };
}
