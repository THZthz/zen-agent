export interface LogDetails {
  [key: string]: unknown;
}

/** Shape of one runtime_log / llm_log entry (stored as JSON in the db). */
export function makeLogEntry(
  level: 'debug' | 'info' | 'warn' | 'error',
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
