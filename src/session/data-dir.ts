import { randomBytes } from 'node:crypto';
import { mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

/**
 * User-level (cross-project) data directory helpers. Session persistence
 * itself lives in the SQLite database (see db.ts / storage.ts); this module
 * only serves the file-based model catalog cache under
 * $XDG_DATA_HOME/zen-agent (see providers/catalog.ts, providers/pi.ts).
 */

export function indexDirectory(): string {
  const dataHome = process.env.XDG_DATA_HOME ?? join(homedir(), '.local', 'share');
  return join(dataHome, 'zen-agent');
}

/** Crash-safe replace used by session state, model metadata and index entries. */
export async function writeFileAtomic(filePath: string, contents: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
  try {
    await writeFile(tmp, contents, 'utf8');
    await rename(tmp, filePath);
  } catch (error) {
    await unlink(tmp).catch(() => {});
    throw error;
  }
}
