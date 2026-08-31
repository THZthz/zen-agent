/**
 * Upper bound on one media payload, in decoded bytes. Shared by prompt-block
 * intake (client-attached images/audio) and the read_media tool so both paths
 * have the same ceiling. Override with ZEN_AGENT_MAX_MEDIA_BYTES.
 */
import { envPositiveInt } from '../util/env.js';

export const DEFAULT_MAX_MEDIA_BYTES = 10_000_000;

export function maxMediaBytes(): number {
  return envPositiveInt('ZEN_AGENT_MAX_MEDIA_BYTES', DEFAULT_MAX_MEDIA_BYTES);
}
