/**
 * Upper bound on one media payload, in decoded bytes. Shared by prompt-block
 * intake (client-attached images/audio) and the read_media tool so both paths
 * have the same ceiling. Override with ZEN_AGENT_MAX_MEDIA_BYTES.
 */
export const DEFAULT_MAX_MEDIA_BYTES = 10_000_000;

export function maxMediaBytes(): number {
  const raw = process.env.ZEN_AGENT_MAX_MEDIA_BYTES;
  if (!raw) return DEFAULT_MAX_MEDIA_BYTES;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_MEDIA_BYTES;
}
