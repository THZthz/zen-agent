#!/usr/bin/env bash
# Run Zen Agent's node process inside a bubblewrap sandbox.
#
# Sandbox policy for the agent process. The bash-tool sandbox in
# src/tool-execution.ts uses the same base policy plus read-only shims
# over rm/grep/find (see README):
#   --bind / /           whole rootfs behaves exactly as on the host
#                        (writable where it was writable, ro where it was ro)
#   --ro-bind /mnt /mnt  /mnt (Windows drives c:/d:/..., WSL mounts) becomes
#                        READ-ONLY: reads work, every write fails with EROFS
#   --dev /dev           fresh devtmpfs; binding the host /dev into a user
#                        namespace breaks device access (/dev/null EACCES)
#   --bind /dev/pts      host PTYs must be visible for ttyname(3) to work
#   --tmpfs /dev/shm     shared memory
#
# The sandboxed process runs as your normal uid in a new user+mount
# namespace, so it cannot remount /mnt read-write or escape the namespace.
#
# NOTE: the agent's bash tool runs in a PTY owned by the client (Zed) on the
# host, OUTSIDE this sandbox. To sandbox the bash tool as well, set
# ZEN_AGENT_SANDBOX=1 in the agent's env: every bash tool call is then
# wrapped in its own bwrap with the same policy.
set -euo pipefail

NODE_BIN="${ZEN_AGENT_NODE_BIN:-$(command -v node)}"
AGENT_MAIN="${ZEN_AGENT_MAIN:-/home/amias/projects/zen-agent/dist/index.js}"

exec bwrap \
  --die-with-parent \
  --bind / / \
  --ro-bind /mnt /mnt \
  --dev /dev \
  --bind /dev/pts /dev/pts \
  --tmpfs /dev/shm \
  "$NODE_BIN" "$AGENT_MAIN"
