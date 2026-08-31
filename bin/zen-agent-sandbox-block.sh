#!/usr/bin/env bash
# Zen Agent bash-tool sandbox: refuses to run commands whose real binaries
# are shadowed inside the bwrap namespace (see src/tools/execution.ts).
#
# The bash tool sandbox bind-mounts this script over /usr/bin/rm,
# /usr/bin/grep and /usr/bin/find (plus /bin/* where they are separate
# files). Inside the namespace the agent can therefore only use the
# substitutes (trash, rg, fdfind); the host binaries are untouched, so
# every other script on the machine keeps working as before.
#
# argv[0] is the path the command was invoked as (e.g. /usr/bin/rm), so
# basename tells us which command the agent tried to run.
set -euo pipefail

name="$(basename "$0")"
case "$name" in
  rm)    substitute="trash" ;;
  grep)  substitute="rg" ;;
  find)  substitute="fdfind" ;;
  *)     substitute="" ;;
esac

echo "'$name' is blocked; use '$substitute' instead (rm->trash, grep->rg, find->fdfind)" >&2
exit 1
