#!/usr/bin/env bash
# Refresh the deployed launcher from this checkout.  DR01 (order 2026-09-26).
#
#   tools/install_crispdm_launcher.sh [--check]
#
# The launcher and its admission module are TRACKED HERE; ~/.local/bin/crispdm-run is a
# deployed copy.  Until DR01 that copy had no tracked origin at all, so it could drift with
# nothing to compare against.  This script is the only sanctioned way to refresh it.
#
#   ~/.local/bin/crispdm-run                        <- tools/crispdm-run
#   ~/.local/libexec/crispdm/crispdm_admission.py   <- tools/crispdm_admission.py
#
# FUTURE LAUNCHES ONLY.  Each file is written to a temporary name in the destination directory
# and then renamed over the old one.  A rename replaces the directory entry; a crispdm-run that
# is already running keeps executing the text it opened, and no limit, cgroup or environment of
# any running child is touched.  Nothing is signalled, started or stopped.
#
# --check prints both digests and exits 1 when the deployed copy differs, changing nothing.
set -euo pipefail

src=$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)
bin=${CRISPDM_INSTALL_BIN:-$HOME/.local/bin}
lib=${CRISPDM_INSTALL_LIBEXEC:-$HOME/.local/libexec/crispdm}
check=0
[ "${1:-}" = "--check" ] && check=1

pairs=("$src/crispdm-run|$bin/crispdm-run" "$src/crispdm_admission.py|$lib/crispdm_admission.py")

rc=0
for pair in "${pairs[@]}"; do
  from=${pair%%|*}; to=${pair##*|}
  want=$(sha256sum "$from" | cut -d' ' -f1)
  have=$(sha256sum "$to" 2>/dev/null | cut -d' ' -f1 || true)
  if [ "$check" = 1 ]; then
    printf '%s\n  tracked  %s\n  deployed %s\n' "$to" "$want" "${have:-ABSENT}"
    [ "$want" = "$have" ] || rc=1
    continue
  fi
  if [ "$want" = "$have" ]; then
    printf 'unchanged %s (%s)\n' "$to" "$want"
    continue
  fi
  mkdir -p -- "$(dirname -- "$to")"
  tmp=$(mktemp "$(dirname -- "$to")/.crispdm-install.XXXXXX")
  cat -- "$from" > "$tmp"
  chmod 0755 "$tmp"
  mv -f -- "$tmp" "$to"          # atomic rename: only launches STARTED AFTER this see the new text
  printf 'installed %s\n  was %s\n  now %s\n' "$to" "${have:-ABSENT}" "$want"
done
exit "$rc"
