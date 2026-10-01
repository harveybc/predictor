#!/usr/bin/env bash
# ADM-DEADCACHE-01: the owner's ONE action.  Deploys the reviewed admission fix (predictor commit
# 775c5545) to the named hosts, or rolls it back.  Run from the coordinator:
#
#     bash DEPLOY_ADM_DEADCACHE.sh <worker_a-alias> <worker_b-alias>            # deploy
#     bash DEPLOY_ADM_DEADCACHE.sh --rollback <worker_a-alias> <worker_b-alias> # restore ddadf4a9 bytes
#
# Per host: fetch the commit, extract both files with `git show`, check their sha256, keep sha-named
# rollback copies of the deployed bytes (checked), replace by atomic rename (running launchers keep
# the text they opened; no running child is touched), then a smoke: `state` shows the dead-cache
# reading and a 256M `true` job leaves a SCOPE_CLEAN_CACHE_RECLAIM ledger record.
# It changes no limit, ceiling, cache, swap, oomd or persistence setting and restarts nothing.
set -euo pipefail
MODE=deploy
if [ "${1:-}" = "--rollback" ]; then MODE=rollback; shift; fi
[ $# -ge 1 ] || { echo "usage: $0 [--rollback] <host-alias>..." >&2; exit 2; }
for H in "$@"; do
  echo "=== $H ($MODE)"
  ssh -o BatchMode=yes "$H" MODE=$MODE bash -s <<'REMOTE'
set -euo pipefail
NEW_RUN=056e207a1f36120cb24d8063932988082f50eca368bb03c6e2bb3de9df69d7d7
NEW_ADM=7882d20fe30782b5f411be01b6ba58e2de6a26aa2e4cd4ed4876cc69b86948e2
OLD_RUN=499fdc1877750337006de8aad6b30a943acfea7416aa157b7537c4b121c1dfc7
OLD_ADM=8dc2c03b17e498697d634666309876686d051243ea4b640d65ec92c51ab8ee35
BIN=$HOME/.local/bin/crispdm-run; LIB=$HOME/.local/libexec/crispdm/crispdm_admission.py
B=$HOME/.local/state/crispdm-run/rollback_ddadf4a9; mkdir -p "$B"
put() { install -m 755 "$1" "$(dirname "$2")/.$(basename "$2").new"; mv -f "$(dirname "$2")/.$(basename "$2").new" "$2"; }
if [ "$MODE" = rollback ]; then
  echo "$OLD_RUN  $B/crispdm-run.499fdc18" | sha256sum -c -; echo "$OLD_ADM  $B/crispdm_admission.py.8dc2c03b" | sha256sum -c -
  put "$B/crispdm-run.499fdc18" "$BIN"; put "$B/crispdm_admission.py.8dc2c03b" "$LIB"
  sha256sum "$BIN" "$LIB"; exit 0
fi
R=$HOME/Documents/GitHub/predictor
git -C "$R" fetch -q origin satoshi/m06-admission-dead-cache-20260930
T=$(mktemp -d "$HOME/.local/state/adm-deploy.XXXXXX")     # not /tmp: /tmp is tmpfs on some hosts
git -C "$R" show 775c5545:tools/crispdm-run > "$T/crispdm-run"
git -C "$R" show 775c5545:tools/crispdm_admission.py > "$T/crispdm_admission.py"
echo "$NEW_RUN  $T/crispdm-run" | sha256sum -c -; echo "$NEW_ADM  $T/crispdm_admission.py" | sha256sum -c -
have_run=$(sha256sum "$BIN" | cut -d' ' -f1); have_adm=$(sha256sum "$LIB" | cut -d' ' -f1)
if [ "$have_run" = "$OLD_RUN" ] && [ ! -f "$B/crispdm-run.499fdc18" ]; then cp -p "$BIN" "$B/crispdm-run.499fdc18"; fi
if [ "$have_adm" = "$OLD_ADM" ] && [ ! -f "$B/crispdm_admission.py.8dc2c03b" ]; then cp -p "$LIB" "$B/crispdm_admission.py.8dc2c03b"; fi
echo "$OLD_RUN  $B/crispdm-run.499fdc18" | sha256sum -c -; echo "$OLD_ADM  $B/crispdm_admission.py.8dc2c03b" | sha256sum -c -
put "$T/crispdm-run" "$BIN"; put "$T/crispdm_admission.py" "$LIB"; rm -rf "$T"
sha256sum "$BIN" "$LIB"
python3 "$LIB" state | grep -E '"slice_(memory_current|charged_bytes)"'
"$BIN" -m 256M -t 2m -n deploy-smoke-adm-deadcache -- true
grep SCOPE_CLEAN_CACHE_RECLAIM "$HOME/.local/state/crispdm/admission/ledger.jsonl" | tail -1
REMOTE
done
