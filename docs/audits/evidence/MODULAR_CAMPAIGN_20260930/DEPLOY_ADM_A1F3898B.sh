#!/usr/bin/env bash
# ADM-STALE-LEASE-01 + ADM-CANCEL-01 (predictor a1f3898b, on top of 775c5545): deploy or roll back.
#   bash DEPLOY_ADM_A1F3898B.sh <host-alias>...              deploy
#   bash DEPLOY_ADM_A1F3898B.sh --rollback <host-alias>...   restore the 775c5545 bytes
# The admission module is installed BEFORE the launcher: the new launcher passes --holder-pid, which
# only the new module accepts; the new module still accepts calls without it.  Atomic renames only;
# running launchers keep the text they opened.  No limit/ceiling/cache/swap/oomd/persistence change.
set -euo pipefail
MODE=deploy
if [ "${1:-}" = "--rollback" ]; then MODE=rollback; shift; fi
[ $# -ge 1 ] || { echo "usage: $0 [--rollback] <host-alias>..." >&2; exit 2; }
for H in "$@"; do
  echo "=== $H ($MODE)"
  ssh -o BatchMode=yes "$H" MODE=$MODE bash -s <<'REMOTE'
set -euo pipefail
NEW_RUN=bdc50f6063942cd16ebcaaedca3dba138738cbdfc92d07278eb8c3915ca1c666
NEW_ADM=e4128596a73014e93ec17856324ea7bc83bd623c8b4b612cf866309225f31022
OLD_RUN=056e207a1f36120cb24d8063932988082f50eca368bb03c6e2bb3de9df69d7d7
OLD_ADM=7882d20fe30782b5f411be01b6ba58e2de6a26aa2e4cd4ed4876cc69b86948e2
BIN=$HOME/.local/bin/crispdm-run; LIB=$HOME/.local/libexec/crispdm/crispdm_admission.py
B=$HOME/.local/state/crispdm-run/rollback_775c5545; mkdir -p "$B"
put() { install -m 755 "$1" "$(dirname "$2")/.$(basename "$2").new"; mv -f "$(dirname "$2")/.$(basename "$2").new" "$2"; }
if [ "$MODE" = rollback ]; then
  echo "$OLD_RUN  $B/crispdm-run.056e207a" | sha256sum -c -; echo "$OLD_ADM  $B/crispdm_admission.py.7882d20f" | sha256sum -c -
  put "$B/crispdm-run.056e207a" "$BIN"; put "$B/crispdm_admission.py.7882d20f" "$LIB"; sha256sum "$BIN" "$LIB"; exit 0
fi
R=$HOME/Documents/GitHub/predictor
git -C "$R" fetch -q origin satoshi/m06-admission-dead-cache-20260930
T=$(mktemp -d "$HOME/.local/state/adm-deploy.XXXXXX")
git -C "$R" show a1f3898b:tools/crispdm-run > "$T/crispdm-run"
git -C "$R" show a1f3898b:tools/crispdm_admission.py > "$T/crispdm_admission.py"
echo "$NEW_RUN  $T/crispdm-run" | sha256sum -c -; echo "$NEW_ADM  $T/crispdm_admission.py" | sha256sum -c -
echo "$OLD_RUN  $BIN" | sha256sum -c -; echo "$OLD_ADM  $LIB" | sha256sum -c -     # refuse unless 775c5545 is what is installed
[ -f "$B/crispdm-run.056e207a" ] || cp -p "$BIN" "$B/crispdm-run.056e207a"
[ -f "$B/crispdm_admission.py.7882d20f" ] || cp -p "$LIB" "$B/crispdm_admission.py.7882d20f"
echo "$OLD_RUN  $B/crispdm-run.056e207a" | sha256sum -c -; echo "$OLD_ADM  $B/crispdm_admission.py.7882d20f" | sha256sum -c -
put "$T/crispdm_admission.py" "$LIB"; put "$T/crispdm-run" "$BIN"; rm -rf "$T"
sha256sum "$BIN" "$LIB"
"$BIN" -m 256M -t 2m -n deploy-smoke-adm-a1f3898b -- true
grep -E '"name": "deploy-smoke-adm-a1f3898b"' "$HOME/.local/state/crispdm/admission/ledger.jsonl" | tail -2 | cut -c1-260
python3 "$LIB" cancel --name deploy-smoke-nonexistent --witness-pid 1 | head -c 300; echo
REMOTE
done
