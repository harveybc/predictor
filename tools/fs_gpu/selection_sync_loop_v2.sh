#!/usr/bin/env bash
# Coordinator relay for the PS3-R GPU queues (FS-GPU, replaces selection_sync_loop.sh v1).
#
# Workers cannot reach each other; this loop is the only channel between them. Each cycle:
#   1. pull terminals (run_manifest.json, results.jsonl, FAILED*) from both workers into the
#      mirror (as v1 did), plus every driver log (BEGIN/END lines feed the ETA publisher);
#   2. pull both claims directories;
#   3. push each worker's claims to the other (only role-named files, never deleting);
#   4. push each worker's batch_002 terminals to the other as peer_terminals;
#   5. write a heartbeat with a monotonically increasing cycle counter to BOTH workers.
# The claim protocol's settle gate counts these cycles (see tools/fs_gpu/ps3r_claims.py);
# the order above (pull a, pull b, push a, push b, heartbeat) is what the gate relies on.
# SSH aliases live outside the repository in $STATE/fs_gpu/hosts.env (WORKER_A_SSH,
# WORKER_B_SSH); the repository only ever names roles.
set -uo pipefail

STATE="$HOME/.local/state/canonical_20261003"
DEST="$STATE/selection_mirror"
FS="$STATE/fs_gpu"
STATUS="$DEST/SYNC_STATUS.json"
CYCLE_FILE="$FS/relay_cycle"
PERIOD="${RELAY_PERIOD_SECONDS:-60}"
# shellcheck disable=SC1091
# hosts.env (outside the repo) defines the ssh aliases WORKER_A_SSH / WORKER_B_SSH and the
# pre-existing mirror/worker directory labels WORKER_A_LABEL / WORKER_B_LABEL (v1 layout).
source "$FS/hosts.env"
: "${WORKER_A_SSH:?}" "${WORKER_B_SSH:?}" "${WORKER_A_LABEL:?}" "${WORKER_B_LABEL:?}"

A_PS3R=".local/state/canonical_20261003/ps3r"
A_SUCC=".local/state/canonical_20261003/selection_successor"
B_ROOT=".local/state/canonical_20261003/ps3r/$WORKER_B_LABEL"
MIRROR_A_ALT="$DEST/$WORKER_A_LABEL"
MIRROR_A_BASE="$DEST/${WORKER_A_LABEL}_baseline"
MIRROR_B_BASE="$DEST/$WORKER_B_LABEL"

mkdir -p "$MIRROR_A_ALT" "$MIRROR_A_BASE" "$MIRROR_B_BASE" "$DEST/claims_worker_a" "$DEST/claims_worker_b" "$DEST/logs/worker_a" "$DEST/logs/worker_b" "$FS"
ln -sfn "$WORKER_A_LABEL" "$DEST/worker_a_alt"; ln -sfn "${WORKER_A_LABEL}_baseline" "$DEST/worker_a_base"; ln -sfn "$WORKER_B_LABEL" "$DEST/worker_b_base"
[[ -f "$CYCLE_FILE" ]] || echo 0 > "$CYCLE_FILE"

TERMINAL_FILTER=(--include='*/' --include='run_manifest.json' --include='results.jsonl' --include='FAILED.codex.json' --include='FAILED.json' --include='FAILED.reason.txt' --exclude='*')
RS=(rsync -a --prune-empty-dirs --timeout=120)

while true; do
  cycle=$(( $(cat "$CYCLE_FILE") + 1 ))
  started="$(date -u +%FT%TZ)"
  declare -A rc=()

  # 0. a one-line reason next to every FAILED marker (light grep on the worker; idempotent)
  REASON_CMD='for f in $(find "$1" -name "FAILED*.json" 2>/dev/null); do d=$(dirname "$f"); [ -f "$d/FAILED.reason.txt" ] || grep -hv "^I0000\|^WARNING\|^W0000\|^E0000" "$d"/stdout*.log 2>/dev/null | tail -3 > "$d/FAILED.reason.txt"; done'
  ssh -o ConnectTimeout=20 "$WORKER_A_SSH" "bash -c '$REASON_CMD' _ $A_PS3R/F; bash -c '$REASON_CMD' _ $A_SUCC/baseline" 2>/dev/null; rc[reason_a]=$?
  ssh -o ConnectTimeout=20 "$WORKER_B_SSH" "bash -c '$REASON_CMD' _ $B_ROOT/runs" 2>/dev/null; rc[reason_b]=$?

  # 1. terminals + logs (pull)
  "${RS[@]}" "${TERMINAL_FILTER[@]}" "$WORKER_A_SSH:$A_PS3R/F/" "$MIRROR_A_ALT/";                      rc[pull_a_alt]=$?
  "${RS[@]}" "${TERMINAL_FILTER[@]}" "$WORKER_A_SSH:$A_SUCC/baseline/" "$MIRROR_A_BASE/";     rc[pull_a_base]=$?
  "${RS[@]}" "${TERMINAL_FILTER[@]}" "$WORKER_B_SSH:$B_ROOT/runs/" "$MIRROR_B_BASE/";                 rc[pull_b_base]=$?
  rsync -a --timeout=60 --include='*.log' --exclude='*' "$WORKER_A_SSH:$A_PS3R/F/" "$DEST/logs/worker_a/alt/" 2>/dev/null; rc[pull_a_log_alt]=$?
  rsync -a --timeout=60 --include='*.log' --exclude='*' "$WORKER_A_SSH:$A_SUCC/" "$DEST/logs/worker_a/" 2>/dev/null;       rc[pull_a_log_succ]=$?
  rsync -a --timeout=60 --include='*.log' --exclude='*' "$WORKER_B_SSH:$B_ROOT/" "$DEST/logs/worker_b/" 2>/dev/null;       rc[pull_b_log]=$?

  # 2. claims (pull)
  rsync -a --timeout=60 --prune-empty-dirs --include='*/' --include='claim*.json' --exclude='*' "$WORKER_A_SSH:$A_PS3R/claims/" "$DEST/claims_worker_a/";   rc[pull_a_claims]=$?
  rsync -a --timeout=60 --prune-empty-dirs --include='*/' --include='claim*.json' --exclude='*' "$WORKER_B_SSH:$B_ROOT/claims/" "$DEST/claims_worker_b/";   rc[pull_b_claims]=$?

  # 3. claims (push, cross)   4. peer terminals (push, cross)
  rsync -a --timeout=60 --prune-empty-dirs --include='*/' --include='claim.worker_b.json' --exclude='*' "$DEST/claims_worker_b/" "$WORKER_A_SSH:$A_PS3R/claims/";  rc[push_a_claims]=$?
  "${RS[@]}" "${TERMINAL_FILTER[@]}" "$MIRROR_B_BASE/batch_002/" "$WORKER_A_SSH:$A_PS3R/peer_terminals/batch_002/";                                                   rc[push_a_terminals]=$?
  rsync -a --timeout=60 --prune-empty-dirs --include='*/' --include='claim.worker_a.json' --exclude='*' "$DEST/claims_worker_a/" "$WORKER_B_SSH:$B_ROOT/claims/";   rc[push_b_claims]=$?
  if [[ -d "$MIRROR_A_BASE/batch_002" ]]; then
    "${RS[@]}" "${TERMINAL_FILTER[@]}" "$MIRROR_A_BASE/batch_002/" "$WORKER_B_SSH:$B_ROOT/peer_terminals/batch_002/";                                        rc[push_b_terminals]=$?
  else
    rc[push_b_terminals]=0
  fi

  # 5. heartbeat (only after this cycle's pushes; cycle counter is monotonic across restarts)
  pushes_ok=true
  for k in push_a_claims push_a_terminals push_b_claims push_b_terminals pull_a_claims pull_b_claims; do
    [[ "${rc[$k]}" -eq 0 ]] || pushes_ok=false
  done
  echo "$cycle" > "$CYCLE_FILE"
  hb="$FS/RELAY_HEARTBEAT.json"
  printf '{"schema":"ps3r_relay_heartbeat.v1","cycle":%d,"cycle_started_at_utc":"%s","relayed_at_utc":"%s","pushes_ok":%s,"period_seconds":%d}\n' \
    "$cycle" "$started" "$(date -u +%FT%TZ)" "$pushes_ok" "$PERIOD" > "$hb"
  if $pushes_ok; then
    rsync -a --timeout=30 "$hb" "$WORKER_A_SSH:$A_PS3R/RELAY_HEARTBEAT.json"; rc[hb_a]=$?
    rsync -a --timeout=30 "$hb" "$WORKER_B_SSH:$B_ROOT/RELAY_HEARTBEAT.json"; rc[hb_b]=$?
  else
    rc[hb_a]=99; rc[hb_b]=99
  fi

  python3 - "$STATUS" "$started" "$cycle" "$pushes_ok" "$(for k in "${!rc[@]}"; do printf '%s=%s ' "$k" "${rc[$k]}"; done)" "$WORKER_A_LABEL" "$WORKER_B_LABEL" <<'PY'
import datetime, json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
rcs = dict(item.split("=") for item in sys.argv[5].split())
label_a, label_b = sys.argv[6], sys.argv[7]
payload = {
    "schema": "feature_selection_sync.v2",
    "started_at": sys.argv[2],
    "finished_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cycle": int(sys.argv[3]),
    "pushes_ok": sys.argv[4] == "true",
    "rc": {k: int(v) for k, v in sorted(rcs.items())},
}
# v1 fields (<label>_rc) kept for existing consumers
payload[f"{label_a}_rc"] = max(int(rcs.get("pull_a_alt", 0)), int(rcs.get("pull_a_base", 0)))
payload[f"{label_b}_rc"] = int(rcs.get("pull_b_base", 0))
fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", text=True)
try:
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
finally:
    if os.path.exists(name):
        os.unlink(name)
PY
  sleep "$PERIOD"
done
