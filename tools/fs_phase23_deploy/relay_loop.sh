#!/usr/bin/env bash
# Coordinator relay for the phase 2/3 selection campaign (fs-phase23-relay.service; the
# equivalent of selection_sync_loop_v2.sh for this campaign, kept separate so the live PS3-R
# relay is untouched). Workers cannot reach each other; this loop is the only channel. Every
# cycle, for each worker role:
#   1. pull its claims  -> <coordinator state>/peer_claims/   (claim.<role>.json only)
#   2. pull its terminals and STATUS.json -> peer_terminals/ and peer_status/<role>.STATUS.json
#   3. push the plan and assignment (one way, coordinator -> worker; immutable once generated)
#   4. push every OTHER role's claims (the coordinator's own + the other worker's relayed ones)
#      into the worker's peer_claims/ -- only role-named files, never deleting
#   5. write a heartbeat with a monotonically increasing cycle counter to every host AFTER the
#      pushes (shard_claims.py settle counts these cycles: pull a, pull b, push a, push b,
#      heartbeat is the order its gate relies on)
# Workers only ever receive claims, never each other's results: a relayed COMPLETED claim is
# what tells a worker a shard is terminal elsewhere. The follower on the coordinator loads the
# pulled terminals into the warehouse.
#
# SSH aliases live OUTSIDE the repository in $FS23_STATE/hosts.env (WORKER_A_SSH, WORKER_B_SSH);
# the state directory has the same relative path on every host (REMOTE_STATE below).
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_STATE:?}"
# shellcheck disable=SC1091
source "$FS23_STATE/hosts.env"
: "${WORKER_A_SSH:?}" "${WORKER_B_SSH:?}"
REMOTE_STATE=".local/state/canonical_20261003/fs_phase23"
PERIOD="${FS23_RELAY_PERIOD:-60}"
CYCLE_FILE="$FS23_STATE/relay_cycle"
HB="$FS23_STATE/RELAY_HEARTBEAT.json"
STATUS="$FS23_STATE/RELAY_STATUS.json"
mkdir -p "$FS23_STATE/peer_claims" "$FS23_STATE/peer_terminals" "$FS23_STATE/peer_status" "$FS23_STATE/logs"
[[ -f "$CYCLE_FILE" ]] || echo 0 > "$CYCLE_FILE"

RS=(rsync -a --timeout=120 --prune-empty-dirs)
CLAIM_FILTER=(--include='*/' --include='claim.*.json' --exclude='*')
TERMINAL_FILTER=(--include='*/' --exclude='.*' --include='*')   # everything except atomic-write temp files
declare -A SSH=([worker_a]="$WORKER_A_SSH" [worker_b]="$WORKER_B_SSH")

while true; do
  cycle=$(( $(cat "$CYCLE_FILE") + 1 ))
  started="$(date -u +%FT%TZ)"
  declare -A rc=()
  for role in worker_a worker_b; do
    h="${SSH[$role]}"
    "${RS[@]}" "${CLAIM_FILTER[@]}" "$h:$REMOTE_STATE/claims/" "$FS23_STATE/peer_claims/";        rc[pull_claims_$role]=$?
    "${RS[@]}" "${TERMINAL_FILTER[@]}" "$h:$REMOTE_STATE/terminals/" "$FS23_STATE/peer_terminals/"; rc[pull_terminals_$role]=$?
    rsync -a --timeout=30 "$h:$REMOTE_STATE/STATUS.json" "$FS23_STATE/peer_status/$role.STATUS.json" 2>/dev/null; rc[pull_status_$role]=$?
    rsync -a --timeout=30 "$h:$REMOTE_STATE/CLAIMS_LEDGER.json" "$FS23_STATE/peer_status/$role.CLAIMS_LEDGER.json" 2>/dev/null || true
  done
  for role in worker_a worker_b; do
    h="${SSH[$role]}"
    ssh -o ConnectTimeout=20 "$h" "mkdir -p $REMOTE_STATE/peer_claims $REMOTE_STATE/plan" 2>/dev/null
    if [[ -d "$FS23_STATE/plan" ]]; then
      "${RS[@]}" "$FS23_STATE/plan/" "$h:$REMOTE_STATE/plan/";                                     rc[push_plan_$role]=$?
    else rc[push_plan_$role]=0; fi
    if [[ -f "$FS23_STATE/assignment.json" ]]; then
      rsync -a --timeout=30 "$FS23_STATE/assignment.json" "$h:$REMOTE_STATE/assignment.json";       rc[push_assignment_$role]=$?
    else rc[push_assignment_$role]=0; fi
    # own claims + the other worker's relayed claims (never this worker's own back to itself)
    other=$([[ $role == worker_a ]] && echo worker_b || echo worker_a)
    "${RS[@]}" --include='*/' --include='claim.coordinator.json' --exclude='*' "$FS23_STATE/claims/" "$h:$REMOTE_STATE/peer_claims/";  rc[push_own_claims_$role]=$?
    "${RS[@]}" --include='*/' --include="claim.$other.json" --exclude='*' "$FS23_STATE/peer_claims/" "$h:$REMOTE_STATE/peer_claims/"; rc[push_peer_claims_$role]=$?
  done
  pushes_ok=true
  for k in "${!rc[@]}"; do
    case $k in pull_status_*) ;; *) [[ "${rc[$k]}" -eq 0 ]] || pushes_ok=false ;; esac
  done
  echo "$cycle" > "$CYCLE_FILE"
  printf '{"schema":"fs_phase23_relay_heartbeat.v1","cycle":%d,"cycle_started_at_utc":"%s","relayed_at_utc":"%s","pushes_ok":%s,"period_seconds":%d}\n' \
    "$cycle" "$started" "$(date -u +%FT%TZ)" "$pushes_ok" "$PERIOD" > "$HB.tmp" && mv -f "$HB.tmp" "$HB"
  if $pushes_ok; then
    for role in worker_a worker_b; do
      rsync -a --timeout=30 "$HB" "${SSH[$role]}:$REMOTE_STATE/RELAY_HEARTBEAT.json"; rc[hb_$role]=$?
    done
  else
    rc[hb_worker_a]=99; rc[hb_worker_b]=99
  fi
  {
    printf '{"schema":"fs_phase23_relay_status.v1","cycle":%d,"started_at":"%s","finished_at":"%s","pushes_ok":%s,"rc":{' "$cycle" "$started" "$(date -u +%FT%TZ)" "$pushes_ok"
    first=1; for k in $(printf '%s\n' "${!rc[@]}" | sort); do [[ $first -eq 1 ]] || printf ','; first=0; printf '"%s":%d' "$k" "${rc[$k]}"; done
    printf '}}\n'
  } > "$STATUS.tmp" && mv -f "$STATUS.tmp" "$STATUS"
  sleep "$PERIOD"
done
