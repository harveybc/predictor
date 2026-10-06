#!/usr/bin/env bash
# Coordinator relay for the phase 2/3 selection campaign (fs-phase23-relay.service; separate
# from the live PS3-R relay, which is untouched). Workers cannot reach each other; this loop is
# the only channel, and the driver's terminal mirrors are "the integration agent's transport"
# (docs/FS_PHASE23_DRIVER.md). Every cycle, for each worker role and each population:
#   1. pull <pop>/terminals (immutable <unit>.json.gz + .meta.json), failures/, claims/,
#      quarantine/ and the host STATUS.json into <coordinator>/peer_terminals/<role>/<pop>/...
#      (the follower reads peer_terminals/<role>/<pop>/terminals; status reads the rest);
#   2. push <pop>/PLAN.json + MANIFEST.json one way, coordinator -> worker (immutable once
#      generated; a worker without them waits);
#   3. push every OTHER host's terminals into the worker's <pop>/terminals/ (the coordinator's
#      own and the other worker's mirrored ones; only *.json.gz/*.meta.json, never deleting) so
#      the driver adopts them instead of recomputing when it steals;
#   4. heartbeat with a monotonic cycle counter (RELAY_HEARTBEAT.json) to every host.
# SSH aliases live OUTSIDE the repository in $FS23_STATE/hosts.env (WORKER_A_SSH, WORKER_B_SSH);
# the state directory has the same relative path on every host (REMOTE_STATE).
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_STATE:?}"
# shellcheck disable=SC1091
source "$FS23_STATE/hosts.env"
: "${WORKER_A_SSH:?}" "${WORKER_B_SSH:?}"
FS23_POPULATIONS="${FS23_POPULATIONS:-eurusd eth}"
REMOTE_STATE=".local/state/canonical_20261003/fs_phase23"
PERIOD="${FS23_RELAY_PERIOD:-60}"
CYCLE_FILE="$FS23_STATE/relay_cycle"; HB="$FS23_STATE/RELAY_HEARTBEAT.json"; STATUS="$FS23_STATE/RELAY_STATUS.json"
mkdir -p "$FS23_STATE/peer_terminals" "$FS23_STATE/peer_status" "$FS23_STATE/logs"
[[ -f "$CYCLE_FILE" ]] || echo 0 > "$CYCLE_FILE"
RS=(rsync -a --timeout=300 --prune-empty-dirs)
TERM_FILTER=(--include='*.json.gz' --include='*.meta.json' --exclude='*')
declare -A SSH=([worker_a]="$WORKER_A_SSH" [worker_b]="$WORKER_B_SSH")

while true; do
  cycle=$(( $(cat "$CYCLE_FILE") + 1 )); started="$(date -u +%FT%TZ)"
  declare -A rc=()
  for role in worker_a worker_b; do
    h="${SSH[$role]}"
    for pop in $FS23_POPULATIONS; do
      m="$FS23_STATE/peer_terminals/$role/$pop"; mkdir -p "$m/terminals" "$m/failures" "$m/claims" "$m/quarantine"
      "${RS[@]}" "${TERM_FILTER[@]}" "$h:$REMOTE_STATE/$pop/terminals/" "$m/terminals/" 2>/dev/null; rc[pull_terminals_${role}_$pop]=$?
      for sub in failures claims quarantine; do
        rsync -a --timeout=60 --exclude='.*' "$h:$REMOTE_STATE/$pop/$sub/" "$m/$sub/" 2>/dev/null; rc[pull_${sub}_${role}_$pop]=$?
      done
    done
    rsync -a --timeout=30 "$h:$REMOTE_STATE/STATUS.json" "$FS23_STATE/peer_status/$role.STATUS.json" 2>/dev/null; rc[pull_status_$role]=$?
    rsync -a --timeout=30 --include='worker_*.log' --include='run_worker_*.log' --exclude='*' "$h:$REMOTE_STATE/logs/" "$FS23_STATE/peer_status/logs_$role/" 2>/dev/null || true
  done
  for role in worker_a worker_b; do
    h="${SSH[$role]}"; other=$([[ $role == worker_a ]] && echo worker_b || echo worker_a)
    for pop in $FS23_POPULATIONS; do
      ssh -o ConnectTimeout=20 "$h" "mkdir -p $REMOTE_STATE/$pop/terminals" 2>/dev/null
      if [[ -f "$FS23_STATE/$pop/PLAN.json" ]]; then
        rsync -a --timeout=60 "$FS23_STATE/$pop/PLAN.json" "$FS23_STATE/$pop/MANIFEST.json" "$h:$REMOTE_STATE/$pop/"; rc[push_plan_${role}_$pop]=$?
      else rc[push_plan_${role}_$pop]=0; fi
      "${RS[@]}" "${TERM_FILTER[@]}" "$FS23_STATE/$pop/terminals/" "$h:$REMOTE_STATE/$pop/terminals/" 2>/dev/null; rc[push_own_terminals_${role}_$pop]=$?
      if [[ -d "$FS23_STATE/peer_terminals/$other/$pop/terminals" ]]; then
        "${RS[@]}" "${TERM_FILTER[@]}" "$FS23_STATE/peer_terminals/$other/$pop/terminals/" "$h:$REMOTE_STATE/$pop/terminals/" 2>/dev/null; rc[push_peer_terminals_${role}_$pop]=$?
      else rc[push_peer_terminals_${role}_$pop]=0; fi
    done
  done
  ok=true; for k in "${!rc[@]}"; do case $k in pull_status_*) ;; *) [[ "${rc[$k]}" -eq 0 ]] || ok=false ;; esac; done
  echo "$cycle" > "$CYCLE_FILE"
  printf '{"schema":"fs_phase23_relay_heartbeat.v1","cycle":%d,"cycle_started_at_utc":"%s","relayed_at_utc":"%s","pushes_ok":%s,"period_seconds":%d}\n' \
    "$cycle" "$started" "$(date -u +%FT%TZ)" "$ok" "$PERIOD" > "$HB.tmp" && mv -f "$HB.tmp" "$HB"
  for role in worker_a worker_b; do rsync -a --timeout=30 "$HB" "${SSH[$role]}:$REMOTE_STATE/RELAY_HEARTBEAT.json" 2>/dev/null; rc[hb_$role]=$?; done
  {
    printf '{"schema":"fs_phase23_relay_status.v1","cycle":%d,"started_at":"%s","finished_at":"%s","pushes_ok":%s,"rc":{' "$cycle" "$started" "$(date -u +%FT%TZ)" "$ok"
    first=1; for k in $(printf '%s\n' "${!rc[@]}" | sort); do [[ $first -eq 1 ]] || printf ','; first=0; printf '"%s":%d' "$k" "${rc[$k]}"; done
    printf '}}\n'
  } > "$STATUS.tmp" && mv -f "$STATUS.tmp" "$STATUS"
  sleep "$PERIOD"
done
