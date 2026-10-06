#!/usr/bin/env bash
# One worker SLOT of the phase 2/3 selection campaign on one host (§E.5-E.8, plan §5).
#
#   worker_loop.sh SLOT                 the durable loop (what fs-phase23-worker@SLOT runs)
#   worker_loop.sh --run-shard SHARD    the bounded computation of ONE shard (what the loop
#                                       hands to crispdm-run; the only place that knows the
#                                       driver's command line)
#
# Loop: claim the next free shard this role owns (shard_claims.py next; steals a straggler of
# another role only when FS23_STEAL=1 and every own shard is done or taken), settle the claim,
# mark RUNNING, run the shard under crispdm-run with the cap/wall/threads of runner.env, then
# mark COMPLETED (only if the terminal verifies) or FAILED with the exact exit code and the log
# tail. Nothing is retried blindly: a failed shard stays FAILED in its claim and carries a
# FAILED.json next to its terminal; STATUS shows the reason; the other slots and hosts keep
# going. Exit 75 (admission refused, terminal) stops this slot without restart. Exit 13 (every
# shard of the plan terminal) stops the slot as a success.
#
# Configuration is runner.env (untracked, written by deploy.sh with absolute paths):
#   FS23_ROLE FS23_STATE FS23_CODE FS23_PYTHON FS23_PLAN FS23_ASSIGNMENT
#   FS23_CAP (crispdm-run -m) FS23_WALL (-t) FS23_THREADS FS23_STEAL FS23_STALE_AFTER
#   FS23_ADMIT_WAIT (-W) FS23_SETTLE_MAX_WAIT FS23_WORKER_ARGS FS23_CODE_COMMIT
set -uo pipefail

if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_ROLE:?}" "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}" "${FS23_PLAN:?}" "${FS23_ASSIGNMENT:?}"
FS23_CAP="${FS23_CAP:-2G}"; FS23_WALL="${FS23_WALL:-6h}"; FS23_THREADS="${FS23_THREADS:-2}"
FS23_STEAL="${FS23_STEAL:-1}"; FS23_STALE_AFTER="${FS23_STALE_AFTER:-14400}"
FS23_ADMIT_WAIT="${FS23_ADMIT_WAIT:-86400}"; FS23_SETTLE_MAX_WAIT="${FS23_SETTLE_MAX_WAIT:-1800}"
FS23_WORKER_ARGS="${FS23_WORKER_ARGS:-}"; FS23_CODE_COMMIT="${FS23_CODE_COMMIT:-unknown}"
DEPLOY="$FS23_CODE/tools/fs_phase23_deploy"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$FS23_THREADS" OPENBLAS_NUM_THREADS="$FS23_THREADS" MKL_NUM_THREADS="$FS23_THREADS" NUMEXPR_NUM_THREADS="$FS23_THREADS"

# ---- the bounded computation of one shard -------------------------------------------------
if [[ "${1:-}" == "--run-shard" ]]; then
  shard="${2:?shard}"
  out="$FS23_STATE/terminals/$shard"
  mkdir -p "$out"
  cd "$FS23_CODE" || exit 2
  # The driver (tools/feature_pairwise_campaign.py, engineering agent) computes ONE shard,
  # block by block, writes an atomic local terminal (run_manifest.json + results) and exits.
  # shellcheck disable=SC2086
  exec "$FS23_PYTHON" tools/feature_pairwise_campaign.py run-worker \
    --plan "$FS23_PLAN" --shard "$shard" --out-dir "$out" --threads "$FS23_THREADS" \
    --host-role "$FS23_ROLE" $FS23_WORKER_ARGS
fi

SLOT="${1:?slot}"
LOGS="$FS23_STATE/logs"; mkdir -p "$LOGS/shards"
LOG="$LOGS/worker_${SLOT}.log"
log() { printf '%s %s slot=%s %s\n' "$(date -u +%FT%TZ)" "$FS23_ROLE" "$SLOT" "$*" >> "$LOG"; }
claims() { "$FS23_PYTHON" "$DEPLOY/shard_claims.py" "$@" --state-dir "$FS23_STATE" --role "$FS23_ROLE"; }
steal_flag=(); [[ "$FS23_STEAL" == "1" ]] && steal_flag=(--steal)
extra="{\"code_commit\":\"$FS23_CODE_COMMIT\",\"cap\":\"$FS23_CAP\",\"threads\":$FS23_THREADS}"

log "BEGIN loop cap=$FS23_CAP wall=$FS23_WALL threads=$FS23_THREADS steal=$FS23_STEAL commit=$FS23_CODE_COMMIT"
while true; do
  shard="$(claims next --plan "$FS23_PLAN" --assignment "$FS23_ASSIGNMENT" --slot "$SLOT" "${steal_flag[@]}" \
            --stale-after-seconds "$FS23_STALE_AFTER" --extra "$extra" 2>>"$LOG")"; rc=$?
  case $rc in
    0) ;;
    13) log "ALL_DONE every shard of the plan is terminal; slot exits"; exit 0 ;;
    10) sleep 120; continue ;;
    *) log "CLAIM_ERROR rc=$rc"; exit 2 ;;
  esac
  log "CLAIMED $shard"

  # settle (own shards win at once; stolen ones wait for the relay gate)
  waited=0; won=0
  while :; do
    claims settle --shard "$shard" --stale-after-seconds "$FS23_STALE_AFTER" 2>>"$LOG"; src=$?
    if [[ $src -eq 0 ]]; then won=1; break; fi
    if [[ $src -eq 11 ]]; then log "LOST $shard to its home role"; break; fi
    if [[ $src -eq 12 ]]; then
      if (( waited >= FS23_SETTLE_MAX_WAIT )); then
        claims mark --shard "$shard" --state ABANDONED --extra '{"reason":"SETTLE_TIMEOUT_RELAY_STALLED"}' 2>>"$LOG"
        log "ABANDONED $shard settle timeout (relay heartbeat not advancing)"; break
      fi
      sleep 60; waited=$((waited + 60)); continue
    fi
    log "SETTLE_ERROR $shard rc=$src"; break
  done
  [[ $won -eq 1 ]] || continue

  claims mark --shard "$shard" --state RUNNING 2>>"$LOG"
  slog="$LOGS/shards/${shard}.log"
  start=$(date +%s)
  log "RUN $shard"
  "$HOME/.local/bin/crispdm-run" -q -W "$FS23_ADMIT_WAIT" -m "$FS23_CAP" -t "$FS23_WALL" -n "fs23-${FS23_ROLE}-${SLOT}" \
      -L "fs23:${shard}" -- bash "$DEPLOY/worker_loop.sh" --run-shard "$shard" >> "$slog" 2>&1
  rc=$?
  wall=$(( $(date +%s) - start ))
  if [[ $rc -eq 0 ]]; then
    if claims mark --shard "$shard" --state COMPLETED --extra "{\"wall_seconds\":$wall}" 2>>"$LOG"; then
      log "COMPLETED $shard wall=${wall}s"; continue
    fi
    rc=90; reason="TERMINAL_INVALID: driver exited 0 but run_manifest/results digest does not verify"
  elif [[ $rc -eq 75 ]]; then
    claims mark --shard "$shard" --state ABANDONED --extra '{"reason":"ADMISSION_REFUSED_TERMINAL"}' 2>>"$LOG"
    log "ADMISSION_REFUSED $shard; slot stops (exit 75, no restart)"; exit 75
  else
    reason="driver exit $rc"
  fi
  tail_txt="$(tail -n 5 "$slog" 2>/dev/null | tr '\n' ' ' | tr -d '"' | cut -c1-600)"
  marker="$FS23_STATE/terminals/$shard"; mkdir -p "$marker"
  printf '{"schema":"fs_phase23_failed_marker.v1","shard_id":"%s","host_role":"%s","slot":"%s","exit_code":%d,"reason":"%s","wall_seconds":%d,"log_tail":"%s","failed_at_utc":"%s"}\n' \
    "$shard" "$FS23_ROLE" "$SLOT" "$rc" "$reason" "$wall" "$tail_txt" "$(date -u +%FT%TZ)" > "$marker/FAILED.json"
  printf '%s\n' "$reason" "$tail_txt" > "$marker/FAILED.reason.txt"
  claims mark --shard "$shard" --state FAILED --extra "{\"exit_code\":$rc,\"reason\":\"$reason\",\"wall_seconds\":$wall}" 2>>"$LOG"
  log "FAILED $shard rc=$rc wall=${wall}s"
done
