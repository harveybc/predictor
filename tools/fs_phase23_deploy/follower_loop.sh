#!/usr/bin/env bash
# Follower for ONE population (coordinator only; fs-phase23-follower@POP.service): the driver's
# `follow` adopts valid terminals (own + relayed peer mirrors), quarantines corrupt ones, submits
# rows to the warehouse, verifies the readback per receipt, closes phase 2 when every shard is
# receipted, chains run-phase3 -> close-phase3, and writes <STATE>/<pop>/STATUS.json each cycle
# (§E.7). `follow` is itself a durable loop (--every) that returns once phase 3 is closed; this
# wrapper runs it under crispdm-run with FS23_FOLLOW_CAP and lets systemd restart a failed pass.
#
#   follower_loop.sh POP            POP in {eurusd, eth}
#
# FS23_WAREHOUSE is the handle given to the driver: the DuckDB adapter file until the DATA
# agent's live module and route are deployed, then whatever tools/fs_phase23_warehouse.py
# accepts (runner.env is rewritten and the unit restarted; the follower is idempotent).
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}" "${FS23_DATA:?}" "${FS23_WAREHOUSE:?}"
POP="${1:?population}"
FS23_FOLLOW_CAP="${FS23_FOLLOW_CAP:-2G}"; FS23_FOLLOW_WALL="${FS23_FOLLOW_WALL:-7d}"
FS23_FOLLOW_EVERY="${FS23_FOLLOW_EVERY:-60}"; FS23_PHASE3_WORKERS="${FS23_PHASE3_WORKERS:-2}"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FS23_STATE/logs" "$(dirname "$FS23_WAREHOUSE")" 2>/dev/null
LOG="$FS23_STATE/logs/follower_${POP}.log"
log() { printf '%s follower %s %s\n' "$(date -u +%FT%TZ)" "$POP" "$*" >> "$LOG"; }
plan="$FS23_STATE/$POP/PLAN.json"
[[ -f "$plan" ]] || { log "NO_PLAN $plan"; exit 2; }
if [[ -f "$FS23_STATE/$POP/PHASE_3_FILTER_COMPLETE.json" ]]; then log "PHASE_3_FILTER_COMPLETE.json present; exit 0"; exit 0; fi
terminals=(--terminals "$FS23_STATE/$POP/terminals")
for role in worker_a worker_b; do terminals+=(--terminals "$FS23_STATE/peer_terminals/$role/$POP/terminals"); mkdir -p "$FS23_STATE/peer_terminals/$role/$POP/terminals"; done
log "BEGIN cap=$FS23_FOLLOW_CAP every=${FS23_FOLLOW_EVERY}s warehouse=$FS23_WAREHOUSE phase3_workers=$FS23_PHASE3_WORKERS"
# register the run with its expected counts (idempotent) so the store's reconcile can judge completeness
( cd "$FS23_CODE" && "$FS23_PYTHON" tools/fs_phase23_deploy/register_run.py --plan "$plan" --warehouse "$FS23_WAREHOUSE" ) >> "$LOG" 2>&1 || log "REGISTER_RUN_FAILED (see above; follow continues, the store registers on first submission)"
"$HOME/.local/bin/crispdm-run" -q -W 3600 -m "$FS23_FOLLOW_CAP" -t "$FS23_FOLLOW_WALL" -n "fs23-follower-$POP" -L "fs23:follow:$POP" -- \
  bash -c "cd '$FS23_CODE' && exec '$FS23_PYTHON' tools/feature_pairwise_campaign.py follow --plan '$plan' --state-root '$FS23_STATE/$POP' \
    $(printf "%q " "${terminals[@]}") --warehouse '$FS23_WAREHOUSE' --data-root '$FS23_DATA' --every '$FS23_FOLLOW_EVERY' --phase3-workers '$FS23_PHASE3_WORKERS'" \
  >> "$FS23_STATE/logs/follow_${POP}.log" 2>&1
rc=$?
if [[ $rc -eq 75 ]]; then log "ADMISSION_REFUSED (exit 75, no restart)"; exit 75; fi
if [[ $rc -ne 0 ]]; then log "FOLLOW_EXITED rc=$rc (systemd restarts after RestartSec)"; exit "$rc"; fi
log "FOLLOW_RETURNED 0 (phase 3 closed)"; exit 0
