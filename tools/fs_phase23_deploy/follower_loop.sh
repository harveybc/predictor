#!/usr/bin/env bash
# Follower (coordinator only; fs-phase23-follower.service): loads terminals into the warehouse,
# verifies readback, and chains phase 3 when phase 2 closes (§E.7).
#
# The driver's `follow` command (tools/feature_pairwise_campaign.py, engineering agent) does
# the work: it scans the terminal roots (this host's own terminals plus the relayed
# peer_terminals), submits every valid terminal it has not yet submitted, verifies the readback
# (counts and digests), and when the phase 2 gate passes runs close-phase2 -> run-phase3 ->
# close-phase3. This loop only makes it durable and memory-capped: every pass runs under
# crispdm-run with FS23_FOLLOW_CAP (coordinator default 1G, the owner's desktop rule), a pass
# that fails exits non-zero so systemd restarts it after RestartSec, and the loop ends with 0
# once PHASE_3_FILTER_COMPLETE.json exists in the campaign root. Nothing is polled by an agent.
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}" "${FS23_PLAN:?}"
FS23_FOLLOW_CAP="${FS23_FOLLOW_CAP:-1G}"; FS23_FOLLOW_WALL="${FS23_FOLLOW_WALL:-2h}"
FS23_FOLLOW_PERIOD="${FS23_FOLLOW_PERIOD:-60}"; FS23_FOLLOW_ARGS="${FS23_FOLLOW_ARGS:-}"
CAMPAIGN="${FS23_CAMPAIGN_ROOT:-$FS23_STATE/campaign}"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FS23_STATE/logs" "$CAMPAIGN"
LOG="$FS23_STATE/logs/follower.log"
log() { printf '%s follower %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

if [[ "${1:-}" == "--pass" ]]; then
  cd "$FS23_CODE" || exit 2
  # shellcheck disable=SC2086
  exec "$FS23_PYTHON" tools/feature_pairwise_campaign.py follow \
    --plan "$FS23_PLAN" --campaign-root "$CAMPAIGN" \
    --terminals-root "$FS23_STATE/terminals" --terminals-root "$FS23_STATE/peer_terminals" \
    --chain-phase3 $FS23_FOLLOW_ARGS
fi

log "BEGIN cap=$FS23_FOLLOW_CAP wall=$FS23_FOLLOW_WALL period=${FS23_FOLLOW_PERIOD}s"
while true; do
  if [[ -f "$CAMPAIGN/PHASE_3_FILTER_COMPLETE.json" ]]; then
    log "PHASE_3_FILTER_COMPLETE.json present; follower exits"; exit 0
  fi
  "$HOME/.local/bin/crispdm-run" -q -W 3600 -m "$FS23_FOLLOW_CAP" -t "$FS23_FOLLOW_WALL" -n fs23-follower \
     -L "fs23:follow" -- bash "$FS23_CODE/tools/fs_phase23_deploy/follower_loop.sh" --pass >> "$FS23_STATE/logs/follower_pass.log" 2>&1
  rc=$?
  if [[ $rc -eq 75 ]]; then log "ADMISSION_REFUSED (exit 75, no restart)"; exit 75; fi
  if [[ $rc -ne 0 ]]; then log "PASS_FAILED rc=$rc (systemd restarts after RestartSec)"; exit "$rc"; fi
  sleep "$FS23_FOLLOW_PERIOD"
done
