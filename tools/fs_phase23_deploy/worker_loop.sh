#!/usr/bin/env bash
# One worker SLOT of the phase 2/3 selection campaign on one host (§E.5-E.8, plan §5), over
# the real driver (docs/FS_PHASE23_DRIVER.md).
#
#   worker_loop.sh SLOT        the durable loop fs-phase23-worker@SLOT runs
#
# The driver's `run-worker` walks every shard the plan assigned to this host id (the ROLE),
# in one process, with its own exclusive per-unit claims under <STATE>/<pop>/claims (several
# slots on one host share them), writes immutable terminals <STATE>/<pop>/terminals/<unit>.json.gz,
# isolates a defective shard under failures/ and quarantines corrupt terminals. With --steal
# (workers only) it then takes shards of other hosts that have no terminal visible here; the
# relay pushes every peer terminal into this host's terminals/ so stolen work is adopted, not
# repeated. This loop only makes that durable and capped: each population pass runs under
# crispdm-run with FS23_CAP / FS23_WALL and FS23_THREADS numeric threads; a pass that fails
# exits non-zero so systemd restarts it after RestartSec; exit 75 (admission refused, terminal)
# stops the slot without restart; when every shard of every population has a terminal here,
# the slot exits 13 (success). Nothing is polled by an agent.
#
# runner.env (untracked, absolute paths, written by deploy_host_step.sh): FS23_ROLE FS23_STATE
# FS23_CODE FS23_PYTHON FS23_DATA FS23_POPULATIONS FS23_CAP FS23_WALL FS23_THREADS FS23_STEAL
# FS23_ADMIT_WAIT FS23_IDLE_SLEEP FS23_CODE_COMMIT
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_ROLE:?}" "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}" "${FS23_DATA:?}"
FS23_POPULATIONS="${FS23_POPULATIONS:-eurusd eth}"
FS23_CAP="${FS23_CAP:-2G}"; FS23_WALL="${FS23_WALL:-12h}"; FS23_THREADS="${FS23_THREADS:-1}"
FS23_STEAL="${FS23_STEAL:-1}"; FS23_ADMIT_WAIT="${FS23_ADMIT_WAIT:-86400}"; FS23_IDLE_SLEEP="${FS23_IDLE_SLEEP:-300}"
SLOT="${1:?slot}"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$FS23_THREADS" OPENBLAS_NUM_THREADS="$FS23_THREADS" MKL_NUM_THREADS="$FS23_THREADS" NUMEXPR_NUM_THREADS="$FS23_THREADS"
LOGS="$FS23_STATE/logs"; mkdir -p "$LOGS"
LOG="$LOGS/worker_${SLOT}.log"
log() { printf '%s %s slot=%s %s\n' "$(date -u +%FT%TZ)" "$FS23_ROLE" "$SLOT" "$*" >> "$LOG"; }
steal_flag=(); [[ "$FS23_STEAL" == "1" ]] && steal_flag=(--steal)

# every shard of a population terminal here? (own + relayed peer terminals land in terminals/)
population_done() {
  "$FS23_PYTHON" - "$FS23_STATE/$1/PLAN.json" "$FS23_STATE/$1/terminals" <<'PY'
import json, pathlib, sys
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
tdir = pathlib.Path(sys.argv[2])
missing = [s["unit_id"] for s in plan["shards"] if not (tdir / f"{s['unit_id']}.json.gz").exists()]
print(len(missing)); sys.exit(0 if not missing else 1)
PY
}

# background + wait so a `systemctl stop` (SIGTERM) reaches the crispdm-run child, whose scope is outside the unit cgroup
child=""
forward() { [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null; }
trap forward TERM INT
log "BEGIN loop cap=$FS23_CAP wall=$FS23_WALL threads=$FS23_THREADS steal=$FS23_STEAL commit=${FS23_CODE_COMMIT:-?} populations=[$FS23_POPULATIONS]"
while true; do
  all_done=1
  for pop in $FS23_POPULATIONS; do
    plan="$FS23_STATE/$pop/PLAN.json"
    if [[ ! -f "$plan" || ! -f "$FS23_STATE/$pop/MANIFEST.json" ]]; then
      log "WAIT_PLAN $pop (plan/manifest not relayed yet)"; all_done=0; continue
    fi
    if missing=$(population_done "$pop"); then
      continue
    fi
    all_done=0
    log "RUN $pop missing_terminals=$missing"
    start=$(date +%s)
    "$HOME/.local/bin/crispdm-run" -q -W "$FS23_ADMIT_WAIT" -m "$FS23_CAP" -t "$FS23_WALL" -n "fs23-${pop}-${FS23_ROLE}-${SLOT}" \
        -L "fs23:$pop:slot$SLOT" -- bash -c "cd '$FS23_CODE' && exec '$FS23_PYTHON' tools/feature_pairwise_campaign.py run-worker \
          --plan '$plan' --state-root '$FS23_STATE/$pop' --data-root '$FS23_DATA' --host-id '$FS23_ROLE' --threads '$FS23_THREADS' ${steal_flag[*]:-}" \
        >> "$LOGS/run_worker_${pop}_slot${SLOT}.log" 2>&1 &
    child=$!
    wait "$child"; rc=$?
    wall=$(( $(date +%s) - start ))
    if [[ $rc -eq 75 ]]; then log "ADMISSION_REFUSED $pop (exit 75, slot stops, no restart)"; exit 75; fi
    # 137 = the cgroup ceiling (FS23_CAP) or the admission monitor stopped the pass: terminal for this
    # slot (a restart would hit the same cap every RestartSec); the other hosts steal its shards
    if [[ $rc -eq 137 ]]; then log "OOM_OR_PRESSURE_KILLED $pop under cap=$FS23_CAP wall=${wall}s (exit 75: slot stops, no restart; peers steal)"; exit 75; fi
    if [[ $rc -ne 0 ]]; then log "PASS_FAILED $pop rc=$rc wall=${wall}s (systemd restarts after RestartSec)"; exit "$rc"; fi
    log "PASS_OK $pop wall=${wall}s"
  done
  if [[ $all_done -eq 1 ]]; then log "ALL_DONE every shard of every population has a terminal here; slot exits 13"; exit 13; fi
  sleep "$FS23_IDLE_SLEEP"
done
