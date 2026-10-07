#!/usr/bin/env bash
# One status tick (fs-phase23-status.timer fires it every minute on every host): runs the
# driver's status module once over every population's plan, this host's state roots and --
# on the coordinator -- the relayed peer terminal mirrors, writing <STATE>/STATUS.json with
# expected / complete / failed / active / pending, rate and ETA per population x host (its
# schema is the driver's). The coordinator's follower also writes <STATE>/<pop>/STATUS.json
# per cycle; this file is the host-wide one the relay collects.
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_ROLE:?}" "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}"
FS23_POPULATIONS="${FS23_POPULATIONS:-eurusd eth}"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$FS23_STATE/logs"
args=()
for pop in $FS23_POPULATIONS; do
  [[ -f "$FS23_STATE/$pop/PLAN.json" ]] || continue
  args+=(--plan "$FS23_STATE/$pop/PLAN.json" --state-root "$FS23_STATE/$pop")
  if [[ "$FS23_ROLE" == "coordinator" ]]; then
    for role in worker_a worker_b; do
      d="$FS23_STATE/peer_terminals/$role/$pop"
      [[ -d "$d/terminals" ]] && args+=(--terminals "$d/terminals")
      [[ -d "$d" ]] && args+=(--state-root "$d")
    done
  fi
done
if [[ ${#args[@]} -eq 0 ]]; then
  printf '{"schema":"fs_phase23_status.no_plan.v1","host_role":"%s","written_at_utc":"%s","note":"no PLAN.json relayed to this host yet"}\n' \
    "$FS23_ROLE" "$(date -u +%FT%TZ)" > "$FS23_STATE/STATUS.json.tmp" && mv -f "$FS23_STATE/STATUS.json.tmp" "$FS23_STATE/STATUS.json"
  exit 0
fi
cd "$FS23_CODE" || exit 2
exec "$FS23_PYTHON" tools/feature_selection_phase23_status.py "${args[@]}" --out "$FS23_STATE/STATUS.json" 2>>"$FS23_STATE/logs/status.log"
