#!/usr/bin/env bash
# One status tick (fs-phase23-status.timer fires it every minute on every host).
#
# 1. Writes CLAIMS_LEDGER.json from this host's own claims + relayed peer claims and own +
#    relayed terminals (shard_claims.py ledger): expected / terminal / live / free counts and
#    any contested shard. Cheap and stdlib-only, so it runs even before the driver exists.
# 2. If the driver's status module exists (tools/feature_selection_phase23_status.py,
#    engineering agent), runs it to produce STATUS.json (expected, complete, failed, active,
#    speed, ETA from evidence). Until then STATUS.json is derived from the ledger so the file
#    still updates every minute with a schema that names its source.
set -uo pipefail
if [[ -z "${FS23_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.local/state/canonical_20261003/fs_phase23/runner.env"
fi
: "${FS23_ROLE:?}" "${FS23_STATE:?}" "${FS23_CODE:?}" "${FS23_PYTHON:?}"
DEPLOY="$FS23_CODE/tools/fs_phase23_deploy"
export PYTHONPATH="$FS23_CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PLAN="${FS23_PLAN:-$FS23_STATE/plan/plan.json}"; ASSIGN="${FS23_ASSIGNMENT:-$FS23_STATE/assignment.json}"
mkdir -p "$FS23_STATE/logs"

if [[ -f "$PLAN" && -f "$ASSIGN" ]]; then
  "$FS23_PYTHON" "$DEPLOY/shard_claims.py" ledger --plan "$PLAN" --assignment "$ASSIGN" \
    --claims "own=$FS23_STATE/claims" --claims "peer=$FS23_STATE/peer_claims" \
    --terminals "own=$FS23_STATE/terminals" --terminals "peer=$FS23_STATE/peer_terminals" \
    --out "$FS23_STATE/CLAIMS_LEDGER.json" 2>>"$FS23_STATE/logs/status.log" || true
fi

STATUS_PY="$FS23_CODE/tools/feature_selection_phase23_status.py"
if [[ -f "$STATUS_PY" ]]; then
  # shellcheck disable=SC2086
  "$FS23_PYTHON" "$STATUS_PY" --state-dir "$FS23_STATE" --plan "$PLAN" --assignment "$ASSIGN" \
    --host-role "$FS23_ROLE" --out "$FS23_STATE/STATUS.json" ${FS23_STATUS_ARGS:-} 2>>"$FS23_STATE/logs/status.log"
  exit $?
fi

"$FS23_PYTHON" - "$FS23_STATE" "$FS23_ROLE" <<'PY'
import json, os, pathlib, sys, tempfile, time
state, role = pathlib.Path(sys.argv[1]), sys.argv[2]
led = {}
try:
    led = json.loads((state / "CLAIMS_LEDGER.json").read_text())
except (OSError, ValueError):
    pass
counts = led.get("counts", {})
expected = sum(counts.values())
doc = {
    "schema": "fs_phase23_status.fallback_from_ledger.v1",
    "host_role": role,
    "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source": "CLAIMS_LEDGER.json (driver status module not installed yet)",
    "expected_shards": expected,
    "complete_shards": counts.get("terminal:COMPLETED", 0),
    "failed_shards": counts.get("terminal:FAILED", 0),
    "active_shards": counts.get("live", 0),
    "free_shards": counts.get("free", 0),
    "contested_shards": led.get("contested", []),
    "speed_shards_per_hour": None,
    "eta_utc": None,
}
fd, tmp = tempfile.mkstemp(dir=str(state), prefix=".STATUS.json.")
with os.fdopen(fd, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, state / "STATUS.json")
PY
