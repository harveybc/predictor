#!/usr/bin/env bash
# Coordinator status loop (FS-GPU v2; replaces selection_status_loop.sh v1 and keeps every
# v1 output). Light work only: JSON over the mirror, nvidia-smi/free over ssh. Each minute:
#   v1  per-plan status JSONs (now telling the status tool the real FAILED marker names so a
#       FAILED*.json receipt counts as failed, not pending) + SELECTION_STATUS.json;
#   v2  GPU/RAM probes under ROLE names, claims ledger, ETA publisher (ps3r_eta.json +
#       ps3r_progress.csv), closure matrix and the live ingest of every authenticated terminal.
# SSH aliases come from $STATE/fs_gpu/hosts.env; the repository only names roles.
set -uo pipefail

REPO="$HOME/Documents/GitHub/.worktrees/predictor-canonical-exec-20261003"
STATE="$HOME/.local/state/canonical_20261003"
FS_STATE="$STATE/fs_gpu"
AUTO="$REPO/docs/audits/evidence/canonical_20261003/automation"
FS="$REPO/docs/audits/evidence/canonical_20261003/fs_closure/fs_gpu"
PYTHON="$HOME/anaconda3/bin/python"
PERIOD="${STATUS_PERIOD_SECONDS:-60}"
# shellcheck disable=SC1091
source "$FS_STATE/hosts.env"   # ssh aliases + v1 directory/file labels, never in the repo
: "${WORKER_A_SSH:?}" "${WORKER_B_SSH:?}" "${WORKER_A_LABEL:?}" "${WORKER_B_LABEL:?}" "${COORD_LABEL:?}"
mkdir -p "$FS_STATE" "$FS"

GPU_Q='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
MEM_Q='free -m | awk "/^Mem:/{printf \"total_mib=%s used_mib=%s available_mib=%s\", \$2, \$3, \$7}"'

while true; do
  # ---- v1 outputs (unchanged consumers) --------------------------------------------------
  "$PYTHON" "$REPO/tools/feature_selection_status.py" \
    --plan "$AUTO/alternative_5090.tsv" --results-root "$STATE/selection_mirror/$WORKER_A_LABEL" --workers 1 \
    --failed-name FAILED.codex.json --output "$STATE/alternative_5090_STATUS.json" >/dev/null
  "$PYTHON" "$REPO/tools/feature_selection_status.py" \
    --plan "$AUTO/baseline_batch_002_4090.tsv" --results-root "$STATE/selection_mirror/$WORKER_B_LABEL" --workers 1 \
    --failed-name FAILED.codex.json --output "$STATE/baseline_4090_STATUS.json" >/dev/null
  "$PYTHON" "$REPO/tools/feature_selection_status.py" \
    --plan "$AUTO/baseline_batch_001_003_5090.tsv" --results-root "$STATE/selection_mirror/${WORKER_A_LABEL}_baseline" --workers 1 \
    --failed-name FAILED.json --output "$STATE/baseline_successor_5090_STATUS.json" >/dev/null
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader > "$STATE/${COORD_LABEL}_GPU.csv"
  ssh -o ConnectTimeout=20 "$WORKER_A_SSH" "$GPU_Q" > "$STATE/${WORKER_A_LABEL}_GPU.csv.tmp" 2>/dev/null && mv "$STATE/${WORKER_A_LABEL}_GPU.csv.tmp" "$STATE/${WORKER_A_LABEL}_GPU.csv"
  ssh -o ConnectTimeout=20 "$WORKER_B_SSH" "$GPU_Q" > "$STATE/${WORKER_B_LABEL}_GPU.csv.tmp" 2>/dev/null && mv "$STATE/${WORKER_B_LABEL}_GPU.csv.tmp" "$STATE/${WORKER_B_LABEL}_GPU.csv"
  "$PYTHON" - "$STATE" "$COORD_LABEL" "$WORKER_A_LABEL" "$WORKER_B_LABEL" <<'PY'
import datetime, json, os, pathlib, tempfile, sys
root = pathlib.Path(sys.argv[1])
labels = sys.argv[2:5]
names = {"alternative_5090": "alternative_5090_STATUS.json", "baseline_4090": "baseline_4090_STATUS.json", "baseline_successor_5090": "baseline_successor_5090_STATUS.json", "ps4": "selection_ps4/STATUS.json"}
lanes = {}
for key, name in names.items():
    try:
        lanes[key] = json.loads((root / name).read_text())
    except (OSError, json.JSONDecodeError) as error:
        lanes[key] = {"error": f"{type(error).__name__}: {error}"}
payload = {"schema": "feature_selection_live_status.v1", "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "denominator": 366, "heavy_candidate_features": 137, "lanes": lanes,
           "gpus": {host: (root / f"{host}_GPU.csv").read_text().splitlines() if (root / f"{host}_GPU.csv").exists() else [] for host in labels}}
destination = root / "SELECTION_STATUS.json"
fd, temporary = tempfile.mkstemp(dir=root, prefix=destination.name + ".", text=True)
try:
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

  # ---- v2: role-named probes ---------------------------------------------------------------
  cp -f "$STATE/${WORKER_A_LABEL}_GPU.csv" "$FS_STATE/worker_a_GPU.csv" 2>/dev/null
  cp -f "$STATE/${WORKER_B_LABEL}_GPU.csv" "$FS_STATE/worker_b_GPU.csv" 2>/dev/null
  cp -f "$STATE/${COORD_LABEL}_GPU.csv" "$FS_STATE/coordinator_GPU.csv" 2>/dev/null
  ssh -o ConnectTimeout=20 "$WORKER_A_SSH" "$MEM_Q" > "$FS_STATE/worker_a_MEM.csv.tmp" 2>/dev/null && mv "$FS_STATE/worker_a_MEM.csv.tmp" "$FS_STATE/worker_a_MEM.csv"
  ssh -o ConnectTimeout=20 "$WORKER_B_SSH" "$MEM_Q" > "$FS_STATE/worker_b_MEM.csv.tmp" 2>/dev/null && mv "$FS_STATE/worker_b_MEM.csv.tmp" "$FS_STATE/worker_b_MEM.csv"
  bash -c "$MEM_Q" > "$FS_STATE/coordinator_MEM.csv"

  # ---- v2: claims ledger, ETA, closure matrix + live ingest --------------------------------
  "$PYTHON" "$REPO/tools/fs_gpu/ps3r_claims.py" ledger \
    --root "worker_a=$STATE/selection_mirror/claims_worker_a" \
    --root "worker_b=$STATE/selection_mirror/claims_worker_b" \
    --out "$FS/claims_ledger.json"
  "$PYTHON" "$REPO/tools/fs_gpu/ps3r_eta.py" --config "$FS/ps3r_eta_config.json" --base "$REPO" \
    --out-json "$FS/ps3r_eta.json" --out-csv "$FS/ps3r_progress.csv" --quiet
  "$PYTHON" "$REPO/tools/fs_gpu/ps3r_closure_matrix.py" --config "$FS/ps3r_closure_matrix_config.json" --base "$REPO" \
    --out-json "$FS/closure_matrix.json" --out-csv "$FS/closure_matrix.csv" \
    --ingest --ingest-out "$FS/ps3r_ingest_live.json" --quiet
  date -u +%FT%TZ > "$FS_STATE/status_loop_last_cycle_utc"
  sleep "$PERIOD"
done
