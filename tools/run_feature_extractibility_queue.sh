#!/usr/bin/env bash
set -euo pipefail

: "${WORKER_CODE:?WORKER_CODE is required}"
: "${PS2_ROOT:?PS2_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${PLAN:?PLAN is required}"
: "${GPU_UUID:?GPU_UUID is required}"
: "${MEMORY_CAP:?MEMORY_CAP is required}"
: "${FAMILIES:?FAMILIES is required}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/tensorflow/bin/python}"
CRISPDM_RUN="${CRISPDM_RUN:-$HOME/.local/bin/crispdm-run}"
LD_LIBRARY_PATH_FILE="${LD_LIBRARY_PATH_FILE:-$HOME/.local/state/scratch/m07/ld_library_path.txt}"
LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/worker.log}"
mkdir -p "$OUTPUT_ROOT"

valid_terminal() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
try:
    manifest = json.loads((root / "run_manifest.json").read_text())
    results = (root / "results.jsonl").read_bytes()
except (OSError, ValueError):
    raise SystemExit(1)
expected = manifest.get("results_sha256") or manifest.get("results_digest")
actual = hashlib.sha256(results).hexdigest()
raise SystemExit(0 if manifest.get("status") == "COMPLETED" and expected == actual else 1)
PY
}

echo "$(date -u +%FT%TZ) START gpu=$GPU_UUID families=$FAMILIES" >> "$LOG_FILE"
while IFS=$'\t' read -r stage batch feature; do
  [[ -n "$feature" ]] || continue
  out="$OUTPUT_ROOT/$batch/$feature"
  valid_terminal "$out" && continue
  [[ -f "$out/FAILED.json" ]] && continue
  mkdir -p "$out"
  echo "$(date -u +%FT%TZ) BEGIN $stage $batch $feature" >> "$LOG_FILE"
  rc=0
  "$CRISPDM_RUN" -q -W 86400 -m "$MEMORY_CAP" -t 8h \
    -n "selection_${feature//./_}" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
      LD_LIBRARY_PATH="$(cat "$LD_LIBRARY_PATH_FILE")" \
      bash -lc "cd '$WORKER_CODE' && exec '$PYTHON_BIN' -m app.univariate_temporal_pilot --batch_dir '$PS2_ROOT/$batch' --out_dir '$out' --features '$feature' --window 168 --latent_dim 8 --seed 0 --families '$FAMILIES' --batch_size 64 --max_fit_windows 16384 --max_val_windows 0 --max_ref_windows 256 --probe_lags 0,1,2,23" \
      > "$out/stdout.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) END rc=$rc $stage $batch $feature" >> "$LOG_FILE"
  if ! valid_terminal "$out"; then
    printf '{"status":"FAILED","rc":%s,"feature":"%s","families":"%s","seed":0}\n' \
      "$rc" "$feature" "$FAMILIES" > "$out/FAILED.json"
  fi
done < "$PLAN"
echo "$(date -u +%FT%TZ) DONE" >> "$LOG_FILE"
