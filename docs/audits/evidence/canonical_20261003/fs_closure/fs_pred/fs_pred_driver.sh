#!/bin/bash
# FS-PRED durable driver (worker_b, CPU only). One crispdm-run job per target, sequential.
# Resumable: the runner skips method records whose identity digests already match.
# Never touches GPU jobs, MT5/lts services or /tmp; all state lives under $ROOT.
set -uo pipefail
ROOT="$HOME/.local/state/canonical_20261003/fs_pred"
CODE="$ROOT/code"
PY="$ROOT/venv_chrono/bin/python"
CAP="${FS_PRED_CAP:-4G}"
PEAK_EVIDENCE="${FS_PRED_PEAK_EVIDENCE:-}"
PEAK_BYTES="${FS_PRED_PEAK_BYTES:-}"
TARGETS="${FS_PRED_TARGETS:-Y_s_1h Y_s_2h Y_s_3h Y_s_4h Y_s_5h Y_s_6h Y_l_24h Y_l_48h Y_l_72h Y_l_96h Y_l_120h Y_l_144h Y_b_s6 Y_b_l144}"
LOG="$ROOT/logs/driver.log"
export TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR" "$ROOT/logs" "$ROOT/out"
export CRISPDM_PYTHON=python3
echo "$(date -u +%FT%TZ) START cap=$CAP targets=[$TARGETS]" >> "$LOG"
for target in $TARGETS; do
  extra=()
  if [ -n "$PEAK_EVIDENCE" ]; then extra+=(-E "$PEAK_EVIDENCE" -P "$PEAK_BYTES"); fi
  echo "$(date -u +%FT%TZ) TARGET $target begin" >> "$LOG"
  "$HOME/.local/bin/crispdm-run" -m "$CAP" -t 20h -n "fs-pred-$target" -q -W 7200 -L "fs-pred-$target" "${extra[@]}" -- \
    env PYTHONPATH="$CODE" OMP_NUM_THREADS=1 "$PY" -m tools.feature_selection_batch_runner \
      --input-root "$ROOT/input/ps1" --input-manifest "$ROOT/input/input_manifest.json" \
      --clusters "$ROOT/input/fold_clusters.json" --output-root "$ROOT/out" \
      --targets "$target" --k 24 --sensitivity-ks 8,16,24,32,48 --seed 0 --chronoepilogi auto \
      >> "$ROOT/logs/runner_$target.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) TARGET $target rc=$rc" >> "$LOG"
  if [ "$rc" = 75 ]; then echo "$(date -u +%FT%TZ) ADMISSION REFUSED for $target; cap never lowered; stopping driver" >> "$LOG"; exit 75; fi
done
echo "$(date -u +%FT%TZ) DONE" >> "$LOG"
