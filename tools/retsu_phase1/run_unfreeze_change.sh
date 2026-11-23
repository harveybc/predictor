#!/bin/bash
# Fine-tune, and the frozen control, on the CLOSE change. Windows already match.
set -u
PHASE=/var/tmp/retsu-phase1
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
echo $$ > "$FOLLOW/unfreeze_change.pid"
rm -f "$FOLLOW/UNFREEZE_CHANGE_DONE" "$FOLLOW/UNFREEZE_CHANGE_FAILED" "$FOLLOW/unfreeze_change.log" "$FOLLOW/unfreeze_change_scale_checked"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/var/tmp/sac399-series/runtime
export RETSU_DATASET=$PHASE/dataset_dclose_pretrained.npz

run_one() {
  local mode="$1"
  local out="/var/tmp/retsu-phase1-unfreeze-change-$mode"
  rm -rf "$out"
  echo "START $(date -Is) $mode" >> "$FOLLOW/unfreeze_change.log"
  RETSU_OUT="$out" RETSU_MODE="$mode" \
    "$PY" -u "$PHASE/train_unfreeze.py" >> "$FOLLOW/unfreeze_change.log" 2>&1
  local train_code=$?
  echo "TRAIN_EXIT:$train_code $mode $(date -Is)" >> "$FOLLOW/unfreeze_change.log"
  if [[ "$train_code" -ne 0 ]]; then
    echo "train_$mode=$train_code" > "$FOLLOW/UNFREEZE_CHANGE_FAILED"
    exit 1
  fi
}

run_one frozen
run_one tuned
"$PY" -u "$PHASE/summarize_unfreeze_change.py" >> "$FOLLOW/unfreeze_change.log" 2>&1
code=$?
if [[ "$code" -ne 0 ]]; then
  echo "summary=$code" > "$FOLLOW/UNFREEZE_CHANGE_FAILED"
  exit 1
fi
date -Is > "$FOLLOW/UNFREEZE_CHANGE_DONE"
echo UNFREEZE_CHANGE_DONE
