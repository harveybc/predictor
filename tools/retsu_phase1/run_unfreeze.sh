#!/bin/bash
# Windows on CPU, then the frozen torch head, then the fine-tuned encoder.
set -u
PHASE=/var/tmp/retsu-phase1
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
echo $$ > "$FOLLOW/unfreeze.pid"
rm -f "$FOLLOW/UNFREEZE_DONE" "$FOLLOW/UNFREEZE_FAILED" "$FOLLOW/unfreeze_train.log"
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/var/tmp/sac399-series/runtime

echo "START windows $(date -Is)" >> "$FOLLOW/unfreeze_train.log"
"$PY" -u "$PHASE/dump_windows.py" >> "$FOLLOW/unfreeze_train.log" 2>&1
code=$?
echo "WINDOWS_EXIT:$code $(date -Is)" >> "$FOLLOW/unfreeze_train.log"
if [[ "$code" -ne 0 ]]; then
  echo "windows=$code" > "$FOLLOW/UNFREEZE_FAILED"
  exit 1
fi

run_one() {
  local mode="$1"
  local out="/var/tmp/retsu-phase1-unfreeze-$mode"
  rm -rf "$out"
  echo "START $(date -Is) $mode" >> "$FOLLOW/unfreeze_train.log"
  RETSU_OUT="$out" RETSU_MODE="$mode" \
    "$PY" -u "$PHASE/train_unfreeze.py" >> "$FOLLOW/unfreeze_train.log" 2>&1
  local train_code=$?
  echo "TRAIN_EXIT:$train_code $mode $(date -Is)" >> "$FOLLOW/unfreeze_train.log"
  if [[ "$train_code" -ne 0 ]]; then
    echo "train_$mode=$train_code" > "$FOLLOW/UNFREEZE_FAILED"
    exit 1
  fi
}

run_one frozen
run_one tuned
date -Is > "$FOLLOW/UNFREEZE_DONE"
echo UNFREEZE_DONE
