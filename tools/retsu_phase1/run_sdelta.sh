#!/bin/bash
# Causal seasonal increment. Frozen temporal representation, then fine-tuned.
set -u
PHASE=/var/tmp/retsu-phase1
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
echo $$ > "$FOLLOW/sdelta.pid"
rm -f "$FOLLOW/SDELTA_DONE" "$FOLLOW/SDELTA_FAILED" "$FOLLOW/sdelta.log"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/var/tmp/sac399-series/runtime

echo "START build $(date -Is)" >> "$FOLLOW/sdelta.log"
"$PY" -u "$PHASE/build_sdelta.py" >> "$FOLLOW/sdelta.log" 2>&1
code=$?
echo "BUILD_EXIT:$code $(date -Is)" >> "$FOLLOW/sdelta.log"
if [[ "$code" -ne 0 ]]; then
  echo "build=$code" > "$FOLLOW/SDELTA_FAILED"
  exit 1
fi

run_one() {
  local mode="$1"
  local out="/var/tmp/retsu-phase1-sdelta-$mode"
  rm -rf "$out"
  echo "START $(date -Is) $mode" >> "$FOLLOW/sdelta.log"
  RETSU_OUT="$out" RETSU_MODE="$mode" RETSU_DATASET="$PHASE/dataset_sdelta.npz" \
    "$PY" -u "$PHASE/train_unfreeze.py" >> "$FOLLOW/sdelta.log" 2>&1
  local train_code=$?
  echo "TRAIN_EXIT:$train_code $mode $(date -Is)" >> "$FOLLOW/sdelta.log"
  if [[ "$train_code" -ne 0 ]]; then
    echo "train_$mode=$train_code" > "$FOLLOW/SDELTA_FAILED"
    exit 1
  fi
}

run_one frozen
run_one tuned
"$PY" -u "$PHASE/summarize_sdelta.py" >> "$FOLLOW/sdelta.log" 2>&1
code=$?
if [[ "$code" -ne 0 ]]; then
  echo "summary=$code" > "$FOLLOW/SDELTA_FAILED"
  exit 1
fi
date -Is > "$FOLLOW/SDELTA_DONE"
echo SDELTA_DONE
