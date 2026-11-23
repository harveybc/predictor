#!/bin/bash
# Causal seasonal increment. Fresh seed-7 encoder, frozen then fine-tuned.
set -u
PHASE=/var/tmp/retsu-phase1-causal
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
echo $$ > "$FOLLOW/causal.pid"
rm -f "$FOLLOW/CAUSAL_DONE" "$FOLLOW/CAUSAL_FAILED"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/var/tmp/sac399-series/runtime
export RETSU_PHASE="$PHASE"

echo "START $(date -Is)" >> "$FOLLOW/causal.log"

run_one() {
  local mode="$1"
  local out="/var/tmp/retsu-phase1-causal-$mode"
  rm -rf "$out"
  mkdir -p "$out"
  echo "START $(date -Is) $mode" >> "$FOLLOW/causal.log"
  RETSU_OUT="$out" RETSU_MODE="$mode" \
    "$PY" -u "$PHASE/train_causal.py" >> "$FOLLOW/causal.log" 2>&1
  local train_code=$?
  echo "TRAIN_EXIT:$train_code $mode $(date -Is)" >> "$FOLLOW/causal.log"
  if [[ "$train_code" -ne 0 ]]; then
    echo "train_$mode=$train_code" > "$FOLLOW/CAUSAL_FAILED"
    exit 1
  fi
}

run_one frozen
run_one tuned
"$PY" -u "$PHASE/summarize_causal.py" >> "$FOLLOW/causal.log" 2>&1
code=$?
if [[ "$code" -ne 0 ]]; then
  echo "summary=$code" > "$FOLLOW/CAUSAL_FAILED"
  exit 1
fi
date -Is > "$FOLLOW/CAUSAL_DONE"
echo CAUSAL_DONE
