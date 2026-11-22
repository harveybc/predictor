#!/bin/bash
# CLOSE-change labels on CPU, then seed-7 and the pretrained encoder on the 4090.
set -u
PHASE=/var/tmp/retsu-phase1
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
echo $$ > "$FOLLOW/dclose.pid"
rm -f "$FOLLOW/DCLOSE_DONE" "$FOLLOW/DCLOSE_FAILED" "$FOLLOW/dclose_train.log"
SP=/home/harveybc/anaconda3/envs/trading-stack/lib/python3.12/site-packages/nvidia
NV="$(find "$SP" -type d -name lib | paste -sd:)"
export LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export RETSU_MAE_UNIT=close_change

"$PY" -u "$PHASE/build_dclose_targets.py" > "$FOLLOW/dclose_build.log" 2>&1
code=$?
echo "BUILD_EXIT:$code $(date -Is)" >> "$FOLLOW/dclose_build.log"
if [[ "$code" -ne 0 ]]; then
  echo "build=$code" > "$FOLLOW/DCLOSE_FAILED"
  exit 1
fi

run_one() {
  local name="$1"
  local dataset="$2"
  local out="$3"
  rm -rf "$out"
  echo "START $(date -Is) $name" >> "$FOLLOW/dclose_train.log"
  RETSU_OUT="$out" RETSU_DATASET="$dataset" RETSU_MAE_UNIT=close_change \
    "$PY" -u "$PHASE/train_head.py" >> "$FOLLOW/dclose_train.log" 2>&1
  local train_code=$?
  echo "TRAIN_EXIT:$train_code $name $(date -Is)" >> "$FOLLOW/dclose_train.log"
  if [[ "$train_code" -ne 0 ]]; then
    echo "train_$name=$train_code" > "$FOLLOW/DCLOSE_FAILED"
    exit 1
  fi
}

run_one seed7 "$PHASE/dataset_dclose_seed7.npz" /var/tmp/retsu-phase1-dclose-seed7
run_one pretrained "$PHASE/dataset_dclose_pretrained.npz" /var/tmp/retsu-phase1-dclose-pretrained
"$PY" -u "$PHASE/summarize_dclose.py" > "$FOLLOW/dclose_summary.log" 2>&1
code=$?
if [[ "$code" -ne 0 ]]; then
  echo "summary=$code" > "$FOLLOW/DCLOSE_FAILED"
  exit 1
fi
date -Is > "$FOLLOW/DCLOSE_DONE"
echo DCLOSE_DONE
