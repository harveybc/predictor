#!/bin/bash
# CPU panel and CPU coverage run together. The 4090 trains the control
# only after the seed-7 embeddings exist.
set -u
PHASE=/var/tmp/retsu-phase1
FOLLOW=$PHASE/followup
PY=/var/tmp/discrete-sac-probe-venv/bin/python
mkdir -p "$FOLLOW"
rm -f "$FOLLOW/DONE" "$FOLLOW/FAILED"
SP=/home/harveybc/anaconda3/envs/trading-stack/lib/python3.12/site-packages/nvidia
NV="$(find "$SP" -type d -name lib | paste -sd:)"

run_coverage() {
  CUDA_VISIBLE_DEVICES= \
  LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  TF_USE_LEGACY_KERAS=1 \
  TF_CPP_MIN_LOG_LEVEL=2 \
  OMP_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -u "$PHASE/coverage_semester.py" > "$FOLLOW/coverage.log" 2>&1
  code=$?
  echo "COVERAGE_EXIT:$code $(date -Is)" >> "$FOLLOW/coverage.log"
  return "$code"
}

run_panel() {
  PYTHONPATH=/var/tmp/sac399-series/runtime \
  OMP_NUM_THREADS=8 \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -u "$PHASE/encoder_panel.py" > "$FOLLOW/encoder.log" 2>&1
  code=$?
  echo "PANEL_EXIT:$code $(date -Is)" >> "$FOLLOW/encoder.log"
  return "$code"
}

run_coverage &
coverage_pid=$!
run_panel &
panel_pid=$!

wait "$panel_pid"
panel_code=$?
if [[ ! -f "$PHASE/dataset_random_encoder.npz" ]]; then
  echo "panel_failed code=$panel_code" > "$FOLLOW/FAILED"
  wait "$coverage_pid" || true
  exit 1
fi

LD_LIBRARY_PATH="$NV${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
TF_USE_LEGACY_KERAS=1 \
TF_CPP_MIN_LOG_LEVEL=2 \
OMP_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
RETSU_OUT=/var/tmp/retsu-phase1-control \
RETSU_DATASET=$PHASE/dataset_random_encoder.npz \
"$PY" -u "$PHASE/train_head.py" > "$FOLLOW/control_train.log" 2>&1
train_code=$?
echo "TRAIN_EXIT:$train_code $(date -Is)" >> "$FOLLOW/control_train.log"
wait "$coverage_pid"
coverage_code=$?

if [[ "$train_code" -ne 0 || "$coverage_code" -ne 0 ]]; then
  echo "train=$train_code coverage=$coverage_code panel=$panel_code" > "$FOLLOW/FAILED"
  exit 1
fi
"$PY" -u "$PHASE/compare_control.py" > "$FOLLOW/compare.log" 2>&1
compare_code=$?
if [[ "$compare_code" -ne 0 || ! -f "$FOLLOW/control_vs_pretrained.json" ]]; then
  echo "compare=$compare_code" > "$FOLLOW/FAILED"
  exit 1
fi
date -Is > "$FOLLOW/DONE"
echo DONE
