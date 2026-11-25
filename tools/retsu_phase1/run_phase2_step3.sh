#!/bin/bash
# Step 3 on Dragon. Torch needs the nvidia cuDNN engine libs that ship inside
# the trading-stack env. This does not predict 2024.
set -u
NVIDIA=/home/harveybc/anaconda3/envs/trading-stack/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(echo "$NVIDIA"/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /var/tmp/retsu-phase1-causal-p2
exec /var/tmp/discrete-sac-probe-venv/bin/python -u train_phase2_step3.py --plan /var/tmp/retsu-phase1-causal-p2/MINI_WORK_PLAN.md
