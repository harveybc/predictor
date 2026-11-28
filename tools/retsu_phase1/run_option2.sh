#!/bin/bash
# Bounded DEAP search on Dragon. Same cuDNN path as the phase-2 trainer.
set -u
NVIDIA=/home/harveybc/anaconda3/envs/trading-stack/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(echo "$NVIDIA"/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd /var/tmp/retsu-phase1-option2
exec /var/tmp/discrete-sac-probe-venv/bin/python -u train_option2.py --plan /var/tmp/retsu-phase1-option2/OPTION2_PLAN.md
