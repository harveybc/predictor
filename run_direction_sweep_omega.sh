#!/bin/bash
# =============================================================================
# Omega direction sweep (local, lower-tier GPU) — lightweight models
# ANN (2) + CNN (2) + Logistic (2) = 6 models
# =============================================================================
set -e
cd /home/harveybc/Documents/GitHub/predictor

CONFIGS=(
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_ann_direction_long_1d_optimization_config.json"
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_ann_direction_short_1d_optimization_config.json"
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_cnn_direction_long_1d_optimization_config.json"
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_cnn_direction_short_1d_optimization_config.json"
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_logistic_direction_long_1d_optimization_config.json"
    "examples/config/phase_1c_direction/optimization/phase_1c_direction_logistic_direction_short_1d_optimization_config.json"
)

echo "=============================================="
echo "  Omega direction sweep: ${#CONFIGS[@]} models"
echo "  GPU: lower-tier (8GB)"
echo "  Phase: 1c direction classification"
echo "=============================================="

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME=$(basename "$CFG" .json)
    echo ""
    echo "----------------------------------------------"
    echo "  [$((i+1))/${#CONFIGS[@]}] $NAME"
    echo "  Started: $(date)"
    echo "----------------------------------------------"
    python app/main.py --load_config "$CFG"
    echo "  Finished: $(date)"
done

echo ""
echo "=============================================="
echo "  Omega direction sweep COMPLETE: $(date)"
echo "=============================================="
