#!/bin/bash
# =============================================================================
# Omega sweep (local, lower-tier GPU) — lightweight models
# CNN (3) + TCN (4) + ANN (4) + Logistic (1) = 12 models
# =============================================================================
set -e
cd /home/harveybc/Documents/GitHub/predictor

CONFIGS=(
    "examples/config/phase_1b_binary/phase_1b_binary_cnn_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_cnn_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_cnn_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tcn_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tcn_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tcn_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tcn_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_ann_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_ann_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_ann_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_ann_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_logistic_buy_entry_1d_config.json"
)

echo "=============================================="
echo "  Omega sweep: ${#CONFIGS[@]} models"
echo "  GPU: lower-tier"
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
echo "  Omega sweep COMPLETE: $(date)"
echo "=============================================="
