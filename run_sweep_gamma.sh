#!/bin/bash
# =============================================================================
# Gamma sweep (192.168.0.106, RTX 5070 Ti 12GB) — mid models, fastest GPU
# Transformer sell (2) + MIMO (4) + LSTM (4) + CNN buy_entry (1) = 11 models
# =============================================================================
set -e
cd /home/harveybc/Documents/GitHub/predictor

CONFIGS=(
    "examples/config/phase_1b_binary/phase_1b_binary_transformer_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_transformer_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_mimo_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_mimo_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_mimo_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_mimo_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_lstm_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_lstm_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_lstm_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_lstm_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_cnn_buy_entry_1d_config.json"
)

echo "=============================================="
echo "  Gamma sweep: ${#CONFIGS[@]} models"
echo "  GPU: RTX 5070 Ti (12GB, fastest)"
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
echo "  Gamma sweep COMPLETE: $(date)"
echo "=============================================="
