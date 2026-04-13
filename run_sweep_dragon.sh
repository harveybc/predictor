#!/bin/bash
# =============================================================================
# Dragon sweep (192.168.0.107, RTX 4090 16GB) — heavy models
# TFT (4) + N-BEATS (4) + Transformer buy_entry/buy_exit (2) = 10 models
# =============================================================================
set -e
cd /home/harveybc/Documents/GitHub/predictor

CONFIGS=(
    "examples/config/phase_1b_binary/phase_1b_binary_tft_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tft_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tft_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_tft_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_n_beats_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_n_beats_buy_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_n_beats_sell_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_n_beats_sell_exit_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_transformer_buy_entry_1d_config.json"
    "examples/config/phase_1b_binary/phase_1b_binary_transformer_buy_exit_1d_config.json"
)

echo "=============================================="
echo "  Dragon sweep: ${#CONFIGS[@]} models"
echo "  GPU: RTX 4090 (16GB)"
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
echo "  Dragon sweep COMPLETE: $(date)"
echo "=============================================="
