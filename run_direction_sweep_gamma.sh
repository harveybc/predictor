#!/bin/bash
# =============================================================================
# Gamma direction sweep (192.168.0.106, RTX 5070 Ti 12GB) — medium models
# TCN (2) + MIMO (2) + LSTM (2) = 6 models
# =============================================================================
set -e

# GPU environment (pip nvidia packages)
NB=/home/harveybc/anaconda3/envs/tensorflow/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="${NB}/cudnn/lib:${NB}/cublas/lib:${NB}/cuda_runtime/lib:${NB}/cufft/lib:${NB}/curand/lib:${NB}/cusolver/lib:${NB}/cusparse/lib:${NB}/cuda_cupti/lib:${NB}/nvjitlink/lib:${NB}/cuda_nvrtc/lib:${NB}/nccl/lib"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

cd /home/harveybc/Documents/GitHub/predictor

CONFIGS=(
    "examples/config/phase_1c_direction/inference/phase_1c_direction_tcn_direction_long_1d_inference_config.json"
    "examples/config/phase_1c_direction/inference/phase_1c_direction_tcn_direction_short_1d_inference_config.json"
    "examples/config/phase_1c_direction/inference/phase_1c_direction_mimo_direction_long_1d_inference_config.json"
    "examples/config/phase_1c_direction/inference/phase_1c_direction_mimo_direction_short_1d_inference_config.json"
    "examples/config/phase_1c_direction/inference/phase_1c_direction_lstm_direction_long_1d_inference_config.json"
    "examples/config/phase_1c_direction/inference/phase_1c_direction_lstm_direction_short_1d_inference_config.json"
)

echo "=============================================="
echo "  Gamma direction sweep: ${#CONFIGS[@]} models"
echo "  GPU: RTX 5070 Ti (12GB)"
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
echo "  Gamma direction sweep COMPLETE: $(date)"
echo "=============================================="
