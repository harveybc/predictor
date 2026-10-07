#!/usr/bin/env bash
set -euo pipefail

code="$HOME/.local/state/canonical_20261003/ps2/laneD_reader_8987c57"
input="$HOME/.local/state/canonical_20261003/ps2/batch_001"
output="$HOME/.local/state/canonical_20261003/fs4/cost_pilot_px_rv5_20261007"
python_bin="$HOME/anaconda3/envs/tensorflow/bin/python"
launcher="$HOME/.local/bin/crispdm-run"
gpu_uuid="${FS4_GPU_UUID:?physical CUDA UUID required}"

test -f "$input/batch_manifest.json"
test -x "$python_bin"
test -x "$launcher"
nvidia-smi -L | grep -F "$gpu_uuid" >/dev/null
mkdir -p "$output"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export LD_LIBRARY_PATH="$(< "$HOME/.local/state/scratch/m07/ld_library_path.txt")"

cd "$code"
exec "$launcher" -q -m 4G -t 30m -n fs4-cost-pilot-px-rv5 -- \
  "$python_bin" -m app.univariate_temporal_pilot \
  --batch_dir "$input" --out_dir "$output" --features px.rv5 \
  --window 24 --latent_dim 8 --seed 0 --families identity,random,ae,dae \
  --batch_size 64 --max_fit_windows 4096 --max_val_windows 512 \
  --max_ref_windows 256 --probe_lags 0,1,2,23
