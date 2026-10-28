#!/usr/bin/env bash
# Durable coordinator: deploy immutable code, start isolated weekly workers,
# collect artifacts and close the complete common-row contrast automatically.
set -euo pipefail
: "${PREFERRED_HOST:?required}" "${SECONDARY_HOST:?required}"
: "${PREFERRED_GPU:?required}" "${SECONDARY_GPU:?required}"
: "${PIN:?required}" "${MEMORY_BYTES:?measured peak times 1.25 required}"
ROOT=${ROOT:-$HOME/.local/state/canonical_20261003}
WT=$(cd "$(dirname "$0")/.." && pwd)
RWT=$HOME/Documents/GitHub/.worktrees/predictor-i6e-compact-worker
OUT=$ROOT/i6e/weekly_compact72_v1
PARENT=$ROOT/i6d/weekly_h1_branch_only_robust_v1/WEEKLY_DESIGN.json
PILOT=$ROOT/i6e/compact72_cost_v1
PY=$HOME/anaconda3/bin/python
mkdir -p "$OUT"
exec >>"$OUT/COORDINATOR.log" 2>&1
trap 'printf "FAILED line=%s\n" "$LINENO"' ERR
cd "$WT"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.i6e_campaign init --output "$OUT" --parent-design "$PARENT"
for host in "$PREFERRED_HOST" "$SECONDARY_HOST"; do
  ssh "$host" bash -s -- "$PIN" "$RWT" <<'REMOTE'
set -euo pipefail
repo=$HOME/Documents/GitHub/predictor
git -C "$repo" fetch origin codex/i6d-weekly-walk-forward-20261009
if [[ ! -d "$2" ]]; then git -C "$repo" worktree add --detach "$2" "$1"; fi
test "$(git -C "$2" rev-parse HEAD)" = "$1"
test -z "$(git -C "$2" status --porcelain)"
REMOTE
  ssh "$host" mkdir -p "$OUT" "$PILOT"
  rsync -a "$OUT/DESIGN.json" "$host:$OUT/"
  rsync -a "$PILOT/" "$host:$PILOT/"
done
launch() {
  ssh "$1" bash -s -- "$RWT" "$ROOT" "$2" "$3" "$4" "$MEMORY_BYTES" <<'REMOTE'
set -euo pipefail
WT=$1; ROOT=$2; slot=$3; gpu=$4; unit=$5; cap=$6
TF=$HOME/anaconda3/envs/tensorflow
LD=$(cat "$HOME/.local/state/scratch/m07/ld_library_path.txt")
OUT=$ROOT/i6e/weekly_compact72_v1
args=(worker --output "$OUT" --slot "$slot" --memory-bytes "$cap"
 --admission "$HOME/.local/bin/crispdm-run"
 --parent-design "$ROOT/i6d/weekly_h1_branch_only_robust_v1/WEEKLY_DESIGN.json"
 --pilot-root "$ROOT/i6e/compact72_cost_v1"
 --target-parquet "$ROOT/fs_causal/inputs/ps1/batch_001/targets_train.parquet"
 --validation-target-parquet "$ROOT/i6a/validation_2024/batch_001/targets_train.parquet")
for batch in 001 002 003; do
  args+=(--feature-parquet "$ROOT/ps1/batch_$batch/features_train.parquet"
         --validation-feature-parquet "$ROOT/i6a/validation_2024/batch_$batch/features_train.parquet")
done
printf -v command '%q ' "$TF/bin/python" -m tools.i6e_campaign "${args[@]}"
systemd-run --user --unit="$unit" --collect --property=MemoryMax=268435456 /bin/bash -lc \
 "cd '$WT'; exec env CUDA_VISIBLE_DEVICES='$gpu' LD_LIBRARY_PATH='$LD' XLA_FLAGS=--xla_gpu_cuda_data_dir='$TF/lib/python3.12/site-packages/nvidia/cuda_nvcc' PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=2 $command"
REMOTE
}
launch "$PREFERRED_HOST" preferred "$PREFERRED_GPU" predictor-i6e72-preferred
launch "$SECONDARY_HOST" secondary "$SECONDARY_GPU" predictor-i6e72-secondary
collect() {
  for host in "$PREFERRED_HOST" "$SECONDARY_HOST"; do
    rsync -a --include='cells/***' --include='WORKER_*.json' --exclude='*' "$host:$OUT/" "$OUT/"
  done
  CUDA_VISIBLE_DEVICES='' "$PY" -m tools.i6e_campaign status --output "$OUT"
}
while ssh "$PREFERRED_HOST" systemctl --user is-active --quiet predictor-i6e72-preferred.service || \
      ssh "$SECONDARY_HOST" systemctl --user is-active --quiet predictor-i6e72-secondary.service; do
  collect
  sleep 45
done
collect
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.i6e_campaign close --output "$OUT"
printf 'NEURAL_CONTRAST_COMPLETE\n'
