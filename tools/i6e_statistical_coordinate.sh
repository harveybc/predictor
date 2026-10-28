#!/usr/bin/env bash
# Immutable two-model dispatch, unattended artifact collection and exact closure.
set -euo pipefail
: "${PREFERRED_HOST:?required}" "${SECONDARY_HOST:?required}" "${PIN:?required}"
ROOT=${ROOT:-$HOME/.local/state/canonical_20261003}
WT=$(cd "$(dirname "$0")/.." && pwd)
RWT=$HOME/Documents/GitHub/.worktrees/predictor-i6e-statistical-worker
OUT=$ROOT/i6e/statistical72_v1
PY=$HOME/anaconda3/bin/python
mkdir -p "$OUT"
exec >>"$OUT/COORDINATOR.log" 2>&1
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
trap 'printf "FAILED line=%s\n" "$LINENO"' ERR
deploy() {
  local host=$1 model=$2
  ssh "$host" bash -s -- "$PIN" "$RWT" <<'REMOTE'
set -euo pipefail
git -C "$HOME/Documents/GitHub/predictor" fetch origin codex/i6d-weekly-walk-forward-20261009
if [[ ! -d "$2" ]]; then git -C "$HOME/Documents/GitHub/predictor" worktree add --detach "$2" "$1"; fi
test "$(git -C "$2" rev-parse HEAD)" = "$1"
test -z "$(git -C "$2" status --porcelain)"
REMOTE
  ssh "$host" mkdir -p "$OUT"
  rsync -a "$OUT/DESIGN.json" "$OUT/INPUT.npz" "$OUT/TRAIN.npz" "$host:$OUT/"
  ssh "$host" bash -s -- "$RWT" "$OUT" "$model" <<'REMOTE'
set -euo pipefail
systemd-run --user --unit="predictor-i6e-stat-${3,,}" --collect --property=MemoryMax=268435456 \
  --working-directory="$1" --setenv=CUDA_VISIBLE_DEVICES= --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=OMP_NUM_THREADS=1 "$HOME/anaconda3/bin/python" -m tools.i6e_statistical_campaign worker \
  --output "$2" --model "$3"
REMOTE
}
deploy "$PREFERRED_HOST" ARIMA
deploy "$SECONDARY_HOST" ARIMAX
collect() {
  for host in "$PREFERRED_HOST" "$SECONDARY_HOST"; do
    rsync -a --include='cells/***' --include='cost_*/***' --exclude='*' "$host:$OUT/" "$OUT/"
  done
  cd "$WT"
  "$PY" -m tools.i6e_statistical_campaign status --output "$OUT"
}
while ssh "$PREFERRED_HOST" systemctl --user is-active --quiet predictor-i6e-stat-arima.service || \
      ssh "$SECONDARY_HOST" systemctl --user is-active --quiet predictor-i6e-stat-arimax.service; do
  collect
  sleep 45
done
collect
cd "$WT"
"$PY" -m tools.i6e_statistical_campaign close --output "$OUT"
if [[ -n "${WAREHOUSE_URL:-}" && -n "${WAREHOUSE_TOKEN_FILE:-}" ]]; then
  "$PY" -m tools.i6e_publish_olap --statistical-root "$OUT" --output "$OUT/olap" \
    --url "$WAREHOUSE_URL" --token-file "$WAREHOUSE_TOKEN_FILE" --code-commit "$PIN"
fi
printf 'STATISTICAL_CONTRAST_COMPLETE\n'
