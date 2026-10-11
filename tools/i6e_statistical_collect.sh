#!/usr/bin/env bash
# One bounded collection pass, suitable for a timer; no log-watching agent.
set -euo pipefail
: "${PREFERRED_HOST:?required}" "${SECONDARY_HOST:?required}"
ROOT=${ROOT:-$HOME/.local/state/canonical_20261003}
OUT=$ROOT/i6e/statistical72_v1
WT=$(cd "$(dirname "$0")/.." && pwd)
if [[ -f "$OUT/olap/PUBLICATION.json" ]]; then exit 0; fi
for host in "$PREFERRED_HOST" "$SECONDARY_HOST"; do
  rsync -a --include='cells/***' --include='cost_*/***' --exclude='*' "$host:$OUT/" "$OUT/"
done
cd "$WT"
OPENBLAS_NUM_THREADS=1 "$HOME/anaconda3/bin/python" -m tools.i6e_progress \
  --root "$OUT" --png "$OUT/PROGRESS.png"
