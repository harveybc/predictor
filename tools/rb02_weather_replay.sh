#!/bin/sh
# RB02 AUDIT driver: the twelve INDEPENDENTLY INVOKED evaluations of the twelve retained Weather checkpoints.
#
# It is a loop and nothing else. Every scientific decision lives in `df_tsl_replay.py`, in the sealed design and in the
# frozen replay rule it reads; every placement decision lives in `crispdm-run`, which this script never bypasses.
#
#   * NOTHING IS TRAINED. Each child reloads a retained checkpoint through the author's own `test(setting, test=1)`.
#   * ONE cell at a time: the next `crispdm-run` starts only after the previous child exits, so there is never a second
#     GPU child on the admitted device. Each child takes its OWN fresh aggregate admission.
#   * The declared cap is the SAME 8 GiB the executed cells were admitted under. It is passed once and NEVER reduced:
#     a refusal is queued (`-q`) and waited out, never evaded by shrinking the request.
#   * The audited lane is READ-ONLY here. Every artifact this driver produces lands in the separate replay lane.
#   * A cell that fails leaves its log and stops the loop: a partial replay is not a replay.
#
# Usage:  rb02_weather_replay.sh <audited-lane> <replay-lane> <gpu-uuid> [device] [python]
set -eu
LANE=${1:?audited lane root}
OUT=${2:?replay lane root}
UUID=${3:?physical GPU UUID}
DEV=${4:-cuda}
PY=${5:-$HOME/anaconda3/envs/trading-stack/bin/python}
EXEC=$OUT/code/tools/df_tsl_replay.py
LOGS=$OUT/logs
mkdir -p "$LOGS" "$OUT/REPLAYS"

[ -f "$LANE/DESIGN.weather.L96.json" ] || { echo "REFUSED: no sealed design at $LANE" >&2; exit 75; }
[ -f "$LANE/data/weather.csv" ] || { echo "REFUSED: the delivered bytes are not present at $LANE/data" >&2; exit 75; }

if [ "$DEV" = "cuda" ]; then
  CUDA_VISIBLE_DEVICES=$UUID
  CRISPDM_REQUIRED_GPU_UUID=$UUID
  export CUDA_VISIBLE_DEVICES CRISPDM_REQUIRED_GPU_UUID
else
  CUDA_VISIBLE_DEVICES=""
  export CUDA_VISIBLE_DEVICES
  unset CRISPDM_REQUIRED_GPU_UUID || true
fi

for H in 96 192 336 720; do
  for S in 2021 2022 2023; do
    CELL=weather_L96_h${H}_s${S}
    if [ -f "$OUT/REPLAYS/$CELL.$DEV.json" ]; then
      echo "== $CELL already replayed on $DEV, not re-run"
      continue
    fi
    echo "== $CELL replay ($DEV) starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$HOME/.local/bin/crispdm-run" -m 8G -t 40m -n "rb02-replay-$CELL-$DEV" -q -W 5400 -L "rb02-replay-$CELL-$DEV" -- \
      "$PY" "$EXEC" replay --lane "$LANE" --out "$OUT" --dataset weather --protocol L96 \
        --horizon "$H" --seed "$S" --data-path "$LANE/data/weather.csv" --device "$DEV" \
      > "$LOGS/$CELL.$DEV.out" 2> "$LOGS/$CELL.$DEV.err" || {
        echo "== $CELL FAILED; the loop stops here. See $LOGS/$CELL.$DEV.err" >&2
        tail -20 "$LOGS/$CELL.$DEV.err" >&2
        exit 1
      }
    echo "== $CELL done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    tail -c 700 "$LOGS/$CELL.$DEV.out"
  done
done
echo "== all twelve replays retained on $DEV"
