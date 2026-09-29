#!/bin/sh
# RB02 execution driver: the twelve sealed Weather cells, SEQUENTIALLY, one governed unit per cell.
#
# It is a loop and nothing else. Every scientific decision is in `df_tsl_execute.py` and in the sealed design; every
# placement decision is in `crispdm-run`, which this script never bypasses and whose cap it never shrinks.
#
#   * ONE cell at a time: the next `crispdm-run` starts only after the previous child has exited, so there is never a
#     duplicate GPU executor on the admitted device.
#   * 8 GiB declared cap per cell, passed once, never reduced. A refusal is queued (`-q`) and waited out, never evaded.
#   * The device is named by PHYSICAL UUID and asserted again INSIDE each child by `df_tsl_execute`.
#   * Gate one's probe record is passed to every cell, which refuses to run if its verdict is not ADMISSIBLE.
#   * A cell that fails leaves its log and stops the loop: twelve partial cells are not a closure.
#
# Usage:  rb02_weather_cells.sh <lane-root> <gpu-uuid> [python]
set -eu
LANE=${1:?lane root}
UUID=${2:?physical GPU UUID}
PY=${3:-$HOME/anaconda3/envs/trading-stack/bin/python}
EXEC=$LANE/code/tools/df_tsl_execute.py
PROBE=$LANE/EVAL_PROBE.weather.h720.json
LOGS=$LANE/logs
mkdir -p "$LOGS" "$LANE/CELLS"

[ -f "$PROBE" ] || { echo "REFUSED: gate one has not run: $PROBE is absent" >&2; exit 75; }
python3 - "$PROBE" <<'EOF' || exit 75
import json, sys
p = json.load(open(sys.argv[1]))
if p.get("verdict") != "ADMISSIBLE":
    print("REFUSED: gate one verdict is %r; no scored cell runs behind a gate that has not passed" % p.get("verdict"),
          file=sys.stderr)
    raise SystemExit(75)
EOF

CUDA_VISIBLE_DEVICES=$UUID
CRISPDM_REQUIRED_GPU_UUID=$UUID
export CUDA_VISIBLE_DEVICES CRISPDM_REQUIRED_GPU_UUID

for H in 96 192 336 720; do
  for S in 2021 2022 2023; do
    CELL=weather_L96_h${H}_s${S}
    if [ -f "$LANE/CELLS/$CELL.json" ]; then
      echo "== $CELL already retained, not re-run"
      continue
    fi
    echo "== $CELL starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$HOME/.local/bin/crispdm-run" -m 8G -t 40m -n "rb02-weather-$CELL" -q -W 5400 -L "rb02-cell-$CELL" -- \
      "$PY" "$EXEC" cell --root "$LANE" --dataset weather --protocol L96 \
        --horizon "$H" --seed "$S" --data-path "$LANE/data/weather.csv" \
        --probe "$PROBE" --transport "$LANE/TRANSPORT.weather.json" \
      > "$LOGS/$CELL.out" 2> "$LOGS/$CELL.err" || {
        echo "== $CELL FAILED; the loop stops here. See $LOGS/$CELL.err" >&2
        tail -20 "$LOGS/$CELL.err" >&2
        exit 1
      }
    echo "== $CELL done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    tail -c 900 "$LOGS/$CELL.out"
  done
done
echo "== all twelve cells retained"
