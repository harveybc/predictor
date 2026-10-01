#!/usr/bin/env bash
# M01: current LTS routes with and without the M01 predictor installed beside them.
#
#   tools/m01_verify_lts.sh BASE_PY LTS_CHECKOUT M01_CHECKOUT OUT [pytest node ids...]
#
# BASE_PY is the LTS environment's python (never modified). An overlay venv
# (--system-site-packages over BASE_PY) receives the M01 predictor non-editable,
# so the LTS environment itself is untouched. LTS tests run from an archived copy
# of LTS_CHECKOUT (its sqlite/test files land in scratch, never in the checkout),
# first with BASE_PY (baseline), then with the overlay; the two outcomes must match.
# Scratch and TMPDIR are on disk, never tmpfs.
set -euo pipefail
BASE_PY=$1; LTS=$(cd "$2" && pwd); M01=$(cd "$3" && pwd); OUT=$4; shift 4
TESTS=("$@")
[ ${#TESTS[@]} -gt 0 ] || TESTS=(tests/test_backtrader_broker.py tests/test_csv_workflow_e2e.py
                                 tests/unit/test_live_model_selection.py)
SCRATCH=${M01_SCRATCH:-$HOME/.local/state/scratch/m01}; mkdir -p "$SCRATCH" "$OUT"
export TMPDIR="$SCRATCH" CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
WORK=$(mktemp -d -p "$SCRATCH"); trap 'rm -rf "$WORK"' EXIT
git -C "$LTS" archive HEAD | (mkdir -p "$WORK/lts" && tar -x -C "$WORK/lts")
git -C "$M01" archive HEAD | (mkdir -p "$WORK/m01" && tar -x -C "$WORK/m01")
"$BASE_PY" -m venv --system-site-packages "$WORK/overlay"
"$WORK/overlay/bin/python" -m pip install -q --no-deps "$WORK/m01"
run () {  # $1 label, $2 python
  ( cd "$WORK/lts" && "$2" -m pytest -q -p no:cacheprovider --basetemp "$WORK/bt_$1" -rs \
      "${TESTS[@]}" ) > "$OUT/lts_$1.log" 2>&1 || true
  tail -1 "$OUT/lts_$1.log"
}
echo "lts $(git -C "$LTS" rev-parse --short HEAD)  m01 $(git -C "$M01" rev-parse --short HEAD)" > "$OUT/lts_summary.txt"
echo "baseline: $(run baseline "$BASE_PY")" >> "$OUT/lts_summary.txt"
"$WORK/overlay/bin/python" -c "import importlib.metadata as m; d=m.distribution('predictor'); \
print('overlay predictor', d.version, sorted(e.name for e in d.entry_points if e.group=='predictor.plugins' and e.name.startswith('modular')))" \
  >> "$OUT/lts_summary.txt"
echo "with_m01: $(run with_m01 "$WORK/overlay/bin/python")" >> "$OUT/lts_summary.txt"
grep -h -E "^(PASSED|FAILED|ERROR)|passed|failed" "$OUT/lts_baseline.log" | sort > "$OUT/lts_baseline.outcomes" || true
grep -h -E "^(PASSED|FAILED|ERROR)|passed|failed" "$OUT/lts_with_m01.log" | sed 's/ in [0-9.]*s.*//' | sort > "$OUT/lts_with_m01.outcomes" || true
cat "$OUT/lts_summary.txt"
