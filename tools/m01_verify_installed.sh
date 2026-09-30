#!/usr/bin/env bash
# M01 installed-environment verification (run under crispdm-run on a CPU worker).
#
#   tools/m01_verify_installed.sh VENV_PY CHECKOUT MASTER_REF PREDICTION_PROVIDER OUT
#
# VENV_PY   python of an ISOLATED venv (no system/user site-packages) holding only
#           third-party dependencies pinned to the campaign environment.
# CHECKOUT  a worktree of the M01 branch (its git repo also provides MASTER_REF).
# Steps, all with CUDA hidden and from working directories outside the checkout
# unless data-relative paths require the checkout as cwd (python -I then keeps
# the checkout off sys.path):
#   1. install MASTER_REF non-editable; boundary probe + legacy e2e  -> OUT/master
#   2. install the M01 tip non-editable; boundary probe + legacy e2e -> OUT/m01
#   3. copy tests/ outside the checkout and run the M01 suites against the
#      INSTALLED package with M01_REQUIRE_INSTALLED=1                 -> OUT/m01/pytest.log
#   4. diff the two probe documents                                   -> OUT/probe_diff.txt
set -euo pipefail
PY=$1; CHECKOUT=$(cd "$2" && pwd); MASTER_REF=$3; PP=$(cd "$4" && pwd); OUT=$5
export CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 PYTHONNOUSERSITE=1
mkdir -p "$OUT"
CONFIG=examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

install_ref () {   # $1 = git ref
  rm -rf "$SRC/pkg" && mkdir -p "$SRC/pkg"
  git -C "$CHECKOUT" archive "$1" | tar -x -C "$SRC/pkg"
  "$PY" -m pip uninstall -y -q predictor >/dev/null 2>&1 || true
  "$PY" -m pip install -q --no-deps --no-build-isolation "$SRC/pkg"
}

probe () {   # $1 = label
  mkdir -p "$OUT/$1"
  ( cd "$OUT/$1" && "$PY" -I "$CHECKOUT/tools/m01_boundary_probe.py" --repo "$CHECKOUT" \
      --prediction-provider "$PP" --out "$OUT/$1/probe.json" ) > "$OUT/$1/probe.log" 2>&1
  ( cd "$CHECKOUT" && "$PY" -I tools/m01_legacy_e2e.py --config "$CONFIG" --out "$OUT/$1/e2e" ) \
      > "$OUT/$1/e2e.log" 2>&1
  "$PY" -m pip freeze > "$OUT/$1/pip_freeze.txt"
  "$PY" -c "import keras, tensorflow, sys; print(sys.version.split()[0], tensorflow.__version__, keras.__version__)" \
      > "$OUT/$1/versions.txt"
}

install_ref "$MASTER_REF"; probe master
install_ref HEAD;          probe m01
rm -rf "$SRC/tests" && cp -r "$CHECKOUT/tests" "$SRC/tests"
( cd "$SRC/tests" && M01_REQUIRE_INSTALLED=1 "$PY" -I -m pytest -q -p no:cacheprovider -rs \
    -W ignore::DeprecationWarning test_modular_temporal.py test_modular_pretrain.py \
    test_modular_assembly.py test_modular_flat_parity_m04.py test_modular_legacy_boundaries.py ) \
    > "$OUT/m01/pytest.log" 2>&1 || echo "PYTEST_FAILED" >> "$OUT/m01/pytest.log"
"$PY" - "$OUT" <<'EOF' > "$OUT/probe_diff.txt"
import json, sys
out = sys.argv[1]
a, b = (json.load(open(f"{out}/{k}/probe.json")) for k in ("master", "m01"))
for section in ("entry_points", "config_resolutions", "direction_models"):
    if a[section] == b[section]:
        print(section, "IDENTICAL")
        continue
    if section == "entry_points":
        for g in sorted(set(a[section]) | set(b[section])):
            old, new = set(map(tuple, a[section].get(g, []))), set(map(tuple, b[section].get(g, [])))
            if old != new:
                print(section, g, "removed", sorted(old - new), "added", sorted(new - old))
    else:
        keys = sorted(set(a[section]) | set(b[section]))
        print(section, "DIFFERENT", [k for k in keys if a[section].get(k) != b[section].get(k)][:20])
ea, eb = (json.load(open(f"{out}/{k}/e2e/e2e_summary.json")) for k in ("master", "m01"))
print("results_header", "IDENTICAL" if ea["results_header"] == eb["results_header"] else "DIFFERENT")
print("metric_labels", "IDENTICAL" if ea["metric_labels"] == eb["metric_labels"] else "DIFFERENT",
      len(eb["metric_labels"]))
print("outputs_present", "IDENTICAL" if ea["outputs_present"] == eb["outputs_present"] else
      ("DIFFERENT", ea["outputs_present"], eb["outputs_present"]))
EOF
cat "$OUT/probe_diff.txt"; tail -3 "$OUT/m01/pytest.log"
