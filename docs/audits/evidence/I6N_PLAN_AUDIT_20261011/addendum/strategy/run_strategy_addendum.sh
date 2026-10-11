#!/usr/bin/env bash
# I6-N audit addendum, lane strategy. Exact commands that produced every retained
# *.out in this directory. CPU only, one crispdm-run job at a time, read-only on
# every audited object. Run from this directory:  bash run_strategy_addendum.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GH="$HOME/Documents/GitHub"                 # parent of the sibling checkouts
CONDA_BASE="$HOME/anaconda3"
PY_BASE="$CONDA_BASE/bin/python3"            # base env (holds the 2025-03-14 site-packages copy)
PY_TS="$CONDA_BASE/envs/trading-stack/bin/python"   # env of every lts/agent-multi user unit
HS="$GH/heuristic-strategy"                  # inspected checkout
LTS="$GH/lts"
PRED_I6N="$GH/.worktrees/predictor-i6d-weekly"   # audited I6-N object (producer side)
export CUDA_VISIBLE_DEVICES=""
RUN="crispdm-run -q -m 1G -t 10m -n i6n-add-strategy --"

# 0. Source attribution (git/sha256/grep only, no python).
bash "$HERE/inspect_sources_and_executors.sh" > "$HERE/inspect_sources_and_executors.out"

# 1. Effective-plugin resolution per environment and import context.
$RUN "$PY_BASE" -I "$HERE/resolve_effective_plugin.py" --gh-root "$GH" --conda-base "$CONDA_BASE" \
  --interp base="$PY_BASE" --interp trading-stack="$PY_TS" \
  --interp venv-feature-eng="$HOME/.venvs/feature-eng/bin/python" \
  --interp venv-feature-extractor="$HOME/.venvs/feature-extractor/bin/python" \
  --interp venv-store-hosts="$HOME/.venvs/store-hosts/bin/python" \
  --context isolated= --context root:heuristic-strategy="$HS" \
  --context root:predictor-i6d-weekly="$PRED_I6N" --context root:lts="$LTS" \
  > "$HERE/resolve_effective_plugin.out"

# 2. Counterexamples against the inspected checkout copy (policy and plugin) and each other copy.
$RUN "$PY_TS" -I "$HERE/counterexample_v2.py" --api hs_policy \
  --module "$HS/app/policies/prediction_entry_exit.py" --import-root "$HS" \
  --gh-root "$GH" --conda-base "$CONDA_BASE" > "$HERE/cx_v2_hs_policy_checkout.out"
$RUN "$PY_TS" -I "$HERE/counterexample_v2.py" --api hs_plugin \
  --module "$HS/app/plugins/plugin_long_short_predictions.py" --import-root "$HS" \
  --gh-root "$GH" --conda-base "$CONDA_BASE" > "$HERE/cx_v2_hs_plugin_checkout.out"
$RUN "$PY_BASE" -I "$HERE/counterexample_v2.py" --api legacy_plugin \
  --module "$CONDA_BASE/lib/python3.12/site-packages/app/plugins/plugin_long_short_predictions.py" \
  --gh-root "$GH" --conda-base "$CONDA_BASE" > "$HERE/cx_v2_legacy_plugin_base_site_packages.out"
$RUN "$PY_TS" -I "$HERE/counterexample_v2.py" --api lts_copy \
  --module "$LTS/plugins_strategy/heuristic_strategy.py" \
  --gh-root "$GH" --conda-base "$CONDA_BASE" > "$HERE/cx_v2_lts_copy.out"

# 3. Clean rerun of the ORIGINAL retained probe (probes/a/strategy_counterexample.py, unchanged)
#    with the interpreter it originally ran under (base env, per its first traceback) and the
#    retained stub restored under the import name it had in the session scratch
#    (stub/trading_contracts.py here == probes/a/stub_trading_contracts.py byte for byte).
ORIG="$HERE/../../probes/a"
$RUN "$PY_BASE" -I "$ORIG/strategy_counterexample.py" "$HERE/stub" "$HS/app/policies/prediction_entry_exit.py" \
  > "$HERE/original_probe_clean_rerun.out"

# 4. Reconciliation of the original retained .out (git/sha256/diff only).
bash "$HERE/reconcile_original_out.sh" > "$HERE/reconcile_original_out.out"
