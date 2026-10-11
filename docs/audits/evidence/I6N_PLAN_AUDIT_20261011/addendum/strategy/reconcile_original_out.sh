#!/usr/bin/env bash
# I6-N audit addendum, lane strategy: reconcile probes/a/strategy_counterexample.out (retained,
# unchanged) with the retained script probes/a/strategy_counterexample.py and with a clean rerun
# of that same script (original_probe_clean_rerun.out, produced by run_strategy_addendum.sh step 3).
# Read-only; no python.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ORIG="$HERE/../../probes/a"
GH="$HOME/Documents/GitHub"
POL="$GH/heuristic-strategy/app/policies/prediction_entry_exit.py"
cd "$HERE"
echo "== hashes"
sha256sum ../../probes/a/strategy_counterexample.py ../../probes/a/strategy_counterexample.out \
  ../../probes/a/stub_trading_contracts.py stub/trading_contracts.py original_probe_clean_rerun.out
echo "== line counts: retained .out / clean rerun"
wc -l < ../../probes/a/strategy_counterexample.out; wc -l < original_probe_clean_rerun.out
echo "== diff clean_rerun retained_out (lines only in the retained .out are prefixed '>')"
diff original_probe_clean_rerun.out ../../probes/a/strategy_counterexample.out
echo "== retained .out lines 1-19 equal clean-rerun lines 1-19:"
cmp <(sed -n 1,19p ../../probes/a/strategy_counterexample.out) <(sed -n 1,19p original_probe_clean_rerun.out) && echo yes
echo "== retained .out lines 21-24 equal clean-rerun lines 17-20 (the last four lines of one run):"
cmp <(sed -n 21,24p ../../probes/a/strategy_counterexample.out) <(sed -n 17,20p original_probe_clean_rerun.out) && echo yes
echo "== does the retained .py print the line 'signature has no short argument'?"
grep -c "signature has no short argument" ../../probes/a/strategy_counterexample.py
echo "== locals of calculate_entry_geometry whose names contain 'short' (why revision 1's substring check printed False)"
awk '/^def calculate_entry_geometry/,/^def should_early_close/' "$POL" | grep -n -o -E "\b[a-z_]*short[a-z_]*\b *=" | sort -u
