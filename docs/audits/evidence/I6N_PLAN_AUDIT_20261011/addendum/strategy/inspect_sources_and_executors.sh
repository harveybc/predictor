#!/usr/bin/env bash
# I6-N audit addendum, lane strategy. Read-only source attribution:
#  (i)  the inspected heuristic-strategy code: checkout HEAD/branch, blob ids of the plugin and
#       policy at every local ref, worktree byte hashes and cleanliness;
#  (ii) the other copies on this host (conda base site-packages install, lts copied logic);
#  (iii) every place in predictor (audited I6-N worktree), lts, agent-multi and gym-fx that could
#       resolve a heuristic strategy, and whether anything calls it.
# Prints with home paths replaced by $GH / $CONDA_BASE. No python, no network, no writes.
set -uo pipefail
GH="$HOME/Documents/GitHub"; CONDA_BASE="$HOME/anaconda3"
HS="$GH/heuristic-strategy"; LTS="$GH/lts"; PRED_I6N="$GH/.worktrees/predictor-i6d-weekly"
POL=app/policies/prediction_entry_exit.py; PLG=app/plugins/plugin_long_short_predictions.py
red() { sed -e "s#$GH#\$GH#g" -e "s#$CONDA_BASE#\$CONDA_BASE#g" -e "s#$HOME#~#g"; }
{
echo "== (i) inspected checkout"
echo "HEAD $(git -C "$HS" rev-parse HEAD) branch $(git -C "$HS" branch --show-current)"
echo "origin/master $(git -C "$HS" rev-parse origin/master)  master $(git -C "$HS" rev-parse master)"
git -C "$HS" merge-base --is-ancestor HEAD origin/master && echo "HEAD is an ancestor of origin/master"
for r in HEAD master origin/master; do
  echo "blob@$r policy=$(git -C "$HS" rev-parse "$r:$POL") plugin=$(git -C "$HS" rev-parse "$r:$PLG")"; done
echo "worktree status of the two files (empty = clean): [$(git -C "$HS" status --short -- "$POL" "$PLG" | tr '\n' ' ')]"
sha256sum "$HS/$POL" "$HS/$PLG"
echo "-- blob ids at every local branch / remote-tracking ref (policy plugin ref)"
for r in $(git -C "$HS" for-each-ref --format='%(refname:short)' refs/heads refs/remotes); do
  echo "$(git -C "$HS" rev-parse "$r:$POL" 2>/dev/null | cut -c1-10) $(git -C "$HS" rev-parse "$r:$PLG" 2>/dev/null | cut -c1-10) $r"
done | sort
echo "== (ii) other copies on this host"
SP="$CONDA_BASE/lib/python3.12/site-packages"
sha256sum "$SP/app/plugins/plugin_long_short_predictions.py"
stat -c '%y %n' "$SP/app/plugins/plugin_long_short_predictions.py"
echo "base-env dist: $(ls -d "$SP"/heuristic_strategy-*.dist-info)"
echo "direct_url: $(cat "$SP"/heuristic_strategy-*.dist-info/direct_url.json)"
echo "base-env install has app/policies: $( [ -d "$SP/app/policies" ] && echo yes || echo no )"
grep -n "^\[\|=" "$SP"/heuristic_strategy-*.dist-info/entry_points.txt
echo "lts HEAD $(git -C "$LTS" rev-parse HEAD) blob plugins_strategy/heuristic_strategy.py=$(git -C "$LTS" rev-parse HEAD:plugins_strategy/heuristic_strategy.py) status=[$(git -C "$LTS" status --short -- plugins_strategy/heuristic_strategy.py)]"
sha256sum "$LTS/plugins_strategy/heuristic_strategy.py"
echo "== (iii) executors: who resolves or calls a heuristic strategy"
echo "-- predictor audited worktree $(git -C "$PRED_I6N" rev-parse --short HEAD) (tracked files only; draft docs excluded)"
git -C "$PRED_I6N" grep -n -I -E "strategy_plugin_group|strategy_plugin_name|use_strategy\"?:|load_plugin\(strategy|from heuristic_strategy|evaluate_candidate\(individual" -- '*.py' ':!examples/*' ':!docs/*'
echo "-- predictor importers of the two files above (empty = none)"
git -C "$PRED_I6N" grep -n -I -E "optimizer_ga|data_processor copy" -- '*.py' ':!docs/*'
echo "-- lts: callers of the copied heuristic logic"
git -C "$LTS" grep -n -I -E "plugins_strategy.heuristic_strategy|_compute_heuristic_signal|start_heartbeat\(|heartbeat_loop\(" -- '*.py'
echo "-- lts entry-point group plugins_strategy (setup.py)"
git -C "$LTS" grep -n -A7 "'plugins_strategy': \[" -- setup.py
echo "-- user systemd units mentioning heuristic / prediction_entry_exit (unit names)"
grep -l -i -E "heuristic|prediction_entry_exit|plugin_long_short" "$HOME"/.config/systemd/user/*.service 2>/dev/null || echo "none"
echo "-- agent-multi / gym-fx: strategy.plugins names declared (none is a heuristic forecast strategy)"
git -C "$GH/gym-fx" grep -n -E "^\s*\"[a-z_]+=strategy_plugins" -- setup.py
git -C "$GH/agent-multi" grep -n -I -E "heuristic_strategy|prediction_entry_exit|plugin_long_short" -- '*.py' ':!docs/*' || echo "agent-multi: no reference"
} 2>&1 | red
