#!/bin/bash
set -uo pipefail  # keep -u and pipefail; manage -e ourselves for per-item reporting

CONFIG_DIR="examples/config/phase_1"
RESULTS_DIR="examples/results/phase_1"
PROJECT_KEY="preliminar_forecast"
PHASE_KEY=$(basename "$CONFIG_DIR")

# Track outcomes
successes=0
failures=0
declare -a failed_items=()

for config_file in "$CONFIG_DIR"/*.json; do
  experiment_key=$(basename "$config_file" .json)
  results_file="$RESULTS_DIR/${experiment_key}_results.csv"
  predictions_file="$RESULTS_DIR/${experiment_key}_predictions.csv"
  uncertainties_file="$RESULTS_DIR/${experiment_key}_uncertainties.csv"

  echo "=========================================================="
  echo " Loading experiment: $experiment_key"
  echo "   Project : $PROJECT_KEY"
  echo "   Phase   : $PHASE_KEY"
  echo "   Config  : $config_file"
  echo "   Results : $results_file"
  echo "=========================================================="

  cmd=(python olap/etl_migrate_v2.py
       --project-key "$PROJECT_KEY"
       --phase-key "$PHASE_KEY"
       --experiment-key "$experiment_key"
       --experiment-config "$config_file"
       --results-csv "$results_file")

  if [[ -f "$predictions_file" && -f "$uncertainties_file" ]]; then
    cmd+=(--predictions-csv "$predictions_file" --uncertainties-csv "$uncertainties_file")
  fi

  # Run and capture exit code (do not abort loop)
  if "${cmd[@]}"; then
    ((successes++))
  else
    ((failures++))
    failed_items+=("$experiment_key")
  fi
done

echo "=========================================================="
echo "Phase: $PHASE_KEY  |  Successes: $successes  |  Failures: $failures"
if (( failures > 0 )); then
  echo "Failed experiments: ${failed_items[*]}"
  exit 1
fi
exit 0
