#!/bin/bash
set -uo pipefail  # keep -u and pipefail; manage -e ourselves for per-item reporting

PROJECT_KEY="preliminar_forecast"

# Global counters
total_successes=0
total_failures=0
total_skipped=0
declare -a failed_items=()

# Iterate over all phase directories under examples/config
for CONFIG_DIR in examples/config/phase_*; do
  # Skip if no such directories
  [[ -d "$CONFIG_DIR" ]] || continue

  PHASE_KEY=$(basename "$CONFIG_DIR")
  RESULTS_DIR="examples/results/$PHASE_KEY"

  # Per-phase counters (optional: could be printed per phase)
  phase_successes=0
  phase_failures=0
  phase_skipped=0

  # Loop over all experiment config JSONs in this phase
  shopt -s nullglob
  phase_configs=("$CONFIG_DIR"/*.json)
  shopt -u nullglob

  if (( ${#phase_configs[@]} == 0 )); then
    echo "No config files found in $CONFIG_DIR; skipping."
    continue
  fi

  for config_file in "${phase_configs[@]}"; do
    experiment_key=$(basename "$config_file" .json)

    # Remove "_config" suffix for result file naming
    experiment_base="${experiment_key%_config}"

    results_file="$RESULTS_DIR/${experiment_base}_results.csv"
    predictions_file="$RESULTS_DIR/${experiment_base}_predictions.csv"
    uncertainties_file="$RESULTS_DIR/${experiment_base}_uncertainties.csv"

    echo "=========================================================="
    echo " Loading experiment: $experiment_key"
    echo "   Project : $PROJECT_KEY"
    echo "   Phase   : $PHASE_KEY"
    echo "   Config  : $config_file"
    echo "   Results : $results_file"
    echo "=========================================================="

    # Skip if results CSV is missing
    if [[ ! -f "$results_file" ]]; then
      echo "[SKIP] Missing results CSV: $results_file"
      ((phase_skipped++))
      ((total_skipped++))
      continue
    fi

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
      ((phase_successes++))
      ((total_successes++))
    else
      ((phase_failures++))
      ((total_failures++))
      failed_items+=("$PHASE_KEY/$experiment_key")
    fi
  done

  echo "=========================================================="
  echo "Phase: $PHASE_KEY  |  Successes: $phase_successes  |  Failures: $phase_failures  |  Skipped: $phase_skipped"
done

echo "=========================================================="
echo "All phases complete | Successes: $total_successes | Failures: $total_failures | Skipped: $total_skipped"
if (( total_failures > 0 )); then
  echo "Failed experiments: ${failed_items[*]}"
  exit 1
fi
exit 0
