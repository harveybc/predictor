#!/bin/bash
set -euo pipefail

# -------------------------------
# Settings
# -------------------------------
CONFIG_DIR="examples/config/phase_1"
RESULTS_DIR="examples/results/phase_1"
PROJECT_KEY="preliminar_forecast"

# Extract phase from directory name (e.g., phase_1)
PHASE_KEY=$(basename "$CONFIG_DIR")

# -------------------------------
# Batch load
# -------------------------------
for config_file in "$CONFIG_DIR"/*.json; do
    # experiment key is the filename without extension
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

    # Run ETL (skip optional files if not found)
    cmd=(python olap/etl_migrate_v2.py
        --project-key "$PROJECT_KEY"
        --phase-key "$PHASE_KEY"
        --experiment-key "$experiment_key"
        --experiment-config "$config_file"
        --results-csv "$results_file")

    if [ -f "$predictions_file" ] && [ -f "$uncertainties_file" ]; then
        cmd+=(--predictions-csv "$predictions_file" --uncertainties-csv "$uncertainties_file")
    fi

    "${cmd[@]}"
done

echo "✅ All experiments for $PHASE_KEY loaded into OLAP cube."


