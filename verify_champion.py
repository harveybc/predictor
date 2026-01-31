#!/usr/bin/env python3
"""
Verify Champion Parameters

Re-evaluates the saved optimization parameters to calculate the actual fitness.
This helps verify if the parameters file truly corresponds to the reported fitness.
"""
import json
import sys

# Load the parameters
with open("examples/results/phase_1_daily/phase_1_mimo_1d_optimization_parameters.json") as f:
    params = json.load(f)

# Load the config - use the OPTIMIZATION config to ensure all settings match
with open("examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json") as f:
    config = json.load(f)

# Update config with optimized parameters
config.update(params)

print("=" * 80)
print("VERIFYING CHAMPION PARAMETERS")
print("=" * 80)
print("\nOptimized Parameters:")
print(json.dumps(params, indent=2))

print("\n" + "=" * 80)
print("Running evaluation with candidate_worker...")
print("=" * 80)

# Create input payload for candidate_worker
worker_input = {
    "gen": 999,  # Dummy generation
    "cand": 0,
    "config": config,
    "hyper": params
}

# Save to temporary file
import tempfile
import os
import subprocess

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in:
    json.dump(worker_input, f_in)
    input_file = f_in.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:
    output_file = f_out.name

try:
    # Run candidate_worker
    cmd = [
        sys.executable, "-m", "optimizer_plugins.candidate_worker",
        "--input", input_file,
        "--output", output_file
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    # Read output
    with open(output_file) as f:
        output = json.load(f)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"\nFitness: {output.get('fitness')}")
    print(f"Train MAE: {output.get('train_mae')}")
    print(f"Train Naive MAE: {output.get('train_naive_mae')}")
    print(f"Val MAE: {output.get('val_mae')}")
    print(f"Val Naive MAE: {output.get('naive_mae')}")
    print(f"Test MAE: {output.get('test_mae')}")
    print(f"Test Naive MAE: {output.get('test_naive_mae')}")
    
    # Calculate fitness manually
    if output.get('train_mae') and output.get('train_naive_mae') and output.get('val_mae') and output.get('naive_mae'):
        train_delta = output['train_mae'] - output['train_naive_mae']
        val_delta = output['val_mae'] - output['naive_mae']
        calculated_fitness = 0.5 * train_delta + 0.5 * val_delta
        print(f"\nCalculated Fitness: {calculated_fitness}")
        print(f"Reported Fitness:   {output.get('fitness')}")
        print(f"Match: {abs(calculated_fitness - output.get('fitness', float('inf'))) < 1e-10}")
    
    # Compare with optimization stats
    with open("examples/results/phase_1_daily/phase_1_mimo_1d_optimization_stats.json") as f:
        stats = json.load(f)
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH OPTIMIZATION STATS")
    print("=" * 80)
    print(f"\nStats Champion Fitness:     {stats['champion_fitness']}")
    print(f"Re-evaluated Fitness:       {output.get('fitness')}")
    print(f"Difference:                 {abs(stats['champion_fitness'] - output.get('fitness', 0))}")
    
    if abs(stats['champion_fitness'] - output.get('fitness', 0)) > 0.001:
        print("\n⚠️  WARNING: Large discrepancy detected!")
        print("The saved parameters do NOT reproduce the reported champion fitness.")
        print("This confirms a bug in the optimization process.")
    else:
        print("\n✓ Parameters match the reported fitness within tolerance.")
    
finally:
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)
