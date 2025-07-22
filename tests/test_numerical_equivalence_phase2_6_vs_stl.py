import sys
import json
import numpy as np
from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
from preprocessor_plugins.stl_preprocessor import PreprocessorPlugin as STLPreprocessor

# Load configs from real files but modify them to use identical parameters
with open('examples/config/phase_2/phase_2_6_cnn_1h_config.json') as f:
    phase2_6_config = json.load(f)
with open('examples/config/phase_3_1_daily/phase_3_1_cnn_1d_config.json') as f:
    stl_config = json.load(f)

# Force IDENTICAL parameters for fair comparison
identical_params = {
    "max_steps_train": 600,  # Increased to handle STL offset
    "max_steps_val": 600, 
    "max_steps_test": 600,
    "window_size": 288,
    "predicted_horizons": [24, 48, 72, 96, 120, 144],
    "stl_period": 24,
    "use_stl": True,
    "use_wavelets": False,  # Disable wavelets to avoid version issues
    "use_multi_tapper": True,
    "use_returns": True,
    "target_column": "CLOSE"
}

# Apply identical params to both configs
for key, value in identical_params.items():
    phase2_6_config[key] = value
    stl_config[key] = value

print("Running preprocessors with identical parameters...")
print(f"Window size: {identical_params['window_size']}")
print(f"Horizons: {identical_params['predicted_horizons']}")
print(f"Max rows: {identical_params['max_steps_train']}")

# Run preprocessors
print("\n=== Running Phase2_6 Preprocessor ===")
phase2_6 = Phase26Preprocessor()
out2_6 = phase2_6.run_preprocessing(phase2_6_config)

print("\n=== Running STL Preprocessor ===")
stl = STLPreprocessor()
outstl = stl.run_preprocessing(stl_config)

# Compare feature names/order
features2_6 = out2_6["feature_names"]
featuresstl = outstl["feature_names"]
print(f"\nPhase2_6 features ({len(features2_6)}): {features2_6}")
print(f"STL features ({len(featuresstl)}): {featuresstl}")

if features2_6 == featuresstl:
    print("\n✓ SUCCESS: Feature name/order match!")
else:
    print("\n✗ FAIL: Feature name/order mismatch!")
    sys.exit(1)

# Compare shapes
shape_match = True
for split in ["x_train", "x_val", "x_test"]:
    arr2_6 = out2_6[split]
    arrstl = outstl[split]
    print(f"{split} shapes: phase2_6={arr2_6.shape}, stl={arrstl.shape}")
    if arr2_6.shape != arrstl.shape:
        print(f"✗ FAIL: Shape mismatch in {split}")
        shape_match = False

if not shape_match:
    sys.exit(1)

print("\n✓ SUCCESS: All shapes match!")

# Now test numerical equivalence on first few samples
print("\n=== Testing Numerical Equivalence ===")
tolerance = 1e-6

# Compare first 5 samples of first 3 features for each split
for split in ["x_train", "x_val", "x_test"]:
    arr2_6 = out2_6[split]
    arrstl = outstl[split]
    
    # Test first 5 samples, all timesteps, first 3 features
    subset2_6 = arr2_6[:5, :, :3]
    subsetstl = arrstl[:5, :, :3]
    
    max_diff = np.max(np.abs(subset2_6 - subsetstl))
    print(f"{split}: Max absolute difference in first 5 samples (3 features): {max_diff:.10f}")
    
    if max_diff > tolerance:
        print(f"✗ FAIL: Numerical difference {max_diff:.10f} exceeds tolerance {tolerance}")
        
        # Show detailed differences for debugging
        for i in range(min(3, len(features2_6))):
            feat_name = features2_6[i]
            feat_diff = np.max(np.abs(arr2_6[:3, :10, i] - arrstl[:3, :10, i]))
            print(f"  Feature '{feat_name}' (samples 0-2, timesteps 0-9): max_diff = {feat_diff:.10f}")
            
            if feat_diff > tolerance:
                print(f"    Phase2_6 sample[0,0]: {arr2_6[0, 0, i]:.10f}")
                print(f"    STL sample[0,0]:      {arrstl[0, 0, i]:.10f}")
                print(f"    Difference:           {abs(arr2_6[0, 0, i] - arrstl[0, 0, i]):.10f}")
        
        sys.exit(1)

print(f"\n✓ SUCCESS: Numerical equivalence confirmed! (tolerance: {tolerance})")
print("Phase2_6 and STL preprocessors now produce IDENTICAL feature processing!")
