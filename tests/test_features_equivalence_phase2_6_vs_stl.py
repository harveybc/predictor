import sys
import json
from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
from preprocessor_plugins.stl_preprocessor import PreprocessorPlugin as STLPreprocessor

# Load configs from real files

with open('examples/config/phase_2/phase_2_6_cnn_1h_config.json') as f:
    phase2_6_config = json.load(f)
with open('examples/config/phase_3_1_daily/phase_3_1_cnn_1d_config.json') as f:
    stl_config = json.load(f)

# Limit to first 1000 rows for all splits
for cfg in (phase2_6_config, stl_config):
    cfg["max_steps_train"] = 1000
    cfg["max_steps_val"] = 1000
    cfg["max_steps_test"] = 1000

# Run preprocessors
print("\n=== Running Phase2_6 Preprocessor ===")
phase2_6 = Phase26Preprocessor()
out2_6 = phase2_6.run_preprocessing(phase2_6_config)
print("\n=== Running STL Preprocessor ===")
stl = STLPreprocessor()
outstl = stl.run_preprocessing(stl_config)

# Compare feature names
features2_6 = out2_6["feature_names"]
featuresstl = outstl["feature_names"]
print("\nPhase2_6 features:", features2_6)
print("STL features:", featuresstl)
if features2_6 == featuresstl:
    print("\nSUCCESS: Feature name/order match!")
else:
    print("\nFAIL: Feature name/order mismatch!")
    sys.exit(1)

# Compare shapes
for split in ["x_train", "x_val", "x_test"]:
    arr2_6 = out2_6[split]
    arrstl = outstl[split]
    print(f"{split} shapes: phase2_6={arr2_6.shape}, stl={arrstl.shape}")
    if arr2_6.shape != arrstl.shape:
        print(f"FAIL: Shape mismatch in {split}: {arr2_6.shape} vs {arrstl.shape}")
        sys.exit(1)

print("\nAll feature names and shapes match between phase2_6 and STL preprocessors!")
