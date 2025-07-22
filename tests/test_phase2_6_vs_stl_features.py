import os
import numpy as np
import pandas as pd
from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
from preprocessor_plugins.stl_preprocessor import PreprocessorPlugin as STLPreprocessor

# Test config (adjust paths as needed)
PHASE2_6_CONFIG = {
    "x_train_file": "examples/data/phase_2_6/normalized_d4.csv",
    "y_train_file": "examples/data/phase_2_6/normalized_d4.csv",
    "x_validation_file": "examples/data/phase_2_6/normalized_d5.csv",
    "y_validation_file": "examples/data/phase_2_6/normalized_d5.csv",
    "x_test_file": "examples/data/phase_2_6/normalized_d6.csv",
    "y_test_file": "examples/data/phase_2_6/normalized_d6.csv",
    "use_normalization_json": "examples/data/phase_2_6/normalization_config_b.json",
    "target_column": "CLOSE",
    "window_size": 144,
    "predicted_horizons": [1,2,3,4,5,6],
    "expected_feature_count": 55,
    "headers": True
}

STL_CONFIG = {
    "x_train_file": "examples/data/phase_3_1/x_train.csv",
    "y_train_file": "examples/data/phase_3_1/y_train.csv",
    "x_validation_file": "examples/data/phase_3_1/x_val.csv",
    "y_validation_file": "examples/data/phase_3_1/y_val.csv",
    "x_test_file": "examples/data/phase_3_1/x_test.csv",
    "y_test_file": "examples/data/phase_3_1/y_test.csv",
    "use_normalization_json": "examples/data/phase_3_1/normalization_config.json",
    "target_column": "CLOSE",
    "window_size": 144,
    "predicted_horizons": [1,2,3,4,5,6],
    "headers": True
}

def test_feature_equivalence():
    print("\n=== Running Phase2_6 Preprocessor ===")
    phase2_6 = Phase26Preprocessor()
    out2_6 = phase2_6.run_preprocessing(PHASE2_6_CONFIG)
    print("\n=== Running STL Preprocessor ===")
    stl = STLPreprocessor()
    outstl = stl.run_preprocessing(STL_CONFIG)

    # Compare feature names
    features2_6 = out2_6["feature_names"]
    featuresstl = outstl["feature_names"]
    print("\nPhase2_6 features:", features2_6)
    print("STL features:", featuresstl)
    assert features2_6 == featuresstl, f"Feature name/order mismatch!\nPhase2_6: {features2_6}\nSTL: {featuresstl}"

    # Compare shapes
    for split in ["x_train", "x_val", "x_test"]:
        arr2_6 = out2_6[split]
        arrstl = outstl[split]
        print(f"{split} shapes: phase2_6={arr2_6.shape}, stl={arrstl.shape}")
        assert arr2_6.shape == arrstl.shape, f"Shape mismatch in {split}: {arr2_6.shape} vs {arrstl.shape}"

    print("\nAll feature names and shapes match between phase2_6 and STL preprocessors!")

if __name__ == "__main__":
    test_feature_equivalence()
