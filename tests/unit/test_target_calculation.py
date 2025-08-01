import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import json

# Add the project root to the path so we can import modules
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

from preprocessor_plugins.target_calculation import TargetCalculationProcessor
from preprocessor_plugins.helpers import denormalize


class TestTargetCalculation(unittest.TestCase):
    """
    Test to verify that target calculation produces correct values that exactly
    match the shifted close prices when reconstructed.
    """

    def setUp(self):
        """Set up test environment with synthetic data."""
        self.processor = TargetCalculationProcessor()
        
        # Configuration
        self.config = {
            "target_column": "CLOSE",
            "predicted_horizons": [1, 3, 6],  # Test multiple horizons
            "window_size": 10,
            "use_returns": True
        }
        
        # Create synthetic price data - simple linear trend for predictability
        self.n_points = 100
        self.dates = pd.date_range(start='2023-01-01', periods=self.n_points, freq='h')
        
        # Linear trend: price = 100 + 0.1 * t (so horizon 1 = +0.1, horizon 3 = +0.3, etc.)
        self.close_prices = 100.0 + 0.1 * np.arange(self.n_points)
        
        # Create dataframes for each split (using same data for simplicity)
        self.x_df_train = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        self.x_df_val = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        self.x_df_test = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        self.y_df_train = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        self.y_df_val = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        self.y_df_test = pd.DataFrame({'CLOSE': self.close_prices}, index=self.dates)
        
        # Prepare baseline data structure
        self.baseline_data = {
            'x_train_df': self.x_df_train,
            'x_val_df': self.x_df_val,
            'x_test_df': self.x_df_test,
            'y_train_df': self.y_df_train,
            'y_val_df': self.y_df_val,
            'y_test_df': self.y_df_test,
            'dates_train': self.dates,
            'dates_val': self.dates,
            'dates_test': self.dates,
            'norm_json': {}  # No normalization for this test - the helper will use identity transformation
        }
        
        # Calculate expected number of samples for windowing
        # The sliding window algorithm: num_possible_windows = n - window_size - max_horizon + 1
        max_horizon = max(self.config['predicted_horizons'])
        window_size = self.config['window_size']
        expected_samples = self.n_points - window_size - max_horizon + 1
        
        self.windowed_data = {
            'num_samples_train': expected_samples,
            'num_samples_val': expected_samples,
            'num_samples_test': expected_samples
        }

    def test_target_price_reconstruction_exact_match(self):
        """
        Core test: Verify that baseline + denormalized_target exactly equals 
        the true future price for all horizons.
        """
        print("\\n=== Testing Target Price Reconstruction ===")
        
        # Run target calculation
        target_data = self.processor.calculate_targets(
            self.baseline_data, self.windowed_data, self.config
        )
        
        # Extract results
        y_test = target_data['y_test']
        baseline_test = target_data['baseline_test']
        target_means = target_data['target_returns_mean']
        target_stds = target_data['target_returns_std']
        
        print(f"Target means: {target_means}")
        print(f"Target stds: {target_stds}")
        print(f"Baseline test shape: {baseline_test.shape}")
        print(f"Number of horizons: {len(self.config['predicted_horizons'])}")
        
        for i, horizon in enumerate(self.config['predicted_horizons']):
            print(f"\\n--- Testing Horizon {horizon} ---")
            
            # Get normalized target for this horizon
            horizon_key = f"output_horizon_{horizon}"
            normalized_target = y_test[horizon_key]
            
            print(f"Normalized target shape: {normalized_target.shape}")
            print(f"Sample normalized values: {normalized_target[:5]}")
            
            # Denormalize the target using the stats calculated by the processor
            mean_h = target_means[i]
            std_h = target_stds[i]
            denormalized_return = (normalized_target * std_h) + mean_h
            
            print(f"Denormalized return shape: {denormalized_return.shape}")
            print(f"Sample denormalized returns: {denormalized_return[:5]}")
            print(f"Sample baselines: {baseline_test[:5]}")
            
            # Reconstruct the target price: baseline + return
            reconstructed_target_price = baseline_test + denormalized_return
            
            print(f"Reconstructed target price shape: {reconstructed_target_price.shape}")
            print(f"Sample reconstructed prices: {reconstructed_target_price[:5]}")
            
            # Calculate what the true future price should be
            window_size = self.config['window_size']
            num_samples = self.windowed_data['num_samples_test']
            
            # The true future price at horizon h
            # baseline corresponds to close_prices[window_size-1 : window_size-1+num_samples]
            # future should be close_prices[window_size-1+horizon : window_size-1+num_samples+horizon]
            baseline_start_idx = window_size - 1
            baseline_end_idx = baseline_start_idx + num_samples
            future_start_idx = baseline_start_idx + horizon
            future_end_idx = baseline_end_idx + horizon
            
            print(f"Baseline indices: {baseline_start_idx} to {baseline_end_idx}")
            print(f"Future indices: {future_start_idx} to {future_end_idx}")
            
            expected_baseline = self.close_prices[baseline_start_idx:baseline_end_idx]
            expected_future_price = self.close_prices[future_start_idx:future_end_idx]
            
            print(f"Expected baseline shape: {expected_baseline.shape}")
            print(f"Expected future price shape: {expected_future_price.shape}")
            print(f"Sample expected baselines: {expected_baseline[:5]}")
            print(f"Sample expected future prices: {expected_future_price[:5]}")
            
            # Verify baseline matches (using reasonable tolerance for floating-point arithmetic)
            np.testing.assert_allclose(
                baseline_test, expected_baseline, rtol=1e-5, atol=1e-5,
                err_msg=f"Baseline mismatch for horizon {horizon}"
            )
            print(f"✓ Baseline verification passed for horizon {horizon}")
            
            # THE CORE TEST: Verify reconstructed price matches expected future price
            np.testing.assert_allclose(
                reconstructed_target_price, expected_future_price, rtol=1e-5, atol=1e-5,
                err_msg=f"Target price reconstruction failed for horizon {horizon}"
            )
            print(f"✓ Target price reconstruction passed for horizon {horizon}")
            
            # Additional verification: check that the return calculation was correct
            expected_return = expected_future_price - expected_baseline
            np.testing.assert_allclose(
                denormalized_return, expected_return, rtol=1e-5, atol=1e-5,
                err_msg=f"Return calculation failed for horizon {horizon}"
            )
            print(f"✓ Return calculation verification passed for horizon {horizon}")
        
        print("\\n🎉 ALL TARGET RECONSTRUCTION TESTS PASSED! 🎉")

    def test_normalization_stats_calculation(self):
        """
        Test that normalization statistics are calculated correctly from training data.
        """
        print("\\n=== Testing Normalization Stats Calculation ===")
        
        target_data = self.processor.calculate_targets(
            self.baseline_data, self.windowed_data, self.config
        )
        
        target_means = target_data['target_returns_mean']
        target_stds = target_data['target_returns_std']
        
        # For our linear data, all returns should be constant
        # horizon 1: return = 0.1, horizon 3: return = 0.3, horizon 6: return = 0.6
        expected_means = [0.1 * h for h in self.config['predicted_horizons']]
        expected_stds = [0.0] * len(self.config['predicted_horizons'])  # Should be 0 for constant returns, but clamped to 1.0
        
        print(f"Expected means: {expected_means}")
        print(f"Actual means: {target_means}")
        print(f"Expected stds: {expected_stds}")
        print(f"Actual stds: {target_stds}")
        
        # Check means match expected (using reasonable tolerance for float32 vs float64)
        np.testing.assert_allclose(target_means, expected_means, rtol=1e-6, atol=1e-6)
        print("✓ Normalization means are correct")
        
        # For constant returns, std should be close to 0 but clamped to 1.0
        for std_val in target_stds:
            self.assertGreaterEqual(std_val, 1e-8)  # Should be clamped
        print("✓ Normalization stds are properly clamped")

    def run_target_calculation_test(self):
        """Public method to run the test and return results for external verification."""
        try:
            self.setUp()
            self.test_target_price_reconstruction_exact_match()
            self.test_normalization_stats_calculation()
            return True, "All tests passed successfully"
        except Exception as e:
            return False, str(e)


def run_target_calculation_verification():
    """Standalone function to run target calculation verification."""
    test_instance = TestTargetCalculation()
    success, message = test_instance.run_target_calculation_test()
    print(f"\\n{'='*60}")
    if success:
        print("✅ TARGET CALCULATION VERIFICATION: SUCCESS")
    else:
        print("❌ TARGET CALCULATION VERIFICATION: FAILED")
        print(f"Error: {message}")
    print(f"{'='*60}")
    return success


if __name__ == '__main__':
    # Run the verification
    success = run_target_calculation_verification()
    
    # Also run as unittest for CI/CD integration
    unittest.main(argv=[''], exit=False, verbosity=2)
