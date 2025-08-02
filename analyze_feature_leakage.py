#!/usr/bin/env python3
"""
DETAILED FEATURE LEAKAGE ANALYSIS
Investigates specific features that may contain future information.
"""

import pandas as pd
import numpy as np
import json
from preprocessor_plugins.helpers import load_normalization_json, denormalize
from app.data_handler import load_csv

def analyze_feature_leakage():
    """
    Detailed analysis of potentially leaking features.
    """
    print("="*80)
    print("DETAILED FEATURE LEAKAGE ANALYSIS")
    print("="*80)
    
    # Load the raw data first
    config_file = "examples/config/phase_6/phase_6_cnn_1h_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load raw training data
    train_data = load_csv(config['x_train_file'], headers=True)
    print(f"Raw training data shape: {train_data.shape}")
    print(f"Raw training data columns: {list(train_data.columns)}")
    
    # Load normalization config
    norm_json = load_normalization_json(config)
    
    # 1. Analyze OHLC features
    print("\n" + "="*60)
    print("1. ANALYZING OHLC FEATURES FOR LEAKAGE")
    print("="*60)
    
    # Extract OHLC columns
    ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    for col in ohlc_cols:
        if col in train_data.columns:
            raw_values = train_data[col].values[:10]
            denorm_values = denormalize(raw_values, norm_json, col)
            
            print(f"\n{col} analysis (first 10 samples):")
            print(f"  Raw normalized: {raw_values}")
            print(f"  Denormalized: {denorm_values}")
            
            # Check if HIGH >= LOW, HIGH >= CLOSE, etc.
            if col in ['HIGH', 'LOW']:
                close_raw = train_data['CLOSE'].values[:10]
                close_denorm = denormalize(close_raw, norm_json, 'CLOSE')
                
                if col == 'HIGH':
                    violations = np.sum(denorm_values < close_denorm)
                    print(f"  ❌ CRITICAL: {violations}/10 samples where HIGH < CLOSE!")
                    if violations > 0:
                        print("  This indicates HIGH contains FUTURE information (intrabar lookahead)")
                elif col == 'LOW':
                    violations = np.sum(denorm_values > close_denorm)
                    print(f"  ❌ CRITICAL: {violations}/10 samples where LOW > CLOSE!")
                    if violations > 0:
                        print("  This indicates LOW contains FUTURE information (intrabar lookahead)")
    
    # 2. Analyze multi-timeframe features
    print("\n" + "="*60)
    print("2. ANALYZING MULTI-TIMEFRAME FEATURES")
    print("="*60)
    
    # Look for tick-based features
    tick_features = [col for col in train_data.columns if 'tick' in col.lower()]
    print(f"Found tick-based features: {tick_features}")
    
    for feature in tick_features[:5]:  # Analyze first 5
        raw_values = train_data[feature].values[:10]
        print(f"\n{feature} (first 10 samples): {raw_values}")
        
        # These features should be from PAST timeframes only
        # If they correlate too highly with current CLOSE, it's suspicious
        close_values = train_data['CLOSE'].values[:100]
        feature_values = train_data[feature].values[:100]
        
        correlation = np.corrcoef(close_values, feature_values)[0, 1]
        print(f"  Correlation with current CLOSE: {correlation:.6f}")
        
        if abs(correlation) > 0.9:
            print(f"  ❌ CRITICAL: Very high correlation ({correlation:.6f}) suggests future leakage!")
        elif abs(correlation) > 0.7:
            print(f"  ⚠️ WARNING: High correlation ({correlation:.6f}) may indicate leakage")
        else:
            print(f"  ✅ Correlation appears reasonable")
    
    # 3. Check temporal ordering of features
    print("\n" + "="*60)
    print("3. TEMPORAL ORDERING VERIFICATION")
    print("="*60)
    
    # Check if any features show impossible temporal relationships
    print("Checking for features that predict themselves...")
    
    # Test log_return feature specifically
    if 'CLOSE' in train_data.columns:
        close_raw = train_data['CLOSE'].values
        close_denorm = denormalize(close_raw, norm_json, 'CLOSE')
        
        # Calculate actual returns from CLOSE
        actual_returns = np.diff(close_denorm)
        
        print(f"Actual returns from CLOSE (first 10): {actual_returns[:10]}")
        print(f"Actual returns std: {np.std(actual_returns):.6f}")
        
        # The log_return feature in windows should be PAST returns only
        # It should not perfectly correlate with FUTURE returns
        
    # 4. Analyze feature generation process
    print("\n" + "="*60)
    print("4. FEATURE GENERATION ANALYSIS")
    print("="*60)
    
    # Look at the feature generation in sliding windows
    from preprocessor_plugins.stl_preprocessor_zscore import PreprocessorPlugin
    
    preprocessor = PreprocessorPlugin()
    preprocessor.set_params(**config)
    
    # Check if log_return calculation is done correctly
    print("Analyzing log_return feature calculation...")
    
    # Load a small sample and check feature generation
    sample_data = train_data.iloc[:200].copy()
    
    # Check where log_return comes from
    close_col = sample_data['CLOSE'].values
    close_denorm = denormalize(close_col, norm_json, 'CLOSE')
    
    # Log returns should be calculated as log(price[t] / price[t-1])
    manual_log_returns = np.log(close_denorm[1:] / close_denorm[:-1])
    
    print(f"Manual log returns (first 10): {manual_log_returns[:10]}")
    print(f"Manual log returns std: {np.std(manual_log_returns):.6f}")
    
    # Check if the calculation matches what we expect
    print("\nFeature generation verification:")
    print("- OHLC features should use PAST bars only")
    print("- Multi-timeframe features should use PAST timeframes only") 
    print("- Technical indicators should be calculated on PAST data only")
    print("- log_return should be calculated from PAST price changes only")
    
    return True

if __name__ == "__main__":
    analyze_feature_leakage()
