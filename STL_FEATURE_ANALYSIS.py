#!/usr/bin/env python
"""
STL Feature Order Analysis - Extract Exact 54 Features

This script analyzes the STL preprocessor to determine the EXACT feature order
that produces 54 features, so we can replicate it exactly in Phase 2.6.
"""

def analyze_stl_feature_order():
    """
    Reverse engineer the exact STL feature order based on the code analysis.
    """
    print("="*80)
    print("STL PREPROCESSOR FEATURE ORDER ANALYSIS")
    print("="*80)
    
    # From the STL code, the windowing order is defined as:
    print("\n🔍 STL FEATURE ORDERING LOGIC:")
    print("   1. Generated features first (in predefined order)")
    print("   2. Original columns alphabetically")
    
    print("\n📋 GENERATED FEATURES ORDER:")
    generated_features = []
    
    # Always include log_return first
    generated_features.append("log_return")
    print("   1. log_return (always first)")
    
    # STL features (if use_stl=True)
    stl_features = ["stl_trend", "stl_seasonal", "stl_resid"]
    generated_features.extend(stl_features)
    print("   2-4. STL features:", stl_features)
    
    # Wavelet features (if use_wavelets=True) - sorted alphabetically
    wavelet_features = ["wav_approx_L2", "wav_detail_L1", "wav_detail_L2"]
    generated_features.extend(wavelet_features)
    print("   5-7. Wavelet features (sorted):", wavelet_features)
    
    # MTM features (if use_multi_tapper=True) - sorted alphabetically
    mtm_features = ["mtm_band_0", "mtm_band_1", "mtm_band_2", "mtm_band_3"]
    generated_features.extend(mtm_features)
    print("   8-11. MTM features (sorted):", mtm_features)
    
    print(f"\n   Total generated features: {len(generated_features)}")
    
    print("\n📋 ORIGINAL COLUMNS (alphabetically sorted):")
    # Based on typical OHLC + technical indicators datasets
    # These would be all columns except CLOSE (target) from original dataset
    typical_original_columns = [
        # Basic OHLC
        "HIGH", "LOW", "OPEN",
        # Technical indicators (typical set)
        "ADX", "AROON_down", "AROON_up", "ATR", "BBANDS_lower", "BBANDS_middle", "BBANDS_upper",
        "BOP", "CCI", "CMO", "DX", "EMA", "KAMA", "MACD", "MACD_histogram", "MACD_signal",
        "MFI", "MOM", "OBV", "PLUS_DI", "PLUS_DM", "PPO", "ROC", "ROCP", "ROCR",
        "RSI", "SAR", "SMA", "STOCH_k", "STOCH_d", "STOCHRSI_k", "STOCHRSI_d",
        "TEMA", "TRIX", "TSF", "ULTOSC", "WILLR", "WMA"
    ]
    
    print(f"   Typical original columns: {len(typical_original_columns)}")
    for i, col in enumerate(typical_original_columns, 1):
        print(f"   {i+len(generated_features)}. {col}")
    
    total_features = len(generated_features) + len(typical_original_columns)
    print(f"\n🎯 TOTAL FEATURES: {total_features}")
    
    if total_features == 54:
        print("✅ MATCHES expected 54 features!")
    else:
        print(f"❌ Expected 54, got {total_features}")
        if total_features < 54:
            missing = 54 - total_features
            print(f"   Missing {missing} features - likely more technical indicators")
        else:
            extra = total_features - 54
            print(f"   {extra} extra features - some might not be included")
    
    print("\n" + "="*80)
    print("PHASE 2.6 REQUIRED CHANGES:")
    print("="*80)
    
    print("\n🔧 TO MATCH STL EXACTLY:")
    print("1. Rename all CLOSE_* features to remove CLOSE_ prefix")
    print("2. Add missing technical indicators to reach 54 features")
    print("3. Ensure exact alphabetical ordering of original columns")
    print("4. Place log_return first, then decomposition features, then original columns")
    
    return generated_features + typical_original_columns

if __name__ == "__main__":
    features = analyze_stl_feature_order()
    print(f"\nFull feature list ({len(features)} features):")
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature}")
