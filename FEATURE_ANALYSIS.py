#!/usr/bin/env python
"""
REVERSE ENGINEERING: Exact Model Input Features Comparison
Phase 2.6 vs Phase 3.1 (STL) - What's Actually Fed to the Model

This script traces the exact features that end up in the sliding windows
for both phases to identify the critical differences.
"""

def analyze_phase_2_6_features():
    """
    Phase 2.6 Features Analysis (from preprocessor reverse engineering)
    """
    print("="*60)
    print("PHASE 2.6 MODEL INPUT FEATURES")
    print("="*60)
    
    print("\n1. DATA SOURCE:")
    print("   - Input: Pre-processed z-score normalized CSV files")
    print("   - Files: normalized_d4.csv, normalized_d5.csv, normalized_d6.csv")
    
    print("\n2. FEATURES CREATION:")
    print("   a) CLOSE column handling:")
    print("      - CLOSE: Extracted and denormalized for target calculation")
    print("      - CLOSE: REMOVED from features (not in sliding windows)")
    
    print("\n   b) close_logreturn creation:")
    print("      - close_logreturn = np.log(CLOSE).diff()")
    print("      - First value filled with 0")
    print("      - INCLUDED in features (in sliding windows)")
    
    print("\n   c) Other features:")
    print("      - ALL OTHER COLUMNS from CSV (except CLOSE)")
    print("      - Already preprocessed and z-score normalized")
    print("      - Likely includes: STL decompositions, wavelets, MTM, technical indicators")
    
    print("\n3. FINAL SLIDING WINDOW FEATURES:")
    print("   feature_columns = [col for col in x_train_df.columns if col != target_column]")
    print("   Therefore: ALL CSV columns EXCEPT 'CLOSE'")
    print("   Including: close_logreturn + all pre-processed features")
    
    print("\n4. KEY POINT:")
    print("   - close_logreturn: Created from NORMALIZED CLOSE (z-score)")
    print("   - Target: Calculated from DENORMALIZED CLOSE (actual prices)")
    print("   - This creates a mismatch in feature vs target domain!")

def analyze_phase_3_1_features():
    """
    Phase 3.1 (STL) Features Analysis (from preprocessor reverse engineering)
    """
    print("\n" + "="*60)
    print("PHASE 3.1 (STL) MODEL INPUT FEATURES")
    print("="*60)
    
    print("\n1. DATA SOURCE:")
    print("   - Input: Raw CSV files with CLOSE column")
    print("   - Files: x_train.csv, y_train.csv, x_val.csv, etc.")
    
    print("\n2. FEATURES CREATION:")
    print("   a) Generated features (in order):")
    print("      1. log_return: np.diff(np.log(close), prepend=log_close[0])")
    print("         - Created from raw CLOSE, then normalized")
    print("      2. STL features (if use_stl=True):")
    print("         - stl_trend, stl_seasonal, stl_resid")
    print("      3. Wavelet features (if use_wavelets=True):")
    print("         - wav_* features (multiple levels)")
    print("      4. MTM features (if use_multi_tapper=True):")
    print("         - mtm_* features")
    
    print("\n   b) Original columns:")
    print("      - ALL columns from input CSV EXCEPT 'CLOSE'")
    print("      - Sorted alphabetically")
    print("      - Aligned to match generated features length")
    
    print("\n3. FINAL SLIDING WINDOW FEATURES (in order):")
    print("   windowing_order = generated_feature_order + original_feature_order")
    print("   1. log_return (always first)")
    print("   2. stl_trend, stl_seasonal, stl_resid (if enabled)")
    print("   3. wav_* features (if enabled)")
    print("   4. mtm_* features (if enabled)")
    print("   5. All original columns alphabetically (except CLOSE)")
    
    print("\n4. KEY POINT:")
    print("   - log_return: Created from RAW CLOSE, then normalized")
    print("   - Target: Calculated from normalized CLOSE (min-max or z-score)")
    print("   - Feature and target domains are consistent!")

def identify_critical_differences():
    """
    Identify the exact differences causing performance issues
    """
    print("\n" + "="*60)
    print("CRITICAL DIFFERENCES ANALYSIS")
    print("="*60)
    
    print("\n🔴 PROBLEM 1: LOG RETURN CALCULATION DOMAIN")
    print("   Phase 2.6: close_logreturn from z-score normalized CLOSE")
    print("   Phase 3.1: log_return from raw CLOSE, then normalized")
    print("   → Different mathematical domains!")
    
    print("\n🔴 PROBLEM 2: FEATURE ORDER")
    print("   Phase 2.6: Arbitrary CSV column order (close_logreturn position unknown)")
    print("   Phase 3.1: Strict order (log_return ALWAYS FIRST)")
    print("   → Model sees different feature positions!")
    
    print("\n🔴 PROBLEM 3: FEATURE NORMALIZATION")
    print("   Phase 2.6: All features pre-normalized (z-score)")
    print("   Phase 3.1: Features normalized individually by STL preprocessor")
    print("   → Different normalization scales and methods!")
    
    print("\n🔴 PROBLEM 4: FEATURE SET COMPLETENESS")
    print("   Phase 2.6: Depends on what's in pre-processed CSV")
    print("   Phase 3.1: Generates features dynamically (STL, wavelets, MTM)")
    print("   → Potentially missing features in 2.6!")
    
    print("\n🔴 PROBLEM 5: TARGET-FEATURE CONSISTENCY")
    print("   Phase 2.6: Features z-score normalized, targets from denormalized CLOSE")
    print("   Phase 3.1: Features and targets use same normalization scheme")
    print("   → Mathematical inconsistency in 2.6!")

def proposed_solutions():
    """
    Proposed solutions to fix the issues
    """
    print("\n" + "="*60)
    print("PROPOSED SOLUTIONS")
    print("="*60)
    
    print("\n✅ SOLUTION 1: Fix log return calculation")
    print("   - Calculate close_logreturn from DENORMALIZED CLOSE")
    print("   - Then normalize it using same method as other features")
    
    print("\n✅ SOLUTION 2: Ensure feature order consistency")
    print("   - Force close_logreturn to be FIRST feature")
    print("   - Sort remaining features alphabetically")
    
    print("\n✅ SOLUTION 3: Verify feature completeness")
    print("   - Check CSV contains all STL, wavelet, MTM features")
    print("   - Add missing features if needed")
    
    print("\n✅ SOLUTION 4: Fix normalization consistency")
    print("   - Use same normalization for features and targets")
    print("   - Or denormalize features to match denormalized targets")
    
    print("\n✅ IMMEDIATE ACTION:")
    print("   1. Compare actual CSV column names vs STL feature names")
    print("   2. Fix close_logreturn calculation domain")
    print("   3. Enforce feature ordering")
    print("   4. Test with matching feature set")

if __name__ == "__main__":
    analyze_phase_2_6_features()
    analyze_phase_3_1_features()
    identify_critical_differences()
    proposed_solutions()
