#!/usr/bin/env python
"""
CSV Feature Analysis - Check what features are actually in the phase 2.6 CSV files
"""

import pandas as pd
import os

def analyze_csv_features():
    """
    Analyze the actual features in phase 2.6 CSV files
    """
    print("="*60)
    print("PHASE 2.6 CSV FEATURE ANALYSIS")
    print("="*60)
    
    csv_files = {
        "Train": "examples/data/phase_2_6/normalized_d4.csv",
        "Val": "examples/data/phase_2_6/normalized_d5.csv", 
        "Test": "examples/data/phase_2_6/normalized_d6.csv"
    }
    
    for name, file_path in csv_files.items():
        print(f"\n--- {name} Dataset: {file_path} ---")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        try:
            # Read just the header
            df = pd.read_csv(file_path, nrows=5)
            columns = list(df.columns)
            
            print(f"✅ Shape: {df.shape}")
            print(f"✅ Total columns: {len(columns)}")
            
            # Check for key columns
            has_close = 'CLOSE' in columns
            has_date = 'DATE_TIME' in columns or any('date' in col.lower() for col in columns)
            
            print(f"✅ Has CLOSE: {has_close}")
            print(f"✅ Has DATE: {has_date}")
            
            # Look for STL-like features
            stl_features = [col for col in columns if any(keyword in col.lower() for keyword in ['stl', 'trend', 'seasonal', 'resid'])]
            wav_features = [col for col in columns if 'wav' in col.lower()]
            mtm_features = [col for col in columns if 'mtm' in col.lower()]
            
            print(f"✅ STL-like features ({len(stl_features)}): {stl_features[:3]}{'...' if len(stl_features) > 3 else ''}")
            print(f"✅ Wavelet features ({len(wav_features)}): {wav_features[:3]}{'...' if len(wav_features) > 3 else ''}")
            print(f"✅ MTM features ({len(mtm_features)}): {mtm_features[:3]}{'...' if len(mtm_features) > 3 else ''}")
            
            # Show first few columns
            print(f"✅ First 10 columns: {columns[:10]}")
            print(f"✅ Last 10 columns: {columns[-10:]}")
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    analyze_csv_features()
