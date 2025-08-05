#!/usr/bin/env python3
"""
Complete X-Shaped Loss Fix Implementation

This script implements ALL SEVEN critical fixes identified in the comprehensive analysis:
1. Fix KL weight (1e-6 → 1e-4)
2. Disable anti-naive-lock (selective → none) 
3. Scale down MMD (1e-3 → 1e-6)
4. Disable adaptive weighting (adaptive → uniform)
5. Shorten early stopping patience
6. Enable earlier monitoring
7. Disable triple normalization

Author: GitHub Copilot  
Date: 2025-08-04
"""

import json
import os
import shutil
from typing import Dict, Any

def backup_config(config_path: str) -> str:
    """Create backup of original config."""
    backup_path = config_path + ".backup"
    shutil.copy2(config_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def apply_critical_fixes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply all seven critical fixes to the configuration."""
    
    print("\n🔧 APPLYING CRITICAL FIXES:")
    
    # FIX #1: KL Weight Regularization
    old_kl = config.get('kl_weight', 'NOT_SET')
    config['kl_weight'] = 1e-4
    print(f"  1. KL Weight: {old_kl} → {config['kl_weight']} (100x STRONGER regularization)")
    
    # FIX #2: Disable Anti-Naive-Lock Destruction
    old_strategy = config.get('feature_preprocessing_strategy', 'NOT_SET')
    config['feature_preprocessing_strategy'] = 'none'
    print(f"  2. Anti-Naive-Lock: '{old_strategy}' → '{config['feature_preprocessing_strategy']}' (DISABLE feature destruction)")
    
    # FIX #3: Scale Down MMD Loss
    old_mmd = config.get('mmd_lambda', 'NOT_SET') 
    config['mmd_lambda'] = 1e-6
    print(f"  3. MMD Lambda: {old_mmd} → {config['mmd_lambda']} (1000x WEAKER distribution penalty)")
    
    # FIX #4: Disable Adaptive Weighting Chaos
    config['callback_weighting_strategy'] = 'uniform'
    print(f"  4. Loss Weighting: → 'uniform' (DISABLE adaptive chaos)")
    
    # FIX #5: Disable Triple Normalization
    old_norm = config.get('normalize_after_preprocessing', 'NOT_SET')
    config['normalize_after_preprocessing'] = False
    print(f"  5. Post-Processing Norm: {old_norm} → {config['normalize_after_preprocessing']} (DISABLE triple normalization)")
    
    # FIX #6: Improve Early Stopping
    old_patience = config.get('early_patience', 'NOT_SET')
    config['early_patience'] = 60  # Reduced from 120
    print(f"  6. Early Patience: {old_patience} → {config['early_patience']} (FASTER failure detection)")
    
    # FIX #7: Earlier Monitoring
    old_start = config.get('start_from_epoch', 'NOT_SET')
    config['start_from_epoch'] = 5  # Reduced from 15  
    print(f"  7. Start Monitoring: {old_start} → {config['start_from_epoch']} (EARLIER overfitting detection)")
    
    return config

def validate_fixes(config: Dict[str, Any]) -> bool:
    """Validate that all fixes were applied correctly."""
    
    print("\n🔍 VALIDATING FIXES:")
    
    checks = [
        ('kl_weight', 1e-4, "KL Weight Regularization"),
        ('feature_preprocessing_strategy', 'none', "Anti-Naive-Lock Disabled"),
        ('mmd_lambda', 1e-6, "MMD Loss Scaled Down"),
        ('callback_weighting_strategy', 'uniform', "Uniform Loss Weighting"),
        ('normalize_after_preprocessing', False, "Triple Normalization Disabled"),
        ('early_patience', 60, "Early Stopping Patience"),
        ('start_from_epoch', 5, "Early Monitoring")
    ]
    
    all_valid = True
    
    for param, expected, description in checks:
        actual = config.get(param)
        if actual == expected:
            print(f"  ✅ {description}: {actual}")
        else:
            print(f"  ❌ {description}: expected={expected}, actual={actual}")
            all_valid = False
    
    # Additional parameter reporting
    print("\n📊 ADDITIONAL PARAMETERS:")
    important_params = ['learning_rate', 'batch_size', 'epochs', 'window_size', 'plotted_horizon']
    for param in important_params:
        value = config.get(param, 'NOT_SET')
        print(f"  {param}: {value}")
    
    return all_valid

def compare_with_working_config(daily_config: Dict[str, Any]) -> None:
    """Compare fixed daily config with working hourly config."""
    
    print("\n🔄 COMPARISON WITH WORKING HOURLY CONFIG:")
    
    # Load hourly config for comparison  
    hourly_path = "examples/config/phase_6/phase_6_cnn_1h_config.json"
    if os.path.exists(hourly_path):
        with open(hourly_path, 'r') as f:
            hourly_config = json.load(f)
        
        critical_params = ['kl_weight', 'mmd_lambda', 'feature_preprocessing_strategy']
        
        print("  Parameter alignment check:")
        for param in critical_params:
            daily_val = daily_config.get(param, 'NOT_SET')
            hourly_val = hourly_config.get(param, 'NOT_SET')
            match = "✅ MATCH" if daily_val == hourly_val else "⚠️  DIFF"
            print(f"    {param}: daily={daily_val}, hourly={hourly_val} - {match}")
            
        # Scale differences (expected)
        scale_params = ['window_size', 'plotted_horizon', 'predicted_horizons']
        print("  Scale parameter differences (expected):")
        for param in scale_params:
            daily_val = daily_config.get(param, 'NOT_SET')
            hourly_val = hourly_config.get(param, 'NOT_SET')
            print(f"    {param}: daily={daily_val}, hourly={hourly_val}")
    else:
        print("  ⚠️  Hourly config not found for comparison")

def main():
    """Main fix implementation."""
    
    print("🚨 COMPREHENSIVE X-SHAPED LOSS FIX IMPLEMENTATION")
    print("=" * 60)
    
    # Configuration file path
    config_path = "examples/config/phase_6_daily/phase_6_cnn_1d_config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Configuration file not found: {config_path}")
        return False
    
    print(f"📁 Target configuration: {config_path}")
    
    # Step 1: Backup original config
    backup_path = backup_config(config_path)
    
    # Step 2: Load configuration
    try:
        config = load_config(config_path)
        print(f"✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading configuration: {e}")
        return False
    
    # Step 3: Apply all critical fixes
    config = apply_critical_fixes(config)
    
    # Step 4: Validate fixes
    if not validate_fixes(config):
        print("❌ VALIDATION FAILED - Fixes not applied correctly")
        return False
    
    # Step 5: Save fixed configuration
    try:
        save_config(config, config_path)
        print(f"\n✅ FIXED CONFIGURATION SAVED: {config_path}")
    except Exception as e:
        print(f"❌ ERROR saving configuration: {e}")
        return False
    
    # Step 6: Compare with working config
    compare_with_working_config(config)
    
    # Step 7: Final summary
    print("\n" + "=" * 60)
    print("🎯 FIX IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("✅ ALL SEVEN CRITICAL FIXES APPLIED:")
    print("   1. KL Weight: 1e-6 → 1e-4 (stronger regularization)")
    print("   2. Anti-Naive-Lock: selective → none (disable destruction)")  
    print("   3. MMD Lambda: 1e-3 → 1e-6 (scale down penalty)")
    print("   4. Loss Weighting: → uniform (disable adaptive chaos)")
    print("   5. Post-Processing: → false (disable triple normalization)")
    print("   6. Early Patience: 120 → 60 (faster detection)")
    print("   7. Start Monitoring: 15 → 5 (earlier detection)")
    print()
    print("🚀 READY TO TEST:")
    print(f"   python -m app.main --config_file {config_path}")
    print()
    print("📊 EXPECTED RESULTS:")
    print("   - Both training AND validation losses decreasing")
    print("   - No X-shaped divergence after epoch 10")
    print("   - Stable, converging loss curves")
    print("   - Proper Bayesian uncertainty calibration")
    print()
    print(f"🔄 RESTORE BACKUP IF NEEDED:")
    print(f"   cp {backup_path} {config_path}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
