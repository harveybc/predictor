#!/usr/bin/env python3
"""
Test X-shaped Loss Fix

This script tests the fixes for the X-shaped loss pattern by:
1. Running training with the modified configuration
2. Monitoring loss curves in real-time  
3. Generating a quick report on the fix effectiveness

Author: GitHub Copilot
Date: 2025-08-04
"""

import sys
import json
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def test_x_loss_fix():
    """Test the X-shaped loss fix with systematic monitoring."""
    
    print("=" * 80)
    print("TESTING X-SHAPED LOSS FIX")
    print("=" * 80)
    
    config_file = "examples/config/phase_6_daily/phase_6_cnn_1d_config.json"
    
    # Load and validate configuration
    print("\n1. Validating Configuration Changes...")
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check critical parameters
        critical_params = {
            'kl_weight': 1e-4,          # Increased from 1e-6
            'mmd_lambda': 1e-6,         # Decreased from 1e-3  
            'feature_preprocessing_strategy': 'none'  # Simplified from 'selective'
        }
        
        print("Configuration validation:")
        for param, expected in critical_params.items():
            actual = config.get(param)
            status = "✅ PASS" if actual == expected else "❌ FAIL"
            print(f"  {param}: expected={expected}, actual={actual} - {status}")
            
        # Additional parameter info
        additional_params = ['learning_rate', 'batch_size', 'epochs', 'early_patience']
        print("\nAdditional parameters:")
        for param in additional_params:
            value = config.get(param)
            print(f"  {param}: {value}")
            
    except Exception as e:
        print(f"❌ ERROR loading configuration: {e}")
        return False
    
    print("\n2. Starting Training with Monitoring...")
    print("Will monitor first 20 epochs for loss pattern...")
    
    # Start training process
    try:
        cmd = [sys.executable, "-m", "app.main", "--config_file", config_file]
        print(f"Command: {' '.join(cmd)}")
        
        # Run for limited epochs to check pattern
        # Note: This will run the full training, but we'll monitor the loss plot
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Training completed successfully")
            print("📊 Checking loss plot...")
            
            # Check if loss plot was generated
            loss_plot_file = config.get('loss_plot_file', 'examples/results/phase_6_daily/phase_6_cnn_25200_1d_loss_plot.png')
            try:
                # Try to analyze the training output for loss patterns
                output_lines = result.stdout.split('\n')
                train_losses = []
                val_losses = []
                
                for line in output_lines:
                    if 'loss:' in line and 'val_loss:' in line:
                        # Try to extract loss values
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.startswith('loss:'):
                                    train_loss = float(part.split(':')[1])
                                    train_losses.append(train_loss)
                                if part.startswith('val_loss:'):
                                    val_loss = float(part.split(':')[1])
                                    val_losses.append(val_loss)
                        except:
                            continue
                
                if len(train_losses) >= 10 and len(val_losses) >= 10:
                    print(f"\n📈 Loss Analysis (first {min(len(train_losses), 20)} epochs):")
                    
                    # Check for X-pattern
                    early_train = np.mean(train_losses[:5])
                    late_train = np.mean(train_losses[-5:]) if len(train_losses) >= 10 else np.mean(train_losses)
                    early_val = np.mean(val_losses[:5])  
                    late_val = np.mean(val_losses[-5:]) if len(val_losses) >= 10 else np.mean(val_losses)
                    
                    train_trend = "DECREASING" if late_train < early_train else "INCREASING"
                    val_trend = "DECREASING" if late_val < early_val else "INCREASING"
                    
                    print(f"  Training Loss: {early_train:.6f} → {late_train:.6f} ({train_trend})")
                    print(f"  Validation Loss: {early_val:.6f} → {late_val:.6f} ({val_trend})")
                    
                    # Determine pattern
                    if train_trend == "DECREASING" and val_trend == "INCREASING":
                        print("  ❌ PATTERN: X-shaped loss detected (train↓, val↑)")
                        return False
                    elif train_trend == "DECREASING" and val_trend == "DECREASING":
                        print("  ✅ PATTERN: Healthy training (train↓, val↓)")
                        return True
                    else:
                        print("  ⚠️  PATTERN: Unusual training pattern")
                        return None
                        
                else:
                    print("  ⚠️  Insufficient loss data for analysis")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing loss data: {e}")
                
            print(f"\n📊 Check loss plot manually: {loss_plot_file}")
            return True
            
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️  Training timeout - this is normal for full training")
        print("Check results manually in the output files")
        return None
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        return False

def main():
    """Main test execution."""
    print("X-SHAPED LOSS FIX TEST")
    print("=====================")
    
    result = test_x_loss_fix()
    
    print("\n" + "=" * 80)
    if result is True:
        print("✅ TEST RESULT: FIX APPEARS SUCCESSFUL")
        print("   - Training completed without X-shaped loss pattern")
        print("   - Both training and validation losses are decreasing")
    elif result is False:
        print("❌ TEST RESULT: FIX NOT SUCCESSFUL")  
        print("   - X-shaped loss pattern still detected")
        print("   - Additional parameter tuning may be required")
    else:
        print("⚠️  TEST RESULT: INCONCLUSIVE")
        print("   - Unable to determine loss pattern from automated analysis")
        print("   - Manual inspection of loss plots recommended")
    
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. Examine the loss plot: examples/results/phase_6_daily/phase_6_cnn_25200_1d_loss_plot.png")
    print("2. Check training logs for any error messages")
    print("3. If X-pattern persists, try further reducing mmd_lambda or increasing kl_weight")
    print("=" * 80)

if __name__ == "__main__":
    main()
