#!/usr/bin/env python3

import json
import sys
import os

# Add the current directory to Python path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_config_anti_naive_lock():
    """Test if anti-naive-lock is enabled in the config file"""
    
    print("============================================================")
    print("TESTING CONFIG ANTI-NAIVE-LOCK SETTING")
    print("============================================================")
    
    # Load the config file
    config_path = "/home/harveybc/Documents/GitHub/predictor/examples/config/phase_6/phase_6_cnn_1h_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        print(f"Config loaded successfully from: {config_path}")
        print(f"anti_naive_lock_enabled: {config.get('anti_naive_lock_enabled', 'NOT FOUND')}")
        print(f"feature_preprocessing_strategy: {config.get('feature_preprocessing_strategy', 'NOT FOUND')}")
        
        # Test the condition from anti_naive_lock.py
        anti_naive_enabled = config.get('anti_naive_lock_enabled', False)
        print(f"\nCondition check: config.get('anti_naive_lock_enabled', False) = {anti_naive_enabled}")
        
        if not anti_naive_enabled:
            print("❌ Anti-naive-lock would be DISABLED with this config")
        else:
            print("✅ Anti-naive-lock would be ENABLED with this config")
            
        # Check strategy
        strategy = config.get('feature_preprocessing_strategy', 'selective')
        print(f"Strategy: {strategy}")
        
        if strategy == 'none':
            print("❌ Strategy is 'none' - preprocessing would be skipped")
        else:
            print(f"✅ Strategy is '{strategy}' - preprocessing would be applied")
            
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n============================================================")
    print("TEST COMPLETE")
    print("============================================================")

if __name__ == "__main__":
    test_config_anti_naive_lock()
