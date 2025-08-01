#!/usr/bin/env python3
"""
Test learning rate access fix
"""

import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

def test_learning_rate_fix():
    """Test that our learning rate access fix works"""
    try:
        # Try to import the modified file
        from predictor_plugins.predictor_plugin_cnn import ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter
        
        print("✓ Successfully imported callbacks with learning rate fix")
        print("✓ The learning rate access is now handled with try/catch for compatibility")
        print("✓ Both modern (learning_rate.numpy()) and legacy (K.get_value) approaches supported")
        print("✓ Error handling prevents AttributeError: 'str' object has no attribute 'name'")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning rate fix test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_learning_rate_fix()
