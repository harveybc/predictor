#!/usimport sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/predictor')
sys.path.append('/home/harveybc/Documents/GitHub/preprocessor/app')

from pipeline_plugins.stl_pipeline_zscore import STLPipelineZScore
import jsonnv python3
"""
Quick training test to verify performance with global normalization
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/predictor')
sys.path.append('/home/harveybc/Documents/GitHub/preprocessor/app')

from stl_pipeline_zscore import STLPipelineZScore
import json

def test_training_performance():
    """Test training with global normalization to verify performance improvement"""
    
    print("🔥 Testing Training Performance with Global Normalization...")
    
    # Load config
    config_path = "/home/harveybc/Documents/GitHub/predictor/input_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up for quick test
    config['epochs'] = 3  # Just a few epochs to test
    config['batch_size'] = 32
    config['validation_split'] = 0.2
    
    print(f"Config loaded: {len(config)} parameters")
    print(f"Epochs: {config['epochs']}")
    print(f"Predicted horizons: {config['predicted_horizons']}")
    
    # Create pipeline
    pipeline = STLPipelineZScore(config)
    
    # Run training
    print("\n--- Starting Training Test ---")
    try:
        result = pipeline.train()
        print(f"\n✅ Training completed successfully!")
        print(f"Training result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Check for training metrics
        if isinstance(result, dict) and 'history' in result:
            history = result['history']
            if 'loss' in history:
                final_loss = history['loss'][-1] if history['loss'] else 'N/A'
                print(f"Final training loss: {final_loss}")
            
            # Check validation metrics for each horizon
            for horizon in config['predicted_horizons']:
                val_metric = f'val_output_horizon_{horizon}_mae_magnitude'
                if val_metric in history:
                    final_val_mae = history[val_metric][-1] if history[val_metric] else 'N/A'
                    print(f"Final validation MAE (horizon {horizon}): {final_val_mae}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_performance()
    print(f"\n{'='*60}")
    if success:
        print("✅ TRAINING PERFORMANCE TEST: SUCCESS")
        print("Global normalization appears to be working!")
    else:
        print("❌ TRAINING PERFORMANCE TEST: FAILED")
        print("Need to investigate further...")
    print(f"{'='*60}")
