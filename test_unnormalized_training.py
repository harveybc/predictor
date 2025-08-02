#!/usr/bin/env python3
"""
Test script to verify unnormalized targets training with simple model.
This will show if the unnormalized approach improves training convergence.
"""

import json
import numpy as np
from app.main import process_data
from app.plugin_loader import load_plugins

def test_unnormalized_training():
    """Test training with unnormalized targets."""
    print("="*80)
    print("TESTING UNNORMALIZED TARGETS TRAINING")
    print("="*80)
    
    # Load config
    with open('phase_6_cnn_1h_config.json', 'r') as f:
        config = json.load(f)
    
    # Reduce epochs for quick test
    config['epochs'] = 5
    config['batch_size'] = 64
    config['iterations'] = 1
    
    print(f"Config: epochs={config['epochs']}, batch_size={config['batch_size']}")
    print(f"Horizons: {config['predicted_horizons']}")
    print(f"Use returns: {config['use_returns']}")
    print(f"Window size: {config['window_size']}")
    
    try:
        # Load plugins
        plugins = load_plugins(config)
        predictor_plugin = plugins['predictor']
        preprocessor_plugin = plugins['preprocessor']
        pipeline_plugin = plugins['pipeline']
        
        # Set parameters
        predictor_plugin.set_params(**config)
        preprocessor_plugin.set_params(**config)
        pipeline_plugin.set_params(**config)
        
        print("\n--- Running preprocessor ---")
        # Get processed data
        datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
        
        # Verify targets are unnormalized
        print(f"\nTarget normalization stats:")
        target_means = preprocessor_params['target_returns_means']
        target_stds = preprocessor_params['target_returns_stds']
        print(f"  Means: {target_means}")
        print(f"  Stds: {target_stds}")
        
        # Check sample target values
        h1_targets = datasets['y_train']['output_horizon_1']
        print(f"\nSample H1 targets (first 10): {h1_targets[:10]}")
        print(f"H1 target stats: mean={np.mean(h1_targets):.6f}, std={np.std(h1_targets):.6f}")
        
        # Check baseline values  
        baseline = datasets['baseline_train']
        print(f"Baseline stats: mean={np.mean(baseline):.6f}, std={np.std(baseline):.6f}")
        
        # Verify target+baseline gives reasonable price predictions
        sample_predictions = baseline[:10] + h1_targets[:10]
        print(f"Sample predicted prices (baseline + H1 targets): {sample_predictions}")
        
        print("\n--- Starting training ---")
        # Run a short training to verify loss behavior
        X_train = datasets["x_train"]
        y_train_dict = {f"output_horizon_{h}": datasets["y_train"][f"output_horizon_{h}"].reshape(-1, 1) 
                       for h in config['predicted_horizons']}
        X_val = datasets["x_val"] 
        y_val_dict = {f"output_horizon_{h}": datasets["y_val"][f"output_horizon_{h}"].reshape(-1, 1)
                     for h in config['predicted_horizons']}
        
        print(f"Training data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        for h in config['predicted_horizons']:
            print(f"  y_train_H{h}: {y_train_dict[f'output_horizon_{h}'].shape}")
        
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
        
        history, _, _, _, _ = predictor_plugin.train(
            X_train, y_train_dict, 
            epochs=config['epochs'], 
            batch_size=config['batch_size'],
            threshold_error=0.001,
            x_val=X_val, 
            y_val=y_val_dict, 
            config=config
        )
        
        print("\n--- Training Results ---")
        print("Loss history:")
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"\nFinal losses:")
        print(f"  Train: {final_train_loss:.6f}")
        print(f"  Val: {final_val_loss:.6f}")
        
        # Check if losses are in reasonable range for unnormalized returns
        if final_train_loss < 0.01:  # Should be much lower than previous ~0.56
            print("✅ SUCCESS: Training loss is much lower than previous normalized approach!")
        else:
            print(f"⚠️  WARNING: Training loss still high: {final_train_loss:.6f}")
        
        print("\n--- Test Prediction ---")
        # Make a test prediction
        X_test = datasets["x_test"][:5]  # Just 5 samples
        predictions, uncertainties = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=10)
        
        print("Sample predictions (raw returns):")
        for i, h in enumerate(config['predicted_horizons']):
            pred_sample = predictions[i][:3]  # First 3 predictions
            print(f"  H{h}: {pred_sample}")
        
        # Convert to prices by adding baseline
        baseline_test = datasets['baseline_test'][:5]
        print("\nSample predicted prices:")
        for i, h in enumerate(config['predicted_horizons']):
            pred_returns = predictions[i][:3]
            pred_prices = baseline_test[:3] + pred_returns.flatten()
            print(f"  H{h}: {pred_prices}")
        
        print("\n" + "="*80)
        print("✅ UNNORMALIZED TARGETS TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR during training test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unnormalized_training()
