"""
Anti-Naive-Lock Preprocessing Module

This module implements selective feature preprocessing to prevent naive lock
where the model simply learns to copy input features to outputs.

The module applies different preprocessing strategies based on feature types:
1. Cyclic encoding for temporal features
2. Log returns for raw price features  
3. First differences for trend features
4. Preserve already-stationary technical indicators
5. Special handling for constant daily features

Author: GitHub Copilot
Date: 2025-08-03
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

class AntiNaiveLockProcessor:
    """
    Processor for applying selective preprocessing to prevent naive lock.
    
    Implements different transformation strategies based on feature characteristics
    to ensure model learns meaningful patterns rather than simple input copying.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_stats = {}  # Store statistics for inverse transforms if needed
        
    def process_sliding_windows(self, 
                               x_train: np.ndarray, 
                               x_val: np.ndarray, 
                               x_test: np.ndarray,
                               feature_names: List[str],
                               config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply anti-naive-lock preprocessing to sliding window matrices.
        
        Args:
            x_train: Training sliding windows (samples, time_steps, features)
            x_val: Validation sliding windows (samples, time_steps, features)  
            x_test: Test sliding windows (samples, time_steps, features)
            feature_names: List of feature column names
            config: Configuration dictionary
            
        Returns:
            Tuple of (processed_x_train, processed_x_val, processed_x_test, processing_stats)
        """
        if not config.get('anti_naive_lock_enabled', False):
            print("Anti-naive-lock preprocessing disabled")
            return x_train, x_val, x_test, {}
            
        print("Starting anti-naive-lock preprocessing...")
        
        # Get preprocessing strategy
        strategy = config.get('feature_preprocessing_strategy', 'selective')
        
        if strategy == 'none':
            return x_train, x_val, x_test, {}
        elif strategy == 'uniform_log_returns':
            return self._apply_uniform_log_returns(x_train, x_val, x_test, feature_names, config)
        elif strategy == 'selective':
            return self._apply_selective_preprocessing(x_train, x_val, x_test, feature_names, config)
        else:
            raise ValueError(f"Unknown preprocessing strategy: {strategy}")
    
    def _apply_selective_preprocessing(self, 
                                     x_train: np.ndarray,
                                     x_val: np.ndarray, 
                                     x_test: np.ndarray,
                                     feature_names: List[str],
                                     config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply selective preprocessing based on feature categories."""
        
        # Copy arrays to avoid modifying originals
        x_train_processed = x_train.copy()
        x_val_processed = x_val.copy() 
        x_test_processed = x_test.copy()
        
        processing_stats = {'applied_transforms': {}}
        
        # Get feature categories from config with defaults
        price_features = config.get('price_features', ['OPEN', 'LOW', 'HIGH', 'CLOSE'])
        temporal_features = config.get('temporal_features', ['day_of_week', 'hour_of_day', 'day_of_month'])
        trend_features = config.get('trend_features', ['stl_trend'])
        subperiodicity_features = config.get('subperiodicity_features', [
            'CLOSE_15m_tick_1', 'CLOSE_15m_tick_2', 'CLOSE_15m_tick_3', 'CLOSE_15m_tick_4',
            'CLOSE_15m_tick_5', 'CLOSE_15m_tick_6', 'CLOSE_15m_tick_7', 'CLOSE_15m_tick_8',
            'CLOSE_30m_tick_1', 'CLOSE_30m_tick_2', 'CLOSE_30m_tick_3', 'CLOSE_30m_tick_4',
            'CLOSE_30m_tick_5', 'CLOSE_30m_tick_6', 'CLOSE_30m_tick_7', 'CLOSE_30m_tick_8'
        ])
        constant_daily_features = config.get('constant_daily_features', ['S&P500_Close', 'vix_close'])
        
        print(f"Processing {len(feature_names)} features with selective strategy...")
        
        for i, feature_name in enumerate(feature_names):
            try:
                if feature_name in price_features and config.get('use_log_returns', True):
                    # Apply log returns to raw price features
                    x_train_processed[:, :, i] = self._apply_log_returns(x_train_processed[:, :, i])
                    x_val_processed[:, :, i] = self._apply_log_returns(x_val_processed[:, :, i])
                    x_test_processed[:, :, i] = self._apply_log_returns(x_test_processed[:, :, i])
                    processing_stats['applied_transforms'][feature_name] = 'log_returns'
                    print(f"  Applied log returns to {feature_name}")
                    
                elif feature_name in temporal_features and config.get('use_cyclic_encoding', True):
                    # Apply cyclic encoding to temporal features (simplified version)
                    x_train_processed[:, :, i] = self._apply_cyclic_encoding_simplified(x_train_processed[:, :, i], feature_name)
                    x_val_processed[:, :, i] = self._apply_cyclic_encoding_simplified(x_val_processed[:, :, i], feature_name)
                    x_test_processed[:, :, i] = self._apply_cyclic_encoding_simplified(x_test_processed[:, :, i], feature_name)
                    processing_stats['applied_transforms'][feature_name] = 'cyclic_encoding'
                    print(f"  Applied cyclic encoding to {feature_name}")
                    
                elif feature_name in trend_features and config.get('use_first_differences', True):
                    # Apply first differences to trend features
                    x_train_processed[:, :, i] = self._apply_first_differences(x_train_processed[:, :, i])
                    x_val_processed[:, :, i] = self._apply_first_differences(x_val_processed[:, :, i])
                    x_test_processed[:, :, i] = self._apply_first_differences(x_test_processed[:, :, i])
                    processing_stats['applied_transforms'][feature_name] = 'first_differences'
                    print(f"  Applied first differences to {feature_name}")
                    
                elif feature_name in subperiodicity_features and config.get('use_log_returns', True):
                    # Apply log returns to sub-periodicity features
                    x_train_processed[:, :, i] = self._apply_log_returns(x_train_processed[:, :, i])
                    x_val_processed[:, :, i] = self._apply_log_returns(x_val_processed[:, :, i])
                    x_test_processed[:, :, i] = self._apply_log_returns(x_test_processed[:, :, i])
                    processing_stats['applied_transforms'][feature_name] = 'log_returns'
                    print(f"  Applied log returns to sub-periodicity {feature_name}")
                    
                elif feature_name in constant_daily_features and config.get('handle_constant_daily_features', True):
                    # Special handling for constant daily features
                    x_train_processed[:, :, i] = self._handle_constant_daily_feature(x_train_processed[:, :, i])
                    x_val_processed[:, :, i] = self._handle_constant_daily_feature(x_val_processed[:, :, i])
                    x_test_processed[:, :, i] = self._handle_constant_daily_feature(x_test_processed[:, :, i])
                    processing_stats['applied_transforms'][feature_name] = 'daily_differences'
                    print(f"  Applied daily differences to {feature_name}")
                    
                else:
                    # Keep feature as-is (technical indicators, decomposed features, etc.)
                    processing_stats['applied_transforms'][feature_name] = 'preserved'
                    
            except Exception as e:
                print(f"ERROR processing feature {feature_name}: {e}")
                processing_stats['applied_transforms'][feature_name] = f'error: {str(e)}'
        
        # Apply final normalization per feature to ensure consistent scales
        if config.get('normalize_after_preprocessing', True):
            x_train_processed, x_val_processed, x_test_processed, norm_stats = self._apply_feature_normalization(
                x_train_processed, x_val_processed, x_test_processed, feature_names
            )
            processing_stats['normalization_stats'] = norm_stats
        
        transforms_applied = len([k for k, v in processing_stats['applied_transforms'].items() if v != 'preserved'])
        print(f"Anti-naive-lock preprocessing completed. Transforms applied to {transforms_applied} features")
        
        return x_train_processed, x_val_processed, x_test_processed, processing_stats
    
    def _apply_log_returns(self, feature_data: np.ndarray) -> np.ndarray:
        """
        Apply log returns transformation: ln(x_t / x_{t-1})
        
        Args:
            feature_data: 2D array (samples, time_steps)
            
        Returns:
            Transformed data with same shape
        """
        # Avoid log(0) and negative values
        feature_data_safe = np.where(feature_data <= 0, 1e-8, feature_data)
        
        # Calculate log returns along time axis
        log_returns = np.zeros_like(feature_data_safe)
        log_returns[:, 1:] = np.log(feature_data_safe[:, 1:] / feature_data_safe[:, :-1])
        log_returns[:, 0] = 0  # First time step has no previous value
        
        # Handle any remaining NaN/inf values
        log_returns = np.where(np.isfinite(log_returns), log_returns, 0)
        
        return log_returns
    
    def _apply_cyclic_encoding_simplified(self, feature_data: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Apply simplified cyclic encoding: sin(2π * x / period)
        Only returns sin component to keep same dimensionality.
        
        Args:
            feature_data: 2D array (samples, time_steps)
            feature_name: Name of feature to determine period
            
        Returns:
            Sin-encoded data with same shape
        """
        # Determine period based on feature name
        if 'hour_of_day' in feature_name:
            period = 24
        elif 'day_of_week' in feature_name:
            period = 7
        elif 'day_of_month' in feature_name:
            period = 31  # Approximate, could be made more sophisticated
        else:
            period = np.max(feature_data) + 1  # Fallback: use data range
        
        # Apply cyclic encoding (sin component only)
        angle = 2 * np.pi * feature_data / period
        sin_encoded = np.sin(angle)
        
        return sin_encoded
    
    def _apply_first_differences(self, feature_data: np.ndarray) -> np.ndarray:
        """
        Apply first differences transformation: x_t - x_{t-1}
        
        Args:
            feature_data: 2D array (samples, time_steps)
            
        Returns:
            Transformed data with same shape
        """
        differences = np.zeros_like(feature_data)
        differences[:, 1:] = feature_data[:, 1:] - feature_data[:, :-1]
        differences[:, 0] = 0  # First time step has no previous value
        
        return differences
    
    def _handle_constant_daily_feature(self, feature_data: np.ndarray) -> np.ndarray:
        """
        Handle features that are constant within each day (like S&P500, VIX).
        Use differences between days instead of returns.
        
        Args:
            feature_data: 2D array (samples, time_steps)
            
        Returns:
            Transformed data emphasizing daily changes
        """
        # For features constant within day, use a longer-term difference
        # This could be improved with knowledge of the exact daily structure
        differences = np.zeros_like(feature_data)
        
        # Use a 24-step difference assuming hourly data (could be configurable)
        step_size = 24
        differences[:, step_size:] = feature_data[:, step_size:] - feature_data[:, :-step_size]
        
        return differences
    
    def _apply_uniform_log_returns(self, 
                                  x_train: np.ndarray,
                                  x_val: np.ndarray,
                                  x_test: np.ndarray,
                                  feature_names: List[str],
                                  config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply log returns to all features uniformly."""
        
        x_train_processed = x_train.copy()
        x_val_processed = x_val.copy()
        x_test_processed = x_test.copy()
        
        processing_stats = {'applied_transforms': {}}
        
        print(f"Applying uniform log returns to all {len(feature_names)} features...")
        
        for i, feature_name in enumerate(feature_names):
            try:
                x_train_processed[:, :, i] = self._apply_log_returns(x_train_processed[:, :, i])
                x_val_processed[:, :, i] = self._apply_log_returns(x_val_processed[:, :, i])
                x_test_processed[:, :, i] = self._apply_log_returns(x_test_processed[:, :, i])
                processing_stats['applied_transforms'][feature_name] = 'log_returns'
            except Exception as e:
                print(f"ERROR applying uniform log returns to {feature_name}: {e}")
                processing_stats['applied_transforms'][feature_name] = f'error: {str(e)}'
        
        print("Applied uniform log returns to all features")
        return x_train_processed, x_val_processed, x_test_processed, processing_stats
    
    def _apply_feature_normalization(self, 
                                   x_train: np.ndarray,
                                   x_val: np.ndarray, 
                                   x_test: np.ndarray,
                                   feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply feature-wise z-score normalization after preprocessing.
        
        Args:
            x_train, x_val, x_test: Sliding window matrices
            feature_names: Feature names for logging
            
        Returns:
            Normalized matrices and normalization statistics
        """
        norm_stats = {}
        
        print("Applying post-processing normalization...")
        
        # Calculate statistics from training data only
        for i in range(x_train.shape[2]):
            # Flatten across samples and time steps for each feature
            train_feature_flat = x_train[:, :, i].flatten()
            
            # Remove any NaN or infinite values for statistics calculation
            valid_mask = np.isfinite(train_feature_flat)
            if np.any(valid_mask):
                train_feature_clean = train_feature_flat[valid_mask]
                feature_mean = np.mean(train_feature_clean)
                feature_std = np.std(train_feature_clean)
                
                # Avoid division by zero
                if feature_std < 1e-8:
                    feature_std = 1.0
                    
                # Apply normalization to all datasets
                x_train[:, :, i] = (x_train[:, :, i] - feature_mean) / feature_std
                x_val[:, :, i] = (x_val[:, :, i] - feature_mean) / feature_std
                x_test[:, :, i] = (x_test[:, :, i] - feature_mean) / feature_std
                
                norm_stats[feature_names[i]] = {'mean': feature_mean, 'std': feature_std}
            else:
                print(f"WARNING: Feature {feature_names[i]} has no valid values for normalization")
                norm_stats[feature_names[i]] = {'mean': 0.0, 'std': 1.0}
        
        return x_train, x_val, x_test, norm_stats
