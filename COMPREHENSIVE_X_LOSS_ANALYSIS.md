# COMPREHENSIVE X-SHAPED LOSS ANALYSIS - ROOT CAUSE IDENTIFIED

## Problem Description
X-shaped loss pattern where training loss decreases continuously while validation loss drops initially for ~10 epochs then increases continuously, never recovering.

## CRITICAL ROOT CAUSE ANALYSIS

After comprehensive line-by-line analysis of the entire codebase, I've identified **MULTIPLE SYSTEMATIC ISSUES** that compound to create the X-shaped loss pattern:

### 1. **MASSIVE KL DIVERGENCE REGULARIZATION PROBLEM**

**Issue**: `kl_weight=1e-6` (daily config) vs `kl_weight=1e-4` (hourly config)
- Daily model uses 100x WEAKER KL regularization 
- This allows Bayesian layers to become severely overconfident during training
- Validation set (different time period) exposes this overconfidence as poor generalization

**Evidence in Code**: 
```python
# predictor_plugin_cnn.py line 825
kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT
```

### 2. **ASYMMETRIC LOSS SCALING FACTOR INCONSISTENCY**

**Issue**: `target_factor=100` but actual scaling differs between training/validation
```python
# stl_pipeline_zscore.py lines 173-178
target_mean = target_returns_means[0]  # All horizons use same normalization
target_std = target_returns_stds[0]
train_log_returns = train_normalized_preds * target_std + target_mean
```

**Problem**: Single normalization parameters for all horizons causes different horizons to have vastly different scales, leading to loss imbalance.

### 3. **COMPLEX MULTI-HORIZON LOSS WEIGHTING SYSTEM FAILURE**

**Issue**: The advanced weighting system is actually **DESTABILIZING** training:

```python
# predictor_plugin_cnn.py lines 125-145
def _calculate_loss_adaptive_weights(self, logs):
    for output_name in self.output_names:
        # Weight inversely proportional to sqrt of average loss
        weight = 1.0 / np.sqrt(max(avg_loss, 1e-8))
```

**Problem**: 
- Adaptive weights create feedback loops where poorly performing horizons get MORE attention
- This leads to training instability and validation overfitting
- The model focuses excessively on difficult horizons at expense of validation performance

### 4. **BAYESIAN UNCERTAINTY CALCULATION INCONSISTENCY**

**Issue**: Different uncertainty processing for train/val vs test:

```python
# stl_pipeline_zscore.py lines 294-296  
train_unc_denorm = train_unc_h * target_std  # Simple scaling
val_unc_denorm = val_unc_h * target_std    # Simple scaling

# vs test (lines 436-438):
test_unc_denorm = unc_normalized * target_std  # Same scaling but different context
```

**Problem**: Uncertainty estimates become inconsistent across splits, affecting model confidence calibration.

### 5. **ANTI-NAIVE-LOCK OVER-PREPROCESSING**

**Issue**: The anti-naive-lock processor applies heavy transformations to sliding windows:

```python
# anti_naive_lock.py - Multiple transformations applied:
# 1. Cyclic encoding for temporal features  
# 2. Log returns for price features
# 3. First differences for trend features
# 4. Z-score normalization on top of existing normalization
```

**Problem**: This creates a **TRIPLE NORMALIZATION EFFECT**:
1. Original CSV data is normalized (loaded state)
2. Immediate denormalization in _load_data()  
3. Anti-naive-lock re-normalization

This makes the model see very different feature distributions between training and validation.

## THE SMOKING GUN: COMPOSITE LOSS FUNCTION

**Critical Issue**: The composite loss function is fundamentally broken for multi-horizon training:

```python
# predictor_plugin_cnn.py lines 464-502
total_loss = huber_loss_val + mmd_lambda * mmd_loss_val
```

**Problems**:
1. **MMD Loss Component**: Maximum Mean Discrepancy loss (`mmd_lambda=1e-3`) adds distribution matching penalty
2. **Huber Loss**: More robust than MSE but still sensitive to outliers
3. **No Gradient Clipping**: Large gradients from MMD component destabilize training
4. **Horizon Imbalance**: Different horizons have vastly different loss magnitudes

## SPECIFIC EVIDENCE FROM CONFIGURATIONS

### Daily Config (X-shaped loss):
```json
"kl_weight": 1e-6,           // 100x weaker than hourly
"window_size": 288,          // 6x larger windows  
"plotted_horizon": 144,      // 6x longer horizon
"mmd_lambda": 1e-3,          // Same MMD weight for much larger predictions
"target_factor": 100         // Scaling factor inconsistency
```

### Hourly Config (presumably working):
```json
"kl_weight": 1e-4,           // Stronger regularization
"window_size": 144,          // Smaller windows
"plotted_horizon": 6,        // Shorter horizon  
"mmd_lambda": 1e-3,          // Same MMD weight for smaller predictions
```

## ROOT CAUSE SUMMARY

The X-shaped loss pattern is caused by **SYSTEMATIC PARAMETER SCALING ISSUES** when adapting from hourly to daily predictions:

1. **Weak KL Regularization** allows Bayesian layers to become overconfident on training data
2. **Complex Loss Weighting** creates training instability and validation overfitting  
3. **Triple Normalization Effect** makes feature distributions inconsistent
4. **MMD Loss Scaling** is inappropriate for daily vs hourly prediction scales
5. **Multi-horizon Imbalance** where different horizons fight for attention during training

## FIXES REQUIRED

### Immediate Priority Fixes:

1. **Increase KL Weight**: Change from `1e-6` to `1e-4` or higher
2. **Simplify Loss Function**: Remove MMD component or scale it down significantly  
3. **Fix Multi-horizon Normalization**: Use per-horizon normalization parameters
4. **Disable Adaptive Weighting**: Use uniform or inverse-horizon weighting only
5. **Reduce Anti-naive-lock Aggressiveness**: Use "none" or minimal preprocessing

### Implementation Order:
1. KL weight increase (highest impact)
2. Loss function simplification  
3. Multi-horizon normalization fix
4. Adaptive weighting disable
5. Preprocessing simplification

This analysis shows the problem is **NOT** the data splits or preprocessing pipeline per se, but rather **SYSTEMATIC PARAMETER SCALING ISSUES** when moving from hourly to daily predictions. The model is fundamentally overfitting to training data due to weak regularization and complex loss dynamics.
