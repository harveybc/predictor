# CRITICAL ISSUES AND TODO TASKS

## EXECUTIVE SUMMARY
The preprocessor and pipeline have **FUNDAMENTAL ARCHITECTURAL FLAWS** that cause terrible performance. These issues must be fixed systematically.

---

## 🚨 CRITICAL ERRORS IDENTIFIED

### 1. **BASELINE TEMPORAL MISALIGNMENT** (CRITICAL)
**Problem**: Baselines extracted from sliding windows don't correspond to correct time indices for target calculation.

**Location**: `stl_preprocessor_zscore.py` lines 234-243
**Impact**: Targets calculated with wrong baseline times → completely invalid training data
**Status**: ❌ NOT FIXED

**Current Code Issue**:
```python
# WRONG: Using sliding window dates as baseline dates
baseline_dates = sliding_windows[dates_key]
baselines[f'baseline_{split}_dates'] = baseline_dates
```

**Required Fix**: Map sliding window indices back to original time series indices correctly.

### 2. **DATA CARDINALITY MISMATCH** (CRITICAL)
**Problem**: Target calculation produces different numbers of samples per horizon, breaking model training.

**Location**: `target_calculation.py` lines 89-120
**Impact**: Model receives mismatched X/Y data shapes → training fails
**Status**: ❌ NOT FIXED

**Current Issue**: max_samples calculation truncates data differently for each horizon.

### 3. **DOUBLE SLIDING WINDOW CONFUSION** (CRITICAL)
**Problem**: The "two sliding windows" architecture is implemented incorrectly.

**Location**: `stl_preprocessor_zscore.py` lines 72-81
**Impact**: Model receives anti-naive-lock transformed sliding windows but targets calculated from different sliding windows
**Status**: ❌ PARTIALLY IMPLEMENTED

**Architectural Error**: Targets and features come from different data processing paths.

### 4. **ANTI-NAIVE-LOCK FEATURE MISMATCH** (HIGH)
**Problem**: Feature categorization logic is flawed, causing wrong transformations.

**Location**: `stl_preprocessor_zscore.py` lines 286-315
**Impact**: Features transformed incorrectly → poor model performance
**Status**: ❌ NEEDS DEBUGGING

### 5. **NORMALIZATION PARAMETER LOSS** (HIGH)
**Problem**: Individual horizon normalization parameters not properly preserved and used.

**Location**: Multiple files
**Impact**: Predictions can't be denormalized correctly → wrong real-world values
**Status**: ❌ NOT IMPLEMENTED

### 6. **BASELINE PRESERVATION FAILURE** (MEDIUM)
**Problem**: Baselines are lost during final output preparation.

**Location**: `stl_preprocessor_zscore.py` lines 430-445
**Impact**: Prediction reconstruction impossible
**Status**: ❌ ATTEMPTED BUT BROKEN

---

## 📋 SYSTEMATIC TODO LIST

### PHASE 1: CRITICAL ARCHITECTURE FIXES (MUST DO FIRST)

#### TODO-001: Fix Baseline Temporal Alignment
- [ ] **CRITICAL**: Implement correct baseline time mapping in `_extract_baselines_from_windows()`
- [ ] Map sliding window indices to original time series indices correctly
- [ ] Ensure baseline dates correspond to actual baseline extraction times
- [ ] Test temporal alignment with sample data

#### TODO-002: Fix Data Cardinality Consistency
- [ ] **CRITICAL**: Modify target calculation to ensure ALL horizons produce same number of samples
- [ ] Implement horizon-agnostic max_samples calculation
- [ ] Add validation to ensure X and Y data have matching sample counts
- [ ] Test with all horizon combinations

#### TODO-003: Implement Correct Two-Window Architecture
- [ ] **CRITICAL**: Clarify which sliding windows are used for what:
  - First sliding windows: ONLY for baseline extraction and target calculation
  - Second sliding windows: ONLY for model input (after anti-naive-lock)
- [ ] Ensure targets use denormalized price data throughout
- [ ] Validate data flow consistency

#### TODO-004: Fix Anti-Naive-Lock Feature Detection
- [ ] **HIGH**: Debug feature categorization logic
- [ ] Add logging to show which features get which transformations
- [ ] Implement proper case-insensitive feature matching
- [ ] Test with actual feature names from data

#### TODO-005: Implement Individual Horizon Normalization
- [ ] **HIGH**: Store normalization parameters per horizon during training
- [ ] Modify prediction pipeline to use correct normalization per horizon
- [ ] Test prediction denormalization with real data
- [ ] Validate real-world price reconstruction

### PHASE 2: DATA VALIDATION AND CONSISTENCY

#### TODO-006: Add Comprehensive Data Validation
- [ ] **MEDIUM**: Validate input data shapes and types
- [ ] Check for NaN/Inf values throughout pipeline
- [ ] Ensure date consistency across all data structures
- [ ] Add meaningful error messages for data issues

#### TODO-007: Fix Baseline Preservation
- [ ] **MEDIUM**: Properly store and retrieve baselines in final output
- [ ] Ensure baselines correspond to correct time indices
- [ ] Test baseline-based prediction reconstruction
- [ ] Validate against manual calculations

#### TODO-008: Standardize Error Handling
- [ ] **LOW**: Add try-catch blocks with proper fallbacks
- [ ] Implement graceful degradation for edge cases
- [ ] Add informative error messages
- [ ] Log warnings for data quality issues

### PHASE 3: PERFORMANCE AND OPTIMIZATION

#### TODO-009: Optimize Sliding Window Creation
- [ ] **LOW**: Remove redundant sliding window calculations
- [ ] Optimize memory usage for large datasets
- [ ] Parallelize feature processing where possible
- [ ] Profile and benchmark performance

#### TODO-010: Add Debug and Monitoring
- [ ] **LOW**: Add comprehensive logging throughout pipeline
- [ ] Implement data flow visualization
- [ ] Add performance metrics and timing
- [ ] Create data quality reports

---

## 🔧 IMMEDIATE ACTION PLAN

### Step 1: Stop Current Development
- **DO NOT** continue with current implementation
- **DO NOT** test until critical issues are fixed
- **DO NOT** add new features until architecture is correct

### Step 2: Fix Critical Issues (TODO-001 to TODO-003)
1. Start with TODO-001: Fix baseline temporal alignment
2. Then TODO-002: Fix data cardinality consistency  
3. Finally TODO-003: Implement correct two-window architecture

### Step 3: Validate and Test
- Test each fix with small dataset
- Validate data flow manually
- Ensure X/Y shapes match
- Check temporal alignment

### Step 4: Only Then Proceed
- Continue with remaining TODO items
- Add optimizations and features
- Perform full system testing

---

## 🎯 SUCCESS CRITERIA

### For TODO-001 (Baseline Alignment):
- Baselines correspond to correct time indices
- Baseline dates match sliding window extraction times
- Manual verification shows correct temporal mapping

### For TODO-002 (Data Cardinality):
- ALL horizons produce exactly the same number of samples
- X_train.shape[0] == y_train[horizon_1].shape[0] == y_train[horizon_N].shape[0]
- No data truncation differences between horizons

### For TODO-003 (Two-Window Architecture):
- First sliding windows used ONLY for targets
- Second sliding windows used ONLY for model input
- Clear separation of concerns
- Consistent data flow documentation

---

## ⚠️ WARNINGS

1. **PERFORMANCE WILL REMAIN TERRIBLE** until TODO-001, TODO-002, and TODO-003 are completed
2. **DO NOT SKIP** the critical architecture fixes
3. **TEST EACH FIX** individually before proceeding
4. **VALIDATE MANUALLY** with small datasets

---

## 📊 CURRENT STATUS SUMMARY

- **Architecture**: ❌ FUNDAMENTALLY BROKEN
- **Data Flow**: ❌ INCONSISTENT  
- **Temporal Alignment**: ❌ WRONG
- **Feature Processing**: ❌ UNRELIABLE
- **Target Calculation**: ❌ INVALID
- **Baseline Handling**: ❌ BROKEN
- **Error Handling**: ❌ INSUFFICIENT

**OVERALL STATUS**: 🚨 **CRITICAL FAILURE - REQUIRES COMPLETE REWORK**

The system cannot produce valid results until these critical architectural issues are resolved.
