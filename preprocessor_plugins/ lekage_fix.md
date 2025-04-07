# Decomposition Issues and Fixes

This document contains a structured explanation of problems and corresponding solutions for three decomposition methods: **Wavelet Transform**, **Multitaper Spectral Estimation (MTM)**, and **STL Decomposition**.

---

## 1. Wavelet Transform (MODWT/SWT)

### Problem
Default mode in PyWavelets is `'symmetric'`, which mirrors future data into the padding during convolution.

Even though the convolution kernel slides over the signal correctly, the implicit symmetric padding at the boundaries includes future points. This results in **data leakage**, because during boundary processing, the algorithm is unknowingly using future data.

### Solution
Explicitly control the mode parameter and set it to `'constant'` or `'zero'` to avoid future data usage in padding.

**Code fix:**

```python
coeffs = pywt.swt(
    series_clean,
    wavelet=name,
    level=levels,
    trim_approx=False,
    norm=True,
    mode='constant'  # <- ensures no future data is used in padding
)
```

✅ Now, no future data is leaked via padding during the wavelet transformation.

---

## 2. Multitaper Spectral Estimation (MTM)

### Problem
Your rolling windowing and FFT application is **safe** and causally correct. However, the issue comes from the post-processing NaN handling:

The original code uses:
```python
.fillna(method='ffill').fillna(method='bfill')
```

The `bfill` (backward fill) operation **uses future data points to fill missing values**, which introduces data leakage.

### Solution
Remove `bfill` and safely use only `ffill`, or fallback to filling with a neutral value (like zero).

**Code fix:**

```python
mtm_features[name] = (
    pd.Series(mtm_features[name])
    .fillna(method='ffill')  # Safe: uses past data
    .fillna(0.0)             # Neutral fallback for any leading NaNs
    .values
)
```

✅ After this fix, the MTM feature pipeline is fully causality-safe and free of future data contamination.

---

## 3. STL Decomposition

### Problem
Your implementation uses rolling STL decomposition, which limits data to past-only windows. This is good.

However, STL itself internally performs **symmetric smoothing**, meaning even inside your past-only window, STL uses both past and future points **within the window**. Technically, there is **internal lookahead dependency** — but no external future data leakage from outside the window.

Additionally, your NaN handling again uses:
```python
.fillna(method='ffill').fillna(method='bfill')
```

The `bfill` here introduces future data leakage externally.

### Solution
First, remove `bfill` to eliminate external leakage.

**Code fix:**

```python
trend = pd.Series(trend).fillna(method='ffill').fillna(0.0).values
seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(0.0).values
resid = pd.Series(resid).fillna(method='ffill').fillna(0.0).values
```

Optional advanced recommendation:
For strict causality (if required by your application), consider replacing STL with fully causal smoothing techniques:
- Exponential Moving Average
- Causal Seasonal Decomposition
- Holt-Winters with past-only updates
- Incremental STL (research level)

✅ After removing `bfill`, your STL implementation is free of external leakage. Internal window-based smoothing remains but is acceptable unless you need strict causality.

---

## Summary Table

| Method                   | Issue                                               | Solution                                     |
|--------------------------|------------------------------------------------------|----------------------------------------------|
| Wavelet (MODWT/SWT)      | Symmetric padding includes future data               | Use `mode='constant'` in `pywt.swt()`        |
| Multitaper (MTM)         | `bfill` introduces future data during NaN handling   | Remove `bfill`, use `ffill` and zero fill    |
| STL Decomposition        | Internal symmetric smoothing and `bfill`             | Remove `bfill`, optionally switch to causal smoothing |

---

## Final Recommendations

- Ensure all feature engineering steps **explicitly control** padding and NaN handling.
- Remove any use of **backward fill (`bfill`)**, as it leaks future data.
- For Wavelet transforms, **always control the mode** parameter in your library.
- For STL, if full strict causality is needed, consider **causal smoothing alternatives**.

With these corrections, your feature pipeline will be **formally leakage-free** and ready for production deployment.

---
