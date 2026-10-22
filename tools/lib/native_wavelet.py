# Vendored from financial-data, branch satoshi/b-native-wavelet-20261001, file _scripts/lib/native_wavelet.py,
# origin commit 307ded100 ("Native wavelet: copy the input to a writable array ... regression test").
# Copied verbatim below this header (no functional change) so that predictor's layer-2 pool reuses the genuine
# PyWavelets trailing-window producer instead of the legacy stage-2.3 moving-average proxy.
"""Native wavelet producer (successor to the Stage 2.3 multiscale proxy): PyWavelets DWT on past-only windows.

For every bar t with a complete trailing window [t-W+1, t] of finite inputs, `pywt.wavedec` is applied to that
window alone, and the right-edge (most recent) coefficient of each band is emitted at t, together with the mean
squared detail coefficient per level (band energy). Nothing after t is read, the window is a declared parameter
frozen from TRAIN (never derived from the input length), every bar is evaluated (cadence 1, feature age 0), and
missing inputs make the window's outputs NaN instead of being interpolated. Because each value is a function of
its own window only, prefixes, restarts and chunked replays (with W-1 rows of overlap) emit identical values.

Method identity: method_id NATIVE, method_name DWT_DB4_TRAILING_WINDOW (for wavelet='db4'). The Stage 2.3
`compute_wavelet` remains PROXY (rolling means 16/32/64/128) and is not edited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

UNITS = {"approximation": "input units scaled by the wavelet's analysis filter (price units for close)",
         "detail": "input units scaled by the wavelet's analysis filter (price units for close)",
         "energy": "squared input units (mean of squared detail coefficients in the window)"}


def compute_wavelet_native(df: pd.DataFrame, *, window: int, wavelet: str, level: int, frozen_from: dict,
                           sample_interval_seconds: int, column: str = "close", mode: str = "symmetric"):
    """-> (frame with timestamp + emitted columns, metadata). `window` and `frozen_from` are required."""
    if not frozen_from or not frozen_from.get("fold_id") or not frozen_from.get("train_end"):
        raise ValueError("frozen_from must name the fold and train_end the window was frozen from")
    if type(window) is not int or window < 2:
        raise ValueError("window must be an integer >= 2")
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(window, w.dec_len)
    if not 1 <= level <= max_level:
        raise ValueError(f"level {level} is not supported by a window of {window} for {wavelet} (max {max_level})")
    # a writable copy: pandas >= 3 can hand out read-only views, which PyWavelets' Cython rejects
    x = np.array(pd.to_numeric(df[column], errors="coerce"), dtype=float, copy=True)
    n = x.shape[0]
    names = [f"wavelet_native_A{level}"] + [f"wavelet_native_D{lev}" for lev in range(level, 0, -1)] \
        + [f"wavelet_native_energy_D{lev}" for lev in range(level, 0, -1)]
    out = np.full((n, len(names)), np.nan)
    finite = np.isfinite(x)
    bad = np.concatenate([[0], np.cumsum(~finite)])
    for t in range(window - 1, n):
        if bad[t + 1] - bad[t + 1 - window]:
            continue
        coeffs = pywt.wavedec(x[t + 1 - window:t + 1], w, level=level, mode=mode)
        out[t, 0] = coeffs[0][-1]
        for k in range(1, level + 1):
            out[t, k] = coeffs[k][-1]
            out[t, level + k] = float(np.mean(coeffs[k] ** 2))
    frame = pd.DataFrame(out, columns=names)
    frame.insert(0, "timestamp", df["timestamp"].to_numpy() if "timestamp" in df else np.arange(n))
    meta = {"method_id": "NATIVE", "method_name": f"DWT_{wavelet.upper()}_TRAILING_WINDOW", "proxy_of": None,
            "native_coverage_claimed": True, "library": "pywt", "version": pywt.__version__, "wavelet": wavelet,
            "level": level, "mode": mode, "window": window, "window_frozen_from": dict(frozen_from),
            "inputs_transformed": [column], "sample_interval_seconds": sample_interval_seconds,
            "update_cadence_bars": 1, "max_feature_age_bars": 0, "emission": "right-edge coefficient of each band at t",
            "units": UNITS, "past_only": "each value uses rows [t-window+1, t] only; NaN inside the window -> NaN"}
    return frame, meta
