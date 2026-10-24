"""Regression for the px.er_k producer at flat prices (Musashi's SAC-engine note, 2026-10-24): ``px.er_k`` divides
|log C_t/C_{t-k}| by the summed |1-bar log return| path over the same k bars, with a zero path replaced by NaN.
For k = 1 numerator and denominator are the same quantity, so the column is identically 1.0 wherever the close
moved and NaN at every zero-close-change bar: a constant-when-defined column whose only information is its NaN
pattern. This test pins that behaviour (it is the defect, not the fix): any change of convention needs a new
store identity and is decided in the recipe, not silently here."""
import numpy as np
import pandas as pd

from tools import timing_price_pool as P


def bars(closes):
    c = np.asarray(closes, float)
    idx = pd.date_range("2024-03-04", periods=len(c), freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": c, "HIGH": c + 0.0002, "LOW": c - 0.0002, "CLOSE": c}, index=idx)


def test_px_er_1_is_constant_one_where_defined_and_nan_at_zero_close_change():
    c = 1.1 + 0.001 * np.sin(np.arange(300) / 7.0)
    c[100] = c[99]                      # one flat bar (zero close change)
    c[200:203] = c[199]                 # three flat bars in a row
    out = P.price_pool(bars(c))
    er1 = out["px.er_1"].to_numpy()
    flat = np.r_[False, np.diff(c) == 0]
    assert np.isnan(er1[0])                                   # warm-up
    assert np.all(np.isnan(er1[flat]))                        # 0/0 -> NaN exactly at the flat bars
    defined = np.isfinite(er1)
    assert defined[1:].sum() == (~flat[1:]).sum()             # and nowhere else
    assert np.allclose(er1[defined], 1.0)                     # constant where defined


def test_px_er_k_is_nan_only_when_the_whole_k_bar_path_is_flat():
    c = np.full(120, 1.1)
    c[:20] = 1.1 + 0.001 * np.sin(np.arange(20) / 3.0)        # moving, then 100 flat bars (covers k = 48)
    out = P.price_pool(bars(c))
    for k in P.HORIZONS:
        col = out[f"px.er_{k}"].to_numpy()
        # returns are zero from index 21 on (c[20] still moves against c[19]); the k-bar path is all-zero from 20 + k
        assert np.all(np.isnan(col[20 + k:]))
        assert np.isfinite(col[20 + k - 1])                   # one moving return inside the path keeps it defined


def test_flat_bars_do_not_poison_neighbouring_columns():
    c = 1.1 + 0.001 * np.sin(np.arange(120) / 7.0)
    c[60] = c[59]
    out = P.price_pool(bars(c))
    for name in ("px.logret_1", "px.rv_24", "px.range_24", "px.bar_body"):
        assert np.isfinite(out[name].to_numpy()[60])
