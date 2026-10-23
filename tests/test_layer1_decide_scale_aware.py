"""Scale-aware skew rule of tools/layer1_pool_h1.decide (rev 2026-10-22): the original rule applied asinh in native
units, the identity for magnitudes ~1e-3, so ranges / realized volatilities / energies stayed skewed."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.stats import skew  # noqa: E402

from tools.layer1_pool_h1 import decide, diagnostics  # noqa: E402


def _lognormal(n, scale, seed=0):
    return scale * np.exp(np.random.default_rng(seed).normal(0.0, 0.8, n))


def test_small_scale_positive_variable_gets_log_where_native_asinh_was_a_no_op():
    x = _lognormal(5000, 1e-3)
    tr = np.ones(len(x), bool)
    d = diagnostics(x[tr])
    assert d["skew"] > 2 and abs(skew(np.arcsinh(x))) > 0.5 * d["skew"]          # the old rule could not act
    dec, why, y = decide("px.range_1", x, d, train=tr)
    assert dec == "LOG" and np.allclose(y, np.log(x)) and abs(skew(y)) < 0.5 * d["skew"] and "TRAIN skew" in why


def test_pip_scale_variable_keeps_the_native_asinh_of_the_original_rule():
    x = np.round(_lognormal(5000, 3.0, seed=1), 1)
    d = diagnostics(x)
    dec, _, y = decide("cdl.upper_wick_pips", x, d, train=np.ones(len(x), bool))
    assert dec == "ASINH" and np.allclose(y, np.arcsinh(x))


def test_zeros_force_the_scaled_asinh_with_a_frozen_train_median():
    x = _lognormal(5000, 1e-3, seed=2)
    x[::7] = 0.0
    d = diagnostics(x)
    dec, why, y = decide("px.rv_1", x, d, train=np.ones(len(x), bool))
    s = np.median(x[x > 0])
    assert dec == "ASINHS" and d["transform_scale"] == pytest.approx(s) and np.allclose(y, np.arcsinh(x / s))


def test_regime_probabilities_and_bounded_indicators_are_never_transformed():
    p = np.clip(np.random.default_rng(3).beta(0.2, 5.0, 5000), 0, 1)
    d = diagnostics(p)
    assert d["skew"] > 2
    assert decide("rg.gmm.p2", p, d)[0] == "NONE" and decide("RSI", p, d)[0] == "NONE"


def test_candidate_skew_is_measured_on_train_rows_only():
    x = _lognormal(6000, 1e-3, seed=4)
    tr = np.zeros(len(x), bool)
    tr[:3000] = True
    x[3000:] = 1e-3                                                     # later rows are constant: skew outside TRAIN is 0
    d = diagnostics(x[tr])
    dec, why, y = decide("px.range_2", x, d, train=tr)
    assert dec == "LOG" and f"{skew(np.log(x[tr])):.2f}" in why


def test_recipe_is_decided_on_train_only_and_later_domain_failures_become_nan():
    """Auditor's witness (2026-10-22): changing ONLY a row after TRAIN must leave the recipe and every transformed
    TRAIN value unchanged; the offending later row becomes NaN and is counted."""
    x = _lognormal(5000, 1e-3, seed=5)
    tr = np.zeros(len(x), bool)
    tr[:4000] = True
    d0 = diagnostics(x[tr])
    dec0, why0, y0 = decide("px.rv_4", x, d0, train=tr)
    x2 = x.copy()
    x2[4500] = 0.0                                                      # TRAIN identical, a later zero
    d2 = diagnostics(x2[tr])
    dec2, why2, y2 = decide("px.rv_4", x2, d2, train=tr)
    assert dec0 == dec2 == "LOG"
    assert np.array_equal(y0[tr], y2[tr]) and np.isnan(y2[4500]) and np.isfinite(y2[4501])
    assert d2["domain_nan_rows_after_train"] == 1 and "recipe unchanged" in why2 and d0["domain_nan_rows_after_train"] == 0


def test_a_train_domain_failure_is_decided_on_train():
    x = _lognormal(5000, 1e-3, seed=6)
    x[100] = 0.0                                                        # inside TRAIN: LOG is not admissible
    tr = np.ones(len(x), bool)
    d = diagnostics(x[tr])
    dec, why, y = decide("px.rv_4", x, d, train=tr)
    assert dec == "ASINHS" and "LOG rejected: outside its domain on TRAIN" in why and np.isfinite(y).all()
