"""Lane H tests for tools/h_kalman_arms.py (gather, ridge, naives, bootstrap, diagnostics, channel builders)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arms = _load("h_kalman_arms")
kf = arms.kf


def test_gather_orders_lags_and_never_reads_future_rows():
    C = np.arange(40, dtype=float).reshape(20, 2)
    o = np.array([5, 9])
    X = arms.gather(C, o, 3)
    assert X.shape == (2, 6)
    assert list(X[0]) == [10, 11, 8, 9, 6, 7]
    C2 = C.copy(); C2[6:] = -999
    assert np.array_equal(arms.gather(C2, np.array([5]), 3), X[:1])       # rows after the origin never enter


def test_ridge_recovers_linear_signal_and_picks_alpha_on_inner_holdout_only():
    rng = np.random.RandomState(0)
    N, P = 1500, 12
    X = rng.standard_normal((N, P))
    w = rng.standard_normal((P, 3))
    Y = X @ w + 0.1 * rng.standard_normal((N, 3))
    Xev = rng.standard_normal((200, P))
    (pred,), alpha, inner = arms.ridge_fit_predict(X, Y, [Xev])
    assert np.mean(np.abs(pred - Xev @ w)) < 0.05
    assert alpha in arms.ALPHAS and set(inner) == set(arms.ALPHAS)
    # evaluation rows cannot change the chosen alpha
    (pred2,), alpha2, _ = arms.ridge_fit_predict(X, Y, [Xev * 0 + 100.0])
    assert alpha2 == alpha
    # pure noise target: heavy regularisation is chosen
    Yn = rng.standard_normal((N, 2))
    _, alpha_n, _ = arms.ridge_fit_predict(X, Yn, [Xev])
    assert alpha_n >= 100.0


def test_ridge_is_bitwise_deterministic_in_process():
    rng = np.random.RandomState(1)
    X, Y = rng.standard_normal((600, 8)), rng.standard_normal((600, 2))
    a = arms.ridge_fit_predict(X, Y, [X[:50]])[0][0]
    b = arms.ridge_fit_predict(X, Y, [X[:50]])[0][0]
    assert np.array_equal(a.view("<i8"), b.view("<i8"))


def test_naives_match_the_lane_f2_definitions():
    rng = np.random.RandomState(2)
    zt = rng.standard_normal(200)
    origins = np.arange(30, 120)
    horizons = [1, 2, 3, 4, 5, 6]
    mu, sigma = 0.0002, 0.02
    nv = arms.naive_predictions(zt, origins, horizons, mu, sigma)
    assert np.array_equal(nv["train_mean"], np.zeros((90, 6)))
    for k, h in enumerate(horizons):
        assert np.allclose(nv["zero_return"][:, k], -h * mu / sigma)
        assert np.array_equal(nv["persistence_last_value"][:, k], zt[origins])
        want = sum(zt[origins - 6 + j] for j in range(1, h + 1))
        assert np.allclose(nv["seasonal_6"][:, k], want)
    # the seasonal naive for horizon h reads rows o-5..o-6+h: never a row after the origin
    assert all(bool(np.all(origins - 6 + h <= origins)) for h in horizons)


def test_block_bootstrap_ci_covers_the_mean_and_widens_with_block_length():
    rng = np.random.RandomState(3)
    e = rng.standard_normal(2000)
    d = np.convolve(e, np.ones(20) / 20, mode="same") + 0.1            # autocorrelated paired differences
    m, lo, hi = arms.block_bootstrap_ci(d, 5, B=500, seed=1)
    m2, lo2, hi2 = arms.block_bootstrap_ci(d, 80, B=500, seed=1)
    assert lo < m < hi and (hi2 - lo2) > (hi - lo)
    assert arms.block_bootstrap_ci(d, 80, B=500, seed=1) == (m2, lo2, hi2)
    q = arms.quarter_deltas(d)
    assert len(q) == 4 and abs(np.mean(q) - d.mean()) < 1e-12


def test_phase_lag_detects_a_known_delay_and_no_delay():
    rng = np.random.RandomState(4)
    x = np.cumsum(rng.standard_normal(3000))
    assert arms.phase_lag(x, x)["best_lag_bars"] == 0
    delayed = np.concatenate([np.full(3, x[0]), x[:-3]])
    assert arms.phase_lag(x, delayed)["best_lag_bars"] == 3        # obs = x, level = x delayed by 3 bars
    # the filtered level of a noisy random walk lags the observation by about one bar or less
    lvl = x + 0.0
    assert arms.phase_lag(x, lvl)["corr_at_best"] > 0.99


def test_innovation_stability_reports_both_partitions():
    rng = np.random.RandomState(5)
    z = rng.standard_normal(6000)
    s = arms.innovation_stability(z[:4000], z[4000:])
    assert abs(s["train"]["std"] - 1) < 0.05 and abs(s["validation"]["acf1"]) < 0.1
    assert s["train"]["ljung_box_q10"] < 30 and s["validation"]["rolling_std_min"] is not None


def test_extremes_retained_distinguishes_a_flattening_state_from_an_identity():
    rng = np.random.RandomState(6)
    obs = rng.standard_normal(5000)
    obs[::100] += 8.0
    lvl_flat = np.convolve(obs, np.ones(20) / 20, mode="same")
    zin = obs - lvl_flat
    a = arms.extremes_retained(obs, obs, zin * 0, 0.0, 1.0)
    b = arms.extremes_retained(obs, lvl_flat, zin, 0.0, 1.0)
    assert a["events"] > 20 and a["median_level_retention"] == 1.0
    assert b["median_level_retention"] < 0.5 and b["share_abs_zinnov_over_threshold"] > 0.5


def test_kalman_channels_names_logvar_and_refusal_of_the_smoother():
    rng = np.random.RandomState(7)
    X = np.cumsum(rng.standard_normal((400, 3)), axis=0) + rng.standard_normal((400, 3))
    a = kf.fit(kf.default_spec(kf.LOCAL_LEVEL), X[:300], {"dataset_id": "d", "role": "TRAIN", "row_range": [0, 300],
                                                           "column_ids": ["a", "b", "c"]})
    out = kf.transform_batch(a, X)
    M, names = arms.kalman_channels(out, names=["b"])
    assert names == ["b__kf_level", "b__kf_innov", "b__kf_zinnov", "b__kf_logvar"] and M.shape == (400, 4)
    assert np.isfinite(M).all() and np.allclose(M[:, 3], np.log(out.arrays["state_var"][:, 1]))
    with pytest.raises(kf.OperatorRefusal):
        arms.kalman_channels(kf.smoother_control(a, X))


def test_multi_ridge_fixed_alphas_agree_with_the_chosen_one_and_shrink_monotonically():
    rng = np.random.RandomState(8)
    X, Y = rng.standard_normal((800, 10)), rng.standard_normal((800, 2))
    Xev = rng.standard_normal((50, 10))
    r = arms.ridge_fit_predict_multi(X, Y, [Xev], fixed=(1.0, 1e3, 1e8))
    # a huge alpha collapses to the train mean
    assert np.allclose(r["preds_fixed"][1e8][0], Y.mean(axis=0), atol=1e-3)
    chosen_alpha = r["chosen"]
    (p_old,), a_old, _ = arms.ridge_fit_predict(X, Y, [Xev])
    assert a_old == chosen_alpha and np.allclose(p_old, r["preds_chosen"][0], atol=1e-10)
    assert arms.ALPHAS[-1] >= 1e8
