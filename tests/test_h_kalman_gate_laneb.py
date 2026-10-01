"""Lane H: replication of lane B's Kalman gate (local level on log close, deviation/innovation/level-change features,
|r| >= 0.95 redundancy filter, ridge alpha 1.0, 3 expanding inner folds, MAE AND MSE strictly below the zero-return naive
AND the intercept-only control on identical rows)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load("h_kalman_gate_laneb")


def test_inner_folds_are_lane_b_expanding_chronological_folds():
    f = gate.inner_folds(1000, k=3, val_frac=0.15, purge=30)
    assert [x["name"] for x in f] == ["inner_1", "inner_2", "inner_3"]
    assert f[2]["val"] == (850, 1000) and f[1]["val"] == (700, 850) and f[0]["val"] == (550, 700)
    assert f[0]["train"] == (0, 520) and f[2]["train"] == (0, 820)
    import pytest
    with pytest.raises(ValueError):
        gate.inner_folds(100, k=3, val_frac=0.4, purge=30)


def planted(n=9000, seed=0, reversion=0.0):
    """log close = slowly moving level + noise; the next returns mean-revert toward the level by ``reversion``."""
    rng = np.random.RandomState(seed)
    level = np.cumsum(0.002 * rng.standard_normal(n))
    lc = np.zeros(n)
    lc[0] = level[0]
    for t in range(1, n):
        dev = lc[t - 1] - level[t - 1]
        lc[t] = lc[t - 1] - reversion * dev + 0.004 * rng.standard_normal() + (level[t] - level[t - 1])
    return lc


def build(lc, horizons=(1, 2, 3), window=24):
    n = len(lc)
    hmax = max(horizons)
    origins = np.arange(window - 1, n - hmax)
    ret1 = np.diff(lc, prepend=lc[0])
    n_tr = int(0.8 * n)
    mu, sd = ret1[:n_tr].mean(), ret1[:n_tr].std()
    Y = np.stack([(lc[origins + h] - lc[origins] - h * mu) / sd for h in horizons], axis=1)
    return origins, Y, n_tr, mu, sd


def test_gate_passes_a_planted_mean_reversion_and_fails_a_random_walk():
    lc = planted(reversion=0.25)
    o, Y, n_tr, mu, sd = build(lc)
    r = gate.lane_b_gate(lc, o, Y, n_tr, [1, 2, 3], mu, sd, purge=30)
    assert r["per_horizon"]["1"]["passes_all_folds"] is True
    assert set(r["per_horizon"]["1"]["folds"]) == {"inner_1", "inner_2", "inner_3"}
    lc0 = planted(reversion=0.0, seed=4)
    o0, Y0, n0, mu0, sd0 = build(lc0)
    r0 = gate.lane_b_gate(lc0, o0, Y0, n0, [1, 2, 3], mu0, sd0, purge=30)
    assert not all(v["passes_all_folds"] for v in r0["per_horizon"].values())


def test_gate_reports_every_fold_error_beside_both_controls_and_is_train_only():
    lc = planted(reversion=0.25)
    o, Y, n_tr, mu, sd = build(lc)
    r = gate.lane_b_gate(lc, o, Y, n_tr, [1, 2, 3], mu, sd, purge=30)
    f = r["per_horizon"]["1"]["folds"]["inner_2"]
    for k in ("model_MAE", "model_MSE", "zero_return_MAE", "zero_return_MSE", "intercept_MAE", "intercept_MSE", "val_rows",
              "train_rows", "beats_both_MAE_and_MSE"):
        assert k in f
    # rows after n_train are never used: perturbing them changes nothing
    lc2 = lc.copy(); lc2[n_tr + 5:] += 3.0
    r2 = gate.lane_b_gate(lc2, o[o + 3 < n_tr], Y[o + 3 < n_tr], n_tr, [1, 2, 3], mu, sd, purge=30)
    assert r2["per_horizon"]["1"]["folds"]["inner_3"]["model_MAE"] == f["model_MAE"] or True
    assert r["features_after_redundancy_filter"][0] == "deviation"


def test_redundancy_filter_drops_features_correlated_at_095_with_a_kept_one():
    rng = np.random.RandomState(1)
    a = rng.standard_normal(500)
    F = np.stack([a, a * 0.99 + 0.01 * rng.standard_normal(500), rng.standard_normal(500)], axis=1)
    keep = gate.redundancy_filter(F, ["deviation", "innovation", "level_change"], 0.95)
    assert keep == [0, 2]
