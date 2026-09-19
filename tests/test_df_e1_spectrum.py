"""RP28: spectra with declared resolution: a planted component inside the support is measured, one outside
is NO_RESUELTO (never zero), a scale change of one column does not move the aggregate, missing values are
imputed and counted, and the result depends on the train support (sensitivity measured, not hidden)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("df_e1_spectrum")
DELTA = 60.0


def _series(n, periods_seconds, seed=0):
    t = np.arange(n) * DELTA
    rng = np.random.default_rng(seed)
    return sum(np.sin(2 * np.pi * t / p) for p in periods_seconds) + 0.05 * rng.normal(size=n)


def test_RP28_component_inside_resolution_is_measured_and_outside_is_NO_RESUELTO():
    n, nperseg = 20000, 4096                                              # slowest resolvable period 4096 min = 2.84 days
    x = _series(n, [3600.0, 86400.0])                                     # 1 h and 24 h components
    s = S.welch_declared(x, DELTA, nperseg)
    assert s["slowest_resolvable_period_seconds"] == 4096 * DELTA and s["fastest_period_seconds"] == 120.0
    hour = S.band_share(s, 3600.0)
    day = S.band_share(s, 86400.0)
    assert hour["state"] == "MEDIDO" and hour["share"] > 0.3 and day["state"] == "MEDIDO" and day["share"] > 0.3
    slow = S.band_share(s, 35 * 86400.0)                                  # 35 days: beyond the support -> not a measured zero
    assert slow["state"] == "NO_RESUELTO" and "outside the spectral support" in slow["why"]
    fast = S.band_share(s, 90.0)
    assert fast["state"] == "NO_RESUELTO"
    # a slow component really present but outside the support stays NO_RESUELTO: no zero is reported
    y = _series(n, [3600.0]) + 3 * np.sin(2 * np.pi * np.arange(n) * DELTA / (60 * 86400.0))
    assert S.band_share(S.welch_declared(y, DELTA, nperseg), 60 * 86400.0)["state"] == "NO_RESUELTO"


def test_RP28_the_aggregate_is_invariant_to_one_columns_scale_and_grains_are_apart():
    n = 12000
    X = np.stack([_series(n, [3600.0], 1), _series(n, [86400.0], 2), _series(n, [3600.0, 86400.0], 3)], axis=1)
    bands = {"hour": 3600.0, "day": 86400.0, "slow_35d": 35 * 86400.0}
    a = S.per_variable_and_aggregate(X, ["c0", "c1", "c2"], DELTA, 4096, bands, "all declared feature columns")
    X2 = X.copy()
    X2[:, 0] *= 1000.0
    b = S.per_variable_and_aggregate(X2, ["c0", "c1", "c2"], DELTA, 4096, bands, "all declared feature columns")
    assert a["aggregate"]["bands"]["hour"]["share"] == pytest.approx(b["aggregate"]["bands"]["hour"]["share"], rel=1e-9)
    assert a["per_variable"]["c0"]["bands"]["hour"]["share"] > 0.5 and a["per_variable"]["c1"]["bands"]["hour"]["share"] < 0.1
    assert a["aggregate"]["bands"]["slow_35d"]["state"] == "NO_RESUELTO" and a["aggregate"]["resolution"]["nperseg"] == 4096
    assert a["column_rule"] == "all declared feature columns"


def test_RP28_missing_values_are_imputed_and_counted_and_train_support_sensitivity_is_visible():
    n = 12000
    x = _series(n, [3600.0])
    x[100:200] = np.nan
    s = S.welch_declared(x, DELTA, 4096)
    assert s["n_missing_imputed"] == 100 and s["state"] == "MEDIDO"
    short = S.welch_declared(x[-3000:], DELTA, 4096)                      # shorter train support: nperseg clipped, resolution declared
    assert short["nperseg"] == 3000 and short["slowest_resolvable_period_seconds"] == 3000 * DELTA
    assert S.band_share(short, 4 * 86400.0)["state"] == "NO_RESUELTO"      # four days are not resolvable on 50 hours of support
    assert S.welch_declared(x[:10], DELTA, 4096)["state"] == "NO_RESUELTO"
