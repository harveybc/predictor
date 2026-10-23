"""Fit-scope tests (Musashi's final joint audit A1, 2026-10-23): perturbing HELD-OUT inputs and labels and REBUILDING
(fitting included) must leave every fitted parameter and every training-side value unchanged; the contaminated
global fit must not (positive control). Synthetic data only; no store, no network."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from tools import activity_producer as AP  # noqa: E402
from tools import fold_local_producers as F  # noqa: E402
from tools import layer2_pool_h1 as L2  # noqa: E402


def _bars(n=2600, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    lc = np.cumsum(rng.normal(0, 0.0012, n)) + np.log(1.1)
    c = np.exp(lc)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.0006, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.0006, n)))
    return pd.DataFrame({"timestamp": ts, "BidOpen": o, "BidHigh": h, "BidLow": l, "BidClose": c})


def _scope(ts, lo, hi):
    return {"fit_window": [str(ts[0]), str(ts[-1] + pd.Timedelta(hours=2))], "exclude": [[str(ts[lo]), str(ts[hi])]]}


def test_allowed_mask_uses_availability_and_inclusive_exclusions():
    ts = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    m = F.allowed_mask(ts, {"fit_window": ["2020-01-01 00:00", "2020-01-01 05:00"], "exclude": [["2020-01-01 01:00", "2020-01-01 02:00"]]})
    # bar starts 0..3 are available strictly before 05:00 (start + 1 h < 05:00); 1 and 2 are excluded
    assert m.tolist() == [True, False, False, True, False, False, False, False, False, False]


def _jsonable_params(p):
    return json.dumps(F.jsonable(p), sort_keys=True)


def test_learned_families_rebuilt_with_fit_ignore_the_excluded_span_and_keep_earlier_values():
    raw = _bars()
    P = F.exact_parents(raw)
    ts = P.index
    lo, hi = 1500, 1900
    sc = _scope(ts, lo, hi)
    allowed = F.allowed_mask(ts, sc)
    raw2 = raw.copy()
    rng = np.random.default_rng(5)
    for k in ("BidOpen", "BidHigh", "BidLow", "BidClose"):
        v = raw2[k].to_numpy(float).copy()
        v[lo:hi + 1] *= np.exp(rng.normal(0, 0.004, hi - lo + 1))
        raw2[k] = v
    P2 = F.exact_parents(raw2)
    X0, _, p0, _ = F.build_scoped(P, allowed)
    X1, _, p1, _ = F.build_scoped(P2, allowed)
    assert _jsonable_params(p0) == _jsonable_params(p1)                       # parameters: fitted on allowed bars only
    for c in X0.columns:
        assert np.array_equal(X0[c].to_numpy()[:lo], X1[c].to_numpy()[:lo], equal_nan=True), c
    assert any(not np.allclose(X0[c].to_numpy()[lo:hi + 1], X1[c].to_numpy()[lo:hi + 1], equal_nan=True) for c in X0.columns)
    # positive control: the contaminated fit (span included) moves values BEFORE the span
    full = np.ones(len(ts), bool)
    G0, _, g0 = L2.build(P, ("gmm",), full)
    G1, _, g1 = L2.build(P2, ("gmm",), full)
    assert _jsonable_params(g0) != _jsonable_params(g1)
    assert any(not np.array_equal(G0[c].to_numpy()[:lo], G1[c].to_numpy()[:lo], equal_nan=True) for c in G0.columns)


def test_redecide_ignores_held_out_rows_and_reports_only_real_changes():
    rng = np.random.default_rng(2)
    n = 3000
    x = 1e-3 * np.exp(rng.normal(0, 0.8, n))                                    # skewed, positive: LOG on any TRAIN
    diag = np.zeros(n, bool)
    diag[:2000] = True
    diag[1200:1500] = False                                                     # a held-out span inside the window
    out0, rec0 = F.redecide({"px.range_9": x}, {"px.range_9": "LOG"}, diag)
    x2 = x.copy()
    x2[1200:1500] = 50.0                                                        # held-out rows changed wildly
    out1, rec1 = F.redecide({"px.range_9": x2}, {"px.range_9": "LOG"}, diag)
    assert rec0["px.range_9"]["scope"] == rec1["px.range_9"]["scope"] == "LOG" and not rec0["px.range_9"]["flip"]
    assert out0 == {} and out1 == {}                                           # same decision, no constant: values unchanged
    z = x.copy()
    z[::9] = 0.0                                                                # zeros -> ASINHS with a frozen median
    o0, r0 = F.redecide({"px.rv_9": z}, {"px.rv_9": "ASINHS"}, diag)
    z2 = z.copy()
    z2[1200:1500] *= 30.0
    o1, r1 = F.redecide({"px.rv_9": z2}, {"px.rv_9": "ASINHS"}, diag)
    assert r0["px.rv_9"]["asinhs_scale"] == r1["px.rv_9"]["asinhs_scale"]
    assert np.array_equal(o0["px.rv_9@asinhs"][diag], o1["px.rv_9@asinhs"][diag])


def _activity_data(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(0.8 * X["f1"] - 0.5 * X["f2"])))
    y = (rng.random(n) < p).astype(float)
    res = np.minimum(np.arange(n) + rng.integers(1, 40, n), n - 1)
    return X, y, res


def test_usable_rows_never_resolve_inside_an_excluded_span():
    X, y, res = _activity_data()
    allowed = np.ones(len(y), bool)
    allowed[1000:1300] = False
    u = AP.usable(allowed, y, res)
    idx = np.flatnonzero(u)
    assert not u[1000:1300].any()
    assert all(not ((i < 1000) and (res[i] >= 1000)) for i in idx)            # no label resolves across the span


def test_activity_cross_fit_is_unchanged_by_held_out_labels_and_inputs():
    X, y, res = _activity_data()
    allowed = np.ones(len(y), bool)
    allowed[1000:1300] = False
    p0, r0 = AP.scope_predictions(X, y, res, allowed, k=3, lookback=8)
    y2, X2 = y.copy(), X.copy()
    y2[1000:1300] = 1.0 - y2[1000:1300]
    X2.loc[1000:1299, "f1"] = 9.0
    p1, r1 = AP.scope_predictions(X2, y2, res, allowed, k=3, lookback=8)
    assert np.array_equal(p0[allowed], p1[allowed])                             # every allowed row (OOF) unchanged
    assert not np.allclose(p0[1000:1300], p1[1000:1300])                        # the held-out rows move (their inputs moved)
    # positive control: the retired producer (one in-sample fit on every labelled row) changes earlier rows
    allr = np.flatnonzero(np.isfinite(y))
    q0 = AP._fit(X.to_numpy(float), y, allr).predict_proba(X.to_numpy(float)[:1000])[:, 1]
    q1 = AP._fit(X2.to_numpy(float), y2, allr).predict_proba(X2.to_numpy(float)[:1000])[:, 1]
    assert not np.array_equal(q0, q1)


def test_window_complete_fit_rows():
    allowed = np.ones(20, bool)
    allowed[8:10] = False
    m = F.shrink(allowed, 3)
    assert m.tolist() == [False, False, True, True, True, True, True, True, False, False, False, False, True] + [True] * 7


def test_exact_parents_rv24_has_no_memory_beyond_its_window():
    raw = _bars(600)
    raw2 = raw.copy()
    raw2.loc[100, "BidClose"] *= 1.01
    a, b = F.exact_parents(raw)["rv_24"].to_numpy(), F.exact_parents(raw2)["rv_24"].to_numpy()
    assert np.array_equal(a[:100], b[:100], equal_nan=True) and np.array_equal(a[126:], b[126:], equal_nan=True)
    assert not np.allclose(a[101:125], b[101:125])
    ref = L2.parents(raw)["rv_24"].to_numpy()
    assert np.allclose(a, ref, rtol=1e-12, atol=0, equal_nan=True)       # same quantity as the store's definition
