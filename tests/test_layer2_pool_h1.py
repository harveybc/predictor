"""Tests for tools/layer2_pool_h1.py and its libraries (tools/lib/gaussian_hmm.py, kalman_llt.py, native_wavelet.py).

Run only this file: the repository's legacy suite is stale.
    python -m pytest tests/test_layer2_pool_h1.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import layer2_pool_h1 as L2  # noqa: E402
from tools.lib import gaussian_hmm, kalman_llt  # noqa: E402

N = 1600
CUT = 1100


def synthetic_grid(n=N, seed=3, start="2020-01-02", strength=True):
    rng = np.random.default_rng(seed)
    lc = np.log(1.1) + np.cumsum(rng.normal(0, 0.0008, n))
    c = np.exp(lc)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.0004, n)))
    df = pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="h", tz="UTC"), "BidOpen": o, "BidHigh": h, "BidLow": l, "BidClose": c})
    if strength:
        for col in L2.STRENGTH_INPUTS.values():
            v = rng.normal(0, 0.001, n)
            v[[300, min(1400, n - 1)]] = np.nan                # two missing source bars, away from the perturbation cut
            df[col] = v
    return df


@pytest.fixture(scope="module")
def built():
    raw = synthetic_grid()
    P = L2.parents(raw)
    fit = np.ones(len(P), bool)
    cfg_hmm = dict(L2.CFG["hmm"], max_iter=25)
    params = {"hmm": L2.fit_hmm(P, fit, cfg_hmm), "gmm": L2.fit_gmm(P, fit, dict(L2.CFG["gmm"], n_init=1)),
              "kalman": L2.fit_kalman(P, fit, dict(L2.CFG["kalman"], maxiter=120)), "fracdiff": L2.fit_fracdiff(P, fit)}
    X, meta, params = L2.build(P, L2.ALL, fit, params)
    return raw, P, X, meta, params


# ----------------------------------------------------------------------------- causality: the hard requirement
def test_future_perturbation_every_family(built):
    raw, _, _, _, params = built
    rep = L2.perturbation_check(raw, L2.ALL, params, n_prefix=N, cut=CUT, n_append=500)
    assert set(rep) == set(L2.ALL)
    for fam, r in rep.items():
        assert r["bit_identical_at_or_before_cut"], (fam, r)
        assert r["positive_control_columns_changed_at_cut"] > 0, (fam, r)
        assert r["cells_changed_after_cut"] > 0, fam           # the perturbation really reached the rows after the cut


def test_appending_rows_does_not_change_the_past(built):
    raw, P, X, _, params = built
    app = L2.synthetic_rows(raw.iloc[-1], 500, np.random.default_rng(9), raw["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    X2, _, _ = L2.build(L2.parents(pd.concat([raw, app], ignore_index=True)), L2.ALL, np.ones(len(raw) + 500, bool), params)
    a, b = X.to_numpy(), X2.to_numpy()[:len(raw)]
    assert np.array_equal(a, b, equal_nan=True)


def test_incomplete_windows_are_nan_not_filled(built):
    _, _, X, _, params = built
    assert X["wv.lc.A4"].iloc[:255].isna().all() and np.isfinite(X["wv.lc.A4"].iloc[255])
    assert X["hb.r1.amplitude"].iloc[:128].isna().all() and np.isfinite(X["hb.r1.amplitude"].iloc[128])  # logret_1[0] is NaN
    assert X["mt.r1.total_power"].iloc[:256].isna().all() and np.isfinite(X["mt.r1.total_power"].iloc[256])
    d = params["fracdiff"]["d"]
    assert X[f"fd.lc.d{d:.1f}"].iloc[:255].isna().all()


def test_nan_in_the_middle_only_affects_windows_that_contain_it(built):
    raw, _, X, _, params = built
    bad = raw.copy()
    bad.loc[600, ["BidOpen", "BidHigh", "BidLow", "BidClose"]] = np.nan
    X2, _, _ = L2.build(L2.parents(bad), ("wavelet", "multitaper"), np.ones(len(bad), bool), params)
    # the window of 256 ending in [600, 855] contains the NaN -> NaN; outside, identical
    w = X2["wv.lc.A4"]
    assert w.iloc[600:856].isna().all()
    assert np.array_equal(X["wv.lc.A4"].to_numpy()[:600], w.to_numpy()[:600], equal_nan=True)
    assert np.array_equal(X["wv.lc.A4"].to_numpy()[856:], w.to_numpy()[856:], equal_nan=True)


# ----------------------------------------------------------------------------- wavelet (vendored native producer)
def test_wavelet_right_edge_equals_direct_pywt(built):
    import pywt
    _, P, X, meta, _ = built
    x = P["log_close"].to_numpy()
    t = 900
    coeffs = pywt.wavedec(np.array(x[t - 255:t + 1], copy=True), "db4", level=4, mode="symmetric")
    assert X["wv.lc.A4"].iloc[t] == coeffs[0][-1]
    assert X["wv.lc.D1"].iloc[t] == coeffs[4][-1]
    assert X["wv.lc.energy_D4"].iloc[t] == pytest.approx(float(np.mean(coeffs[1] ** 2)))
    assert meta["wavelet"]["columns"]["wv.lc.A4"]["algorithm"] == "DWT_DB4_TRAILING_WINDOW"
    assert meta["wavelet"]["columns"]["wv.lc.A4"]["proxy_of"] is None


def test_vendored_wavelet_header_names_origin():
    txt = (ROOT / "tools/lib/native_wavelet.py").read_text()
    assert "307ded100" in txt and "financial-data" in txt and "pywt.wavedec" in txt
    assert "rolling" not in txt.split('"""', 2)[2]   # the body never falls back to a moving-average proxy


# ----------------------------------------------------------------------------- hilbert
def test_hilbert_recovers_sinusoid_at_right_edge():
    n, W = 600, 128
    tt = np.arange(n)
    period = 16.0
    x = 0.002 * np.sin(2 * np.pi * tt / period)
    amp, cs, sn, fq = L2._hilbert_windows(x, W, detrend=False)
    # the FFT analytic signal has an edge effect; with a window that is an integer number of periods it is small
    assert np.nanmedian(amp[W:]) == pytest.approx(0.002, rel=0.15)
    assert np.nanmedian(fq[W:]) == pytest.approx(1 / period, rel=0.15)
    assert np.all(np.isnan(amp[:W - 1])) and np.isfinite(amp[W - 1])
    assert np.allclose(cs[W:] ** 2 + sn[W:] ** 2, 1.0)


def test_hilbert_detrended_variant_removes_window_line():
    n, W = 500, 128
    x = 0.01 * np.arange(n) + 1.0                      # a pure line: after OLS detrending the window is ~0
    amp, *_ = L2._hilbert_windows(x, W, detrend=True)
    assert np.nanmax(amp[W:]) < 1e-9


# ----------------------------------------------------------------------------- multitaper
def test_multitaper_dominant_period_and_bands():
    n = 800
    rng = np.random.default_rng(0)
    tt = np.arange(n)
    r = 0.003 * np.sin(2 * np.pi * tt / 16.0) + rng.normal(0, 0.0002, n)
    P = pd.DataFrame({"logret_1": r}, index=pd.date_range("2020-01-02", periods=n, freq="h", tz="UTC"))
    F, meta = L2.fam_multitaper(P)
    dom = F["mt.r1.dominant_period_bars"].dropna()
    assert np.median(dom) == pytest.approx(16.0, rel=0.1)
    bands = F[[c for c in F.columns if c.startswith("mt.r1.power_")]].dropna()
    assert (bands["mt.r1.power_12_48h"] > bands["mt.r1.power_gt48h"]).mean() > 0.95   # 16 h lies in the 12-48 h band
    assert ((F["mt.r1.spectral_entropy"].dropna() >= 0) & (F["mt.r1.spectral_entropy"].dropna() <= 1)).all()
    assert meta["frequency_unit"].startswith("cycles per bar")
    tot = F["mt.r1.total_power"].dropna()
    assert (tot - bands.sum(axis=1)).abs().max() < 1e-6 * tot.max()


# ----------------------------------------------------------------------------- kalman
def test_kalman_filter_matches_matrix_form():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0, 1e-3, 400)) + 1.0
    q1, q2, r = 1e-6, 1e-9, 1e-7
    lvl, slp, inn, inv = kalman_llt.filter_llt(y, q1, q2, r)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q1, q2])
    H = np.array([1.0, 0.0])
    x = np.array([y[0], 0.0])
    Pm = np.eye(2)
    for t in range(1, len(y)):
        x = F @ x
        Pm = F @ Pm @ F.T + Q
        S = H @ Pm @ H + r
        K = Pm @ H / S
        e = y[t] - H @ x
        x = x + K * e
        Pm = Pm - np.outer(K, H) @ Pm
        assert lvl[t] == pytest.approx(x[0], abs=1e-12) and slp[t] == pytest.approx(x[1], abs=1e-12)
        assert inn[t] == pytest.approx(e, abs=1e-12) and inv[t] == pytest.approx(S, rel=1e-9)


def test_kalman_mle_recovers_noise_scales():
    rng = np.random.default_rng(5)
    n = 4000
    slope = np.cumsum(rng.normal(0, 2e-5, n))
    level = np.cumsum(slope + rng.normal(0, 5e-4, n))
    y = level + rng.normal(0, 1e-3, n)
    prm = kalman_llt.fit_llt(y, burn_in=100, maxiter=400)
    assert prm["r"] == pytest.approx(1e-6, rel=0.5)
    assert prm["q_level"] == pytest.approx(2.5e-7, rel=0.6)
    assert prm["q_slope"] == pytest.approx(4e-10, rel=0.9)


def test_kalman_standardised_innovation_is_unit_scale(built):
    _, _, X, _, _ = built
    z = X["kf.lc.innovation_std"].iloc[200:].dropna()
    assert 0.7 < z.std() < 1.4


# ----------------------------------------------------------------------------- HMM (numpy)
def make_regimes(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.02, 0.97]])
    sig = np.array([0.3, 1.0, 2.5])
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.choice(3, p=A[s[t - 1]])
    X = np.stack([rng.normal(0, sig[s]), sig[s] + rng.normal(0, 0.2, n), rng.normal(0, 1, n)], axis=1)
    return X, s


def test_hmm_em_monotone_and_recovers_states():
    X, s = make_regimes()
    prm = gaussian_hmm.fit(X, 3, max_iter=60, order_dim=1)
    h = np.array(prm["loglik_history"])
    assert np.all(np.diff(h) > -1e-6)                           # EM never decreases the likelihood
    assert np.all(np.diff(np.array(prm["means"])[:, 1]) > 0)    # ordered by increasing volatility
    filt, ent, sur = gaussian_hmm.forward_filter(X, prm)
    acc = np.mean(filt.argmax(axis=1) == s)
    assert acc > 0.85
    assert np.allclose(filt.sum(axis=1), 1.0)
    assert np.all(ent >= -1e-12) and np.all(ent <= np.log(3) + 1e-9)
    assert np.isfinite(sur).all()


def test_hmm_forward_filter_is_bayes_recursion_and_causal():
    X, _ = make_regimes(600)
    prm = gaussian_hmm.fit(X, 3, max_iter=10, order_dim=1)
    filt, ent, sur = gaussian_hmm.forward_filter(X, prm)
    means, var = np.array(prm["means"]), np.array(prm["variances"])
    A, pi = np.array(prm["transmat"]), np.array(prm["startprob"])
    belief = pi
    for t in range(60):
        b = np.exp(-0.5 * np.sum(np.log(2 * np.pi * var) + (X[t] - means) ** 2 / var, axis=1))
        joint = belief * b
        assert sur[t] == pytest.approx(-np.log(joint.sum()), rel=1e-9)
        post = joint / joint.sum()
        assert np.allclose(filt[t], post)
        belief = post @ A
    # causal: altering the future never changes the filtered past; a NaN row emits NaN and propagates the belief
    Y = X.copy()
    Y[300:] = np.random.default_rng(0).normal(size=Y[300:].shape)
    f2, _, _ = gaussian_hmm.forward_filter(Y, prm)
    assert np.array_equal(filt[:300], f2[:300])
    Z = X.copy()
    Z[100] = np.nan
    f3, e3, s3 = gaussian_hmm.forward_filter(Z, prm)
    assert np.isnan(f3[100]).all() and np.isnan(e3[100]) and np.isnan(s3[100])
    assert np.array_equal(filt[:100], f3[:100]) and np.isfinite(f3[101]).all()


def test_hmm_outputs_in_pool(built):
    _, _, X, meta, params = built
    p = X[["rg.hmm.p0", "rg.hmm.p1", "rg.hmm.p2"]].dropna()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(np.diff(params["hmm"]["state_rv24_mean_raw"]) > 0)
    assert np.all(np.diff(params["gmm"]["state_rv24_mean_raw"]) > 0)
    assert "forward filter" in meta["hmm"]["note"] and "never forward-backward" in meta["hmm"]["note"]


# ----------------------------------------------------------------------------- fracdiff
def test_fracdiff_weights_and_limits():
    w = L2.fracdiff_weights(1.0, 10)
    assert np.allclose(w[:2], [1.0, -1.0]) and np.allclose(w[2:], 0.0)
    w0 = L2.fracdiff_weights(0.0, 10)
    assert np.allclose(w0, np.r_[1.0, np.zeros(9)])
    x = np.cumsum(np.ones(400))
    y = L2.fracdiff_fixed(x, 1.0, 256)
    assert np.isnan(y[:255]).all() and np.allclose(y[255:], 1.0)


def test_fracdiff_choice_rule_and_table(built):
    _, _, _, _, params = built
    prm = params["fracdiff"]
    tab = pd.DataFrame(prm["adf_table"])
    assert list(tab["d"]) == L2.CFG["fracdiff"]["d_grid"]
    passing = tab[tab["adf_p"] < 0.05]
    assert prm["d"] == (passing["d"].iloc[0] if len(passing) else None)
    assert "adf_stat" in tab and (tab["weights_abs_sum_dropped"].diff().dropna() <= 0).all()


# ----------------------------------------------------------------------------- store contract
def test_build_column_metadata_covers_every_column(built):
    _, _, X, meta, _ = built
    cols = {c for f in meta.values() for c in f["columns"]}
    assert cols == set(X.columns)
    for f, m in meta.items():
        for c, cm in m["columns"].items():
            assert cm["proxy_of"] is None and cm["algorithm"] and cm["parent"] and cm["unit"] and cm["warmup_bars"] >= 0
            assert L2._fam_of(c) == f
    counts = {f: len(m["columns"]) for f, m in meta.items()}
    assert counts == {"wavelet": 18, "hilbert": 8, "multitaper": 7, "kalman": 4, "gmm": 4, "hmm": 5, "fracdiff": 1, "strength": 26}


def test_strength_family_uses_r4_returns_as_is_and_skips_without_them(built):
    raw, P, X, meta, params = built
    assert meta["strength"]["status"].startswith("GENERATED")
    # a NaN source bar voids exactly the 256 wavelet windows (and 128 hilbert windows) containing it, nothing else is filled
    bad = np.flatnonzero(~np.isfinite(raw["fx.usd_strength.ret_1"].to_numpy()))
    for i in bad:
        assert X["wv.usd.A4"].iloc[i:min(i + 256, len(X))].isna().all()
        assert X["hb.usd.amplitude"].iloc[i:min(i + 128, len(X))].isna().all()
    assert X["wv.usd.A4"].notna().sum() == len(X) - 255 - 256 - (len(X) - 1400)   # warm-up + the two voided spans only
    import pywt
    t = 1500
    x = raw["fx.usd_strength.ret_1"].to_numpy()
    if np.isfinite(x[t - 255:t + 1]).all():
        assert X["wv.usd.D1"].iloc[t] == pywt.wavedec(np.array(x[t - 255:t + 1], copy=True), "db4", level=4)[4][-1]
    P2 = L2.parents(synthetic_grid(strength=False))
    F, m = L2.fam_strength(P2)
    assert F.shape[1] == 0 and m["status"].startswith("SKIPPED") and m["columns"] == {}
    rep = L2.perturbation_check(synthetic_grid(strength=False), ("strength",), params, n_prefix=N, cut=CUT, n_append=100)
    assert rep["strength"]["columns"] == 0 and rep["strength"]["passed"] is None


def test_main_refuses_test_rows_and_writes_manifest(tmp_path):
    import hashlib
    import json
    raw = synthetic_grid(n=1500, start="2024-11-01")         # crosses 2025-01-01: the TEST rows must be dropped
    src = tmp_path / "admitted.parquet"
    raw.to_parquet(src, index=False)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    out = tmp_path / "out"
    L2.main(["--admitted", str(src), "--admitted-sha256", digest, "--out", str(out), "--families", "wavelet,fracdiff",
             "--fit-end", "2025-01-01", "--diag-end", "2025-01-01", "--skip-perturbation"])
    man = json.loads((out / "MANIFEST.json").read_text())
    store = pd.read_parquet(out / "layer2_pool_eurusd_h1.parquet")
    assert man["test_read"] is False and man["rows"] == len(store)
    assert pd.Timestamp(store["timestamp"].max()) < pd.Timestamp("2025-01-01", tz="UTC")
    assert man["rows_dropped_test"] == int((raw["timestamp"] >= pd.Timestamp("2025-01-01", tz="UTC")).sum()) > 0
    assert man["family_counts"]["wavelet"] == 18
    assert man["cross_asset_input"] is None
    # with a cross-asset store: strength columns appear, TEST rows in it are refused
    ca = raw[["timestamp", *L2.STRENGTH_INPUTS.values()]]
    ca_ok = ca[ca["timestamp"] < pd.Timestamp("2025-01-01", tz="UTC")]
    ca_ok.to_parquet(tmp_path / "ca.parquet", index=False)
    L2.main(["--admitted", str(src), "--admitted-sha256", digest, "--out", str(out / "s"), "--families", "strength", "--fit-end", "2025-01-01",
             "--diag-end", "2025-01-01", "--skip-perturbation", "--cross-asset", str(tmp_path / "ca.parquet")])
    man2 = json.loads((out / "s" / "MANIFEST.json").read_text())
    assert man2["family_counts"]["strength"] == 26 and man2["cross_asset_input"]["columns"] == list(L2.STRENGTH_INPUTS.values())
    ca.to_parquet(tmp_path / "ca_test.parquet", index=False)
    with pytest.raises(SystemExit, match="TEST_ROWS_FORBIDDEN"):
        L2.main(["--admitted", str(src), "--admitted-sha256", digest, "--out", str(out / "s2"), "--families", "strength", "--skip-perturbation",
                 "--cross-asset", str(tmp_path / "ca_test.parquet")])
    with pytest.raises(SystemExit):
        L2.main(["--admitted", str(src), "--admitted-sha256", "0" * 64, "--out", str(out)])
