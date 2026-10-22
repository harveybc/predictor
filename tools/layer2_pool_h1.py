#!/usr/bin/env python3
"""Layer-2 feature pool for EURUSD H1 on the admitted grid: decomposition / spectral / state-space / regime families
as GENUINE causal producers (every value at t is a function of bars <= t; incomplete window -> NaN; nothing fitted
outside the fit period; parameters frozen and written to the manifest).

Parents (from the admitted Bid OHLC, bar start t, decision at t + 1 h): log_close = log BidClose,
logret_1 = diff(log_close), log_range = log(BidHigh / BidLow), hl_pips = (BidHigh - BidLow) / 1e-4,
rv_24 = sqrt(sum of the last 24 squared logret_1) (the px.rv_24 definition, recomputed here).

Families
  wavelet   wv.*   PyWavelets db4 wavedec on the trailing window of 256 bars, level 4, right-edge coefficient of
                   A4 / D4..D1 and the band energies, on log_close and on logret_1 (vendored native producer from
                   financial-data @307ded100, tools/lib/native_wavelet.py; NOT the stage-2.3 moving-average proxy)
  hilbert   hb.*   scipy.signal.hilbert on the trailing window of 128 bars of logret_1 and of log_close detrended
                   by the window's own OLS line; amplitude, cos/sin phase and instantaneous frequency (cycles/bar)
                   at the window's LAST sample. Edge effect: the FFT analytic signal assumes the window periodic,
                   so the last sample carries a wrap-around discontinuity (x[t] vs x[t-W+1]); detrending/demeaning
                   shrinks it, it is never removed, and it is a function of the window only (causal by construction)
  multitaper mt.*  DPSS (NW = 3.5, K = 5) multitaper PSD of the demeaned trailing 256-bar window of logret_1; band
                   powers (periods > 48 h, 12-48 h, 4-12 h, < 4 h; frequency in cycles/bar), spectral entropy
                   (DC excluded, normalised to [0, 1]) and the dominant period in bars
  kalman    kf.*   local-linear-trend Kalman FILTER on log_close, (q_level, q_slope, r) by ML on the fit period only
                   (tools/lib/kalman_llt.py, from feature-eng cross_asset_h1 @c133ccd); filtered level - log_close,
                   filtered slope, innovation and innovation / sqrt(predicted variance)
  regime    rg.gmm.* sklearn GaussianMixture k = 3 diag, fitted on the fit-period rows of (logret_1, rv_24,
                   log_range) standardised with fit-period mean/std; posterior at t from the row at t alone
            rg.hmm.* numpy Gaussian HMM k = 3 diag (tools/lib/gaussian_hmm.py), EM on the fit period, causal FORWARD
                   FILTER (never forward-backward) emitting filtered state probabilities, entropy and the one-step
                   predictive surprise; states ordered by increasing fitted rv_24 mean
  fracdiff  fd.*   fractional difference of log_close with a FIXED window of 256 weights; d = smallest value in
                   {0.1..0.9} whose fit-period series passes ADF at p < 0.05 (table of ADF statistics kept)
  strength  wv.usd.* wv.eur.* hb.usd.* hb.eur.*  the wavelet and Hilbert producers above applied to the cross-asset
                   r4 strength RETURNS fx.usd_strength.ret_1 / fx.eur_strength.ret_1 (--cross-asset; returns are
                   used directly, never cumulated, so a missing source bar only voids the windows that contain it);
                   skipped with a recorded status when no cross-asset store is given

Diagnostics (ADF/KPSS/skew/kurtosis) and the NONE/DIFF/ASINH decisions reuse tools/layer1_pool_h1.py on the same
TRAIN period as layer-1 (2020-01-01 .. 2023-01-01, --diag-end). Future-perturbation check on real data: on a
prefix of the grid, rows after a cut replaced by noise and 500 synthetic rows appended -> every value at <= cut
identical; positive control: perturbing the bar AT the cut changes the family's value at the cut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.layer1_pool_h1 import decide, diagnostics  # noqa: E402
from tools.lib import gaussian_hmm, kalman_llt  # noqa: E402
from tools.lib.native_wavelet import compute_wavelet_native  # noqa: E402

PIP = 1e-4
TEST_START_CLOSED = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_START = pd.Timestamp("2020-01-01", tz="UTC")
LEARNED = ("kalman", "gmm", "hmm", "fracdiff")
FIXED = ("wavelet", "hilbert", "multitaper", "strength")
STRENGTH_INPUTS = {"usd": "fx.usd_strength.ret_1", "eur": "fx.eur_strength.ret_1"}
ALL = FIXED + LEARNED
CFG = {"wavelet": {"wavelet": "db4", "window": 256, "level": 4, "mode": "symmetric"},
       "hilbert": {"window": 128},
       "multitaper": {"window": 256, "NW": 3.5, "K": 5, "bands": {"gt48h": [0.0, 1 / 48], "12_48h": [1 / 48, 1 / 12], "4_12h": [1 / 12, 0.25], "lt4h": [0.25, 0.5]}},
       "kalman": {"burn_in": 100, "maxiter": 400},
       "gmm": {"k": 3, "covariance_type": "diag", "n_init": 5, "max_iter": 500, "reg_covar": 1e-6, "random_state": 0},
       "hmm": {"k": 3, "max_iter": 100, "tol": 1e-5, "var_floor": 1e-6},
       "fracdiff": {"window": 256, "d_grid": [round(0.1 * i, 1) for i in range(1, 10)], "adf_alpha": 0.05, "adf_maxlag": 24},
       "strength": {"wavelet": {"wavelet": "db4", "window": 256, "level": 4, "mode": "symmetric"}, "hilbert": {"window": 128}, "inputs": STRENGTH_INPUTS}}
REGIME_INPUTS = ("logret_1", "rv_24", "log_range")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- parents
def parents(df: pd.DataFrame) -> pd.DataFrame:
    """df: admitted rows (timestamp, BidOpen/High/Low/Close), already restricted to < test start."""
    c = df["BidClose"].to_numpy(float)
    h = df["BidHigh"].to_numpy(float)
    l = df["BidLow"].to_numpy(float)
    lc = np.log(c)
    r1 = np.r_[np.nan, np.diff(lc)]
    rv = np.sqrt(pd.Series(r1 ** 2).rolling(24, min_periods=24).sum().to_numpy())
    P = pd.DataFrame({"log_close": lc, "logret_1": r1, "log_range": np.log(h / l), "hl_pips": (h - l) / PIP, "rv_24": rv},
                     index=pd.DatetimeIndex(df["timestamp"]))
    for tag, col in STRENGTH_INPUTS.items():
        if col in df.columns:
            P[f"{tag}_strength_ret"] = df[col].to_numpy(float)
    return P


def _complete(x: np.ndarray, W: int) -> np.ndarray:
    """mask over t: the window [t-W+1, t] is complete and finite."""
    n = len(x)
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(x))])
    ok = np.zeros(n, bool)
    ok[W - 1:] = (bad[W:] - bad[:n - W + 1]) == 0
    return ok


# ----------------------------------------------------------------------------- fixed families
def fam_wavelet(P: pd.DataFrame, cfg=CFG["wavelet"]):
    out, cols = {}, {}
    for src in ("log_close", "logret_1"):
        fr, meta = compute_wavelet_native(pd.DataFrame({"timestamp": P.index, "x": P[src].to_numpy()}), window=cfg["window"],
                                          wavelet=cfg["wavelet"], level=cfg["level"], frozen_from={"fold_id": "layer2_h1", "train_end": "declared config, not derived from data"},
                                          sample_interval_seconds=3600, column="x", mode=cfg["mode"])
        tag = "lc" if src == "log_close" else "r1"
        for c in fr.columns[1:]:
            name = f"wv.{tag}.{c.replace('wavelet_native_', '')}"
            out[name] = fr[c].to_numpy()
            kind = "energy" if "energy" in c else "approximation" if c.endswith(f"A{cfg['level']}") else "detail"
            cols[name] = {"parent": src, "component": c.replace("wavelet_native_", ""), "unit": meta["units"][kind],
                          "algorithm": meta["method_name"], "proxy_of": None, "warmup_bars": cfg["window"] - 1}
    return pd.DataFrame(out, index=P.index), {"config": cfg, "library": "pywt " + meta["version"], "columns": cols,
                                              "emission": meta["emission"], "past_only": meta["past_only"]}


def _hilbert_windows(x: np.ndarray, W: int, detrend: bool, chunk: int = 4096):
    from scipy.signal import hilbert
    n = len(x)
    amp, cs, sn, fq = (np.full(n, np.nan) for _ in range(4))
    ok = _complete(x, W)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return amp, cs, sn, fq
    Xw = np.lib.stride_tricks.sliding_window_view(x, W)          # row i -> window ending at i + W - 1
    tt = np.arange(W, dtype=float)
    tt_c = tt - tt.mean()
    for s in range(0, len(idx), chunk):
        rows = idx[s:s + chunk]
        Z = np.array(Xw[rows - (W - 1)], float)
        mu = Z.mean(axis=1, keepdims=True)
        if detrend:
            beta = (Z - mu) @ tt_c / (tt_c @ tt_c)
            Z = Z - mu - beta[:, None] * tt_c[None, :]
        else:
            Z = Z - mu
        A = hilbert(Z, axis=1)
        last = A[:, -1]
        amp[rows] = np.abs(last)
        ph = np.unwrap(np.angle(A), axis=1)
        cs[rows], sn[rows] = np.cos(ph[:, -1]), np.sin(ph[:, -1])
        fq[rows] = (ph[:, -1] - ph[:, -2]) / (2 * np.pi)
    return amp, cs, sn, fq


def fam_hilbert(P: pd.DataFrame, cfg=CFG["hilbert"]):
    W = cfg["window"]
    out, cols = {}, {}
    for src, detrend, tag in (("logret_1", False, "r1"), ("log_close", True, "lc_detr")):
        amp, cs, sn, fq = _hilbert_windows(P[src].to_numpy(float), W, detrend)
        for comp, v, unit in (("amplitude", amp, "parent units (envelope of the demeaned/detrended window at its last sample)"),
                              ("phase_cos", cs, "dimensionless [-1, 1]"), ("phase_sin", sn, "dimensionless [-1, 1]"),
                              ("inst_freq", fq, "cycles per bar (difference of the unwrapped phase at the last two samples)")):
            name = f"hb.{tag}.{comp}"
            out[name] = v
            cols[name] = {"parent": src, "component": comp, "unit": unit, "proxy_of": None, "warmup_bars": W - 1,
                          "algorithm": "HILBERT_FFT_TRAILING_WINDOW" + ("_OLS_DETRENDED" if detrend else "_DEMEANED")}
    edge = ("scipy.signal.hilbert computes the analytic signal by FFT, which treats the W-bar window as periodic; the last "
            "sample therefore sees the wrap-around jump between x[t] and x[t-W+1] (Gibbs-type ringing). Demeaning (logret_1) "
            "and OLS detrending (log_close) within the window reduce the jump; it is not removed and no taper is applied, "
            "so the amplitude/phase at the right edge are biased relative to an infinite-record analytic signal. The value at "
            "t depends on the window [t-W+1, t] only, so appending rows after t never changes it (tested).")
    return pd.DataFrame(out, index=P.index), {"config": cfg, "columns": cols, "edge_effect": edge, "library": "scipy.signal.hilbert"}


def fam_multitaper(P: pd.DataFrame, cfg=CFG["multitaper"], chunk: int = 2048):
    from scipy.signal.windows import dpss
    W, K = cfg["window"], cfg["K"]
    x = P["logret_1"].to_numpy(float)
    n = len(x)
    tapers = dpss(W, cfg["NW"], Kmax=K)                     # (K, W), unit energy each
    freqs = np.fft.rfftfreq(W, d=1.0)                       # cycles per bar
    bands = cfg["bands"]
    masks = {b: (freqs >= lo) & (freqs < hi) & (freqs > 0) for b, (lo, hi) in bands.items()}
    masks["lt4h"] = masks["lt4h"] | (freqs == 0.5)
    names = [f"mt.r1.power_{b}" for b in bands] + ["mt.r1.spectral_entropy", "mt.r1.dominant_period_bars", "mt.r1.total_power"]
    out = np.full((n, len(names)), np.nan)
    ok = _complete(x, W)
    idx = np.flatnonzero(ok)
    Xw = np.lib.stride_tricks.sliding_window_view(x, W)
    nz = freqs > 0
    for s in range(0, len(idx), chunk):
        rows = idx[s:s + chunk]
        Z = np.array(Xw[rows - (W - 1)], float)
        Z = Z - Z.mean(axis=1, keepdims=True)
        psd = np.zeros((len(rows), len(freqs)))
        for k in range(K):
            F = np.fft.rfft(Z * tapers[k][None, :], axis=1)
            psd += (F.real ** 2 + F.imag ** 2)
        psd /= K
        psd[:, 1:-1] *= 2.0                                 # one-sided
        for j, b in enumerate(bands):
            out[rows, j] = psd[:, masks[b]].sum(axis=1)
        p = psd[:, nz]
        tot = p.sum(axis=1)
        pn = p / np.where(tot > 0, tot, 1.0)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.nansum(np.where(pn > 0, pn * np.log(pn), 0.0), axis=1) / np.log(p.shape[1])
        out[rows, len(bands)] = H
        out[rows, len(bands) + 1] = 1.0 / freqs[nz][np.argmax(p, axis=1)]
        out[rows, len(bands) + 2] = tot
    cols = {}
    for j, b in enumerate(bands):
        cols[names[j]] = {"parent": "logret_1", "component": f"band_power_{b}", "unit": "variance of logret_1 (one-sided multitaper PSD summed over the band, cycles/bar)",
                          "algorithm": "MULTITAPER_DPSS_TRAILING_WINDOW", "proxy_of": None, "warmup_bars": W - 1, "band_cycles_per_bar": bands[b]}
    cols[names[-3]] = {"parent": "logret_1", "component": "spectral_entropy", "unit": "dimensionless [0, 1] (Shannon entropy of the normalised PSD, DC excluded, / log(#bins))",
                       "algorithm": "MULTITAPER_DPSS_TRAILING_WINDOW", "proxy_of": None, "warmup_bars": W - 1}
    cols[names[-2]] = {"parent": "logret_1", "component": "dominant_period", "unit": "bars (1 / argmax-frequency, DC excluded; bounded by [2, 256])",
                       "algorithm": "MULTITAPER_DPSS_TRAILING_WINDOW", "proxy_of": None, "warmup_bars": W - 1}
    cols[names[-1]] = {"parent": "logret_1", "component": "total_power", "unit": "variance of logret_1 (sum of the PSD over all non-zero frequencies)",
                       "algorithm": "MULTITAPER_DPSS_TRAILING_WINDOW", "proxy_of": None, "warmup_bars": W - 1}
    return pd.DataFrame(out, index=P.index, columns=names), {"config": cfg, "columns": cols, "library": "scipy.signal.windows.dpss + numpy.fft.rfft",
                                                            "frequency_unit": "cycles per bar (bar = 1 h on the admitted grid)",
                                                            "window_treatment": "demeaned, no zero padding, K tapers averaged with equal weights"}


def fam_strength(P: pd.DataFrame, cfg=CFG["strength"]):
    """Wavelet (db4, W=256, L=4) and Hilbert (W=128, demeaned) on the r4 strength returns; empty when the inputs are absent."""
    out, cols = {}, {}
    present = [tag for tag in cfg["inputs"] if f"{tag}_strength_ret" in P.columns]
    if not present:
        return pd.DataFrame(index=P.index), {"config": cfg, "columns": {}, "status": "SKIPPED: no --cross-asset store with " + ", ".join(cfg["inputs"].values())}
    wc, hc = cfg["wavelet"], cfg["hilbert"]
    for tag in present:
        src = f"{tag}_strength_ret"
        fr, meta = compute_wavelet_native(pd.DataFrame({"timestamp": P.index, "x": P[src].to_numpy()}), window=wc["window"], wavelet=wc["wavelet"],
                                          level=wc["level"], frozen_from={"fold_id": "layer2_h1", "train_end": "declared config, not derived from data"},
                                          sample_interval_seconds=3600, column="x", mode=wc["mode"])
        for c in fr.columns[1:]:
            name = f"wv.{tag}.{c.replace('wavelet_native_', '')}"
            out[name] = fr[c].to_numpy()
            kind = "energy" if "energy" in c else "approximation" if c.endswith(f"A{wc['level']}") else "detail"
            cols[name] = {"parent": cfg["inputs"][tag], "component": c.replace("wavelet_native_", ""), "unit": meta["units"][kind].replace("price units for close", "log-return units"),
                          "algorithm": meta["method_name"], "proxy_of": None, "warmup_bars": wc["window"] - 1}
        amp, cs, sn, fq = _hilbert_windows(P[src].to_numpy(float), hc["window"], False)
        for comp, v, unit in (("amplitude", amp, "log-return units (envelope of the demeaned window at its last sample)"), ("phase_cos", cs, "dimensionless [-1, 1]"),
                              ("phase_sin", sn, "dimensionless [-1, 1]"), ("inst_freq", fq, "cycles per bar (difference of the unwrapped phase at the last two samples)")):
            name = f"hb.{tag}.{comp}"
            out[name] = v
            cols[name] = {"parent": cfg["inputs"][tag], "component": comp, "unit": unit, "proxy_of": None, "warmup_bars": hc["window"] - 1, "algorithm": "HILBERT_FFT_TRAILING_WINDOW_DEMEANED"}
    return pd.DataFrame(out, index=P.index), {"config": cfg, "columns": cols, "status": "GENERATED for " + ", ".join(present), "library": "pywt + scipy.signal.hilbert",
                                              "input_rule": "strength returns used directly (never cumulated, never filled); a missing source bar voids only the windows containing it",
                                              "edge_effect": "same as the hilbert family (FFT analytic signal on the demeaned window; right-edge wrap-around bias; causal by construction)"}


# ----------------------------------------------------------------------------- learned families: fit / apply
def fit_kalman(P: pd.DataFrame, fit_mask: np.ndarray, cfg=CFG["kalman"]):
    return kalman_llt.fit_llt(P["log_close"].to_numpy(float)[fit_mask], burn_in=cfg["burn_in"], maxiter=cfg["maxiter"])


def apply_kalman(P: pd.DataFrame, prm: dict):
    y = P["log_close"].to_numpy(float)
    lvl, slp, inn, inv = kalman_llt.filter_llt(y, prm["q_level"], prm["q_slope"], prm["r"])
    out = pd.DataFrame({"kf.lc.level_minus_close": lvl - y, "kf.lc.slope": slp, "kf.lc.innovation": inn,
                        "kf.lc.innovation_std": inn / np.sqrt(inv)}, index=P.index)
    cols = {"kf.lc.level_minus_close": {"component": "filtered_level_minus_observation", "unit": "log price"},
            "kf.lc.slope": {"component": "filtered_slope", "unit": "log price per bar"},
            "kf.lc.innovation": {"component": "innovation", "unit": "log price"},
            "kf.lc.innovation_std": {"component": "standardised_innovation", "unit": "dimensionless (innovation / sqrt of its predicted variance)"}}
    for c in cols:
        cols[c] |= {"parent": "log_close", "algorithm": "KALMAN_LOCAL_LINEAR_TREND_FILTER", "proxy_of": None, "warmup_bars": prm["burn_in"]}
    return out, {"columns": cols, "params": prm, "note": "filter only (no smoother); first bar initialises the state; the first burn_in bars are a diffuse warm-up (values emitted, flagged as warmup)"}


def _regime_matrix(P: pd.DataFrame) -> np.ndarray:
    return P[list(REGIME_INPUTS)].to_numpy(float)


def _standardise_fit(X: np.ndarray, fit_mask: np.ndarray):
    Xf = X[fit_mask]
    ok = np.isfinite(Xf).all(axis=1)
    mu, sd = Xf[ok].mean(axis=0), Xf[ok].std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def fit_gmm(P: pd.DataFrame, fit_mask: np.ndarray, cfg=CFG["gmm"]):
    from sklearn.mixture import GaussianMixture
    X = _regime_matrix(P)
    mu, sd = _standardise_fit(X, fit_mask)
    Z = (X[fit_mask] - mu) / sd
    Z = Z[np.isfinite(Z).all(axis=1)]
    g = GaussianMixture(n_components=cfg["k"], covariance_type=cfg["covariance_type"], n_init=cfg["n_init"], max_iter=cfg["max_iter"],
                        reg_covar=cfg["reg_covar"], random_state=cfg["random_state"]).fit(Z)
    order = np.argsort(g.means_[:, REGIME_INPUTS.index("rv_24")])
    return {"inputs": list(REGIME_INPUTS), "mu": mu.tolist(), "sd": sd.tolist(), "weights": g.weights_[order].tolist(),
            "means": g.means_[order].tolist(), "variances": g.covariances_[order].tolist(), "converged": bool(g.converged_),
            "n_iter": int(g.n_iter_), "lower_bound": float(g.lower_bound_), "fit_rows": int(len(Z)), "config": cfg,
            "state_rv24_mean_raw": (g.means_[order, 1] * sd[1] + mu[1]).tolist()}


def _gmm_posterior(Z: np.ndarray, prm: dict) -> np.ndarray:
    means, var, w = np.asarray(prm["means"]), np.asarray(prm["variances"]), np.asarray(prm["weights"])
    d = Z[:, None, :] - means[None]
    lp = np.log(w)[None, :] - 0.5 * (np.sum(np.log(2 * np.pi * var) + d * d / var, axis=2))
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def apply_gmm(P: pd.DataFrame, prm: dict):
    X = _regime_matrix(P)
    Z = (X - np.asarray(prm["mu"])) / np.asarray(prm["sd"])
    ok = np.isfinite(Z).all(axis=1)
    k = len(prm["weights"])
    post = np.full((len(Z), k), np.nan)
    post[ok] = _gmm_posterior(Z[ok], prm)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(post > 0, post * np.log(post), 0.0), axis=1)
    ent[~ok] = np.nan
    out = {f"rg.gmm.p{j}": post[:, j] for j in range(k)}
    out["rg.gmm.entropy"] = ent
    cols = {f"rg.gmm.p{j}": {"component": f"posterior_state_{j}_(vol_rank_{j})", "unit": "probability [0, 1]"} for j in range(k)}
    cols["rg.gmm.entropy"] = {"component": "posterior_entropy", "unit": "nats [0, log k]"}
    for c in cols:
        cols[c] |= {"parent": "+".join(REGIME_INPUTS), "algorithm": "GAUSSIAN_MIXTURE_DIAG_K3_ROWWISE_POSTERIOR", "proxy_of": None, "warmup_bars": 24}
    return pd.DataFrame(out, index=P.index), {"columns": cols, "params": prm, "note": "posterior of the row at t alone; no temporal smoothing; states ordered by increasing fitted rv_24 mean"}


def fit_hmm(P: pd.DataFrame, fit_mask: np.ndarray, cfg=CFG["hmm"]):
    X = _regime_matrix(P)
    mu, sd = _standardise_fit(X, fit_mask)
    Z = (X[fit_mask] - mu) / sd
    prm = gaussian_hmm.fit(Z, cfg["k"], max_iter=cfg["max_iter"], tol=cfg["tol"], var_floor=cfg["var_floor"], order_dim=REGIME_INPUTS.index("rv_24"))
    prm |= {"inputs": list(REGIME_INPUTS), "mu": mu.tolist(), "sd": sd.tolist(), "config": cfg,
            "state_rv24_mean_raw": (np.asarray(prm["means"])[:, 1] * sd[1] + mu[1]).tolist(),
            "fit_note": "EM on the fit-period rows with finite inputs treated as one sequence (rows with a NaN input dropped from the fit sequence only)"}
    return prm


def apply_hmm(P: pd.DataFrame, prm: dict):
    X = _regime_matrix(P)
    Z = (X - np.asarray(prm["mu"])) / np.asarray(prm["sd"])
    filt, ent, sur = gaussian_hmm.forward_filter(Z, prm)
    k = prm["k"]
    out = {f"rg.hmm.p{j}": filt[:, j] for j in range(k)}
    out["rg.hmm.entropy"] = ent
    out["rg.hmm.surprise"] = sur
    cols = {f"rg.hmm.p{j}": {"component": f"filtered_state_{j}_(vol_rank_{j})", "unit": "probability [0, 1]"} for j in range(k)}
    cols["rg.hmm.entropy"] = {"component": "filtered_posterior_entropy", "unit": "nats [0, log k]"}
    cols["rg.hmm.surprise"] = {"component": "one_step_predictive_surprise", "unit": "nats (-log p(x_t | x_<t) under the frozen HMM, standardised inputs)"}
    for c in cols:
        cols[c] |= {"parent": "+".join(REGIME_INPUTS), "algorithm": "GAUSSIAN_HMM_DIAG_K3_FORWARD_FILTER", "proxy_of": None, "warmup_bars": 24}
    return pd.DataFrame(out, index=P.index), {"columns": cols, "params": {kk: v for kk, v in prm.items() if kk != "loglik_history"},
                                              "note": "causal forward filter P(s_t | x_1..x_t); never forward-backward; NaN-input rows predict through the transition matrix and emit NaN"}


def fracdiff_weights(d: float, window: int) -> np.ndarray:
    w = np.empty(window)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def fracdiff_fixed(x: np.ndarray, d: float, window: int) -> np.ndarray:
    w = fracdiff_weights(d, window)
    n = len(x)
    y = np.full(n, np.nan)
    ok = _complete(x, window)
    conv = np.convolve(np.where(np.isfinite(x), x, 0.0), w, mode="full")[:n]
    y[ok] = conv[ok]
    return y


def fit_fracdiff(P: pd.DataFrame, fit_mask: np.ndarray, cfg=CFG["fracdiff"]):
    from statsmodels.tsa.stattools import adfuller
    x = P["log_close"].to_numpy(float)
    table, chosen = [], None
    for d in cfg["d_grid"]:
        y = fracdiff_fixed(x, d, cfg["window"])[fit_mask]
        y = y[np.isfinite(y)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, used_lag, nobs, crit = adfuller(y, maxlag=cfg["adf_maxlag"], autolag=None)[:5]
        row = {"d": d, "adf_stat": float(stat), "adf_p": float(p), "nobs": int(nobs), "crit_5pct": float(crit["5%"]),
               "weights_abs_sum_dropped": float(np.abs(fracdiff_weights(d, cfg["window"] + 2000)[cfg["window"]:]).sum())}
        table.append(row)
        if chosen is None and p < cfg["adf_alpha"]:
            chosen = d
    return {"d": chosen, "window": cfg["window"], "adf_table": table, "rule": f"smallest d in the grid with ADF p < {cfg['adf_alpha']} on the fit period (maxlag {cfg['adf_maxlag']}, no autolag)", "config": cfg}


def apply_fracdiff(P: pd.DataFrame, prm: dict):
    d = prm["d"]
    if d is None:
        return pd.DataFrame(index=P.index), {"columns": {}, "params": prm, "status": "PENDING: no d in the grid passes ADF"}
    y = fracdiff_fixed(P["log_close"].to_numpy(float), d, prm["window"])
    name = f"fd.lc.d{d:.1f}"
    cols = {name: {"parent": "log_close", "component": f"fracdiff_d{d:.1f}_w{prm['window']}", "unit": "log price (fractionally differenced, fixed-width weights)",
                   "algorithm": "FRACTIONAL_DIFFERENCE_FIXED_WINDOW", "proxy_of": None, "warmup_bars": prm["window"] - 1}}
    return pd.DataFrame({name: y}, index=P.index), {"columns": cols, "params": prm}


FIT = {"kalman": fit_kalman, "gmm": fit_gmm, "hmm": fit_hmm, "fracdiff": fit_fracdiff}
APPLY = {"kalman": apply_kalman, "gmm": apply_gmm, "hmm": apply_hmm, "fracdiff": apply_fracdiff,
         "wavelet": lambda P, prm: fam_wavelet(P), "hilbert": lambda P, prm: fam_hilbert(P), "multitaper": lambda P, prm: fam_multitaper(P),
         "strength": lambda P, prm: fam_strength(P)}


def build(P: pd.DataFrame, families, fit_mask: np.ndarray, params: dict | None = None):
    """Fit (when params missing) and apply the families. Returns (frame, meta per family, params per family)."""
    params = dict(params or {})
    frames, meta = [], {}
    for fam in families:
        if fam in FIT and fam not in params:
            params[fam] = FIT[fam](P, fit_mask)
        fr, m = APPLY[fam](P, params.get(fam))
        frames.append(fr)
        meta[fam] = m
    return pd.concat(frames, axis=1), meta, params


# ----------------------------------------------------------------------------- future perturbation on real data
def synthetic_rows(last: pd.Series, n: int, rng, start_ts):
    lc = np.log(last["BidClose"])
    steps = rng.normal(0, 0.0008, n)
    c = np.exp(lc + np.cumsum(steps))
    o = np.r_[last["BidClose"], c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"timestamp": pd.date_range(start_ts, periods=n, freq="h"), "BidOpen": o, "BidHigh": h, "BidLow": l, "BidClose": c})


def perturbation_check(raw: pd.DataFrame, families, params: dict, n_prefix: int = 6000, cut: int = 4000, n_append: int = 500, seed: int = 7):
    """Apply the FROZEN producers to (a) the prefix, (b) the prefix with rows after the cut replaced by noise and
    500 synthetic rows appended, (c) the prefix with the bar AT the cut perturbed. Reports per family."""
    rng = np.random.default_rng(seed)
    base = raw.iloc[:n_prefix].reset_index(drop=True)
    P0 = parents(base)
    F0, _, _ = build(P0, families, np.ones(len(P0), bool), params)
    strength = [c for c in STRENGTH_INPUTS.values() if c in base.columns]
    noise = base.copy()
    syn = synthetic_rows(base.iloc[cut], n_prefix - cut - 1, rng, base["timestamp"].iloc[cut + 1])
    for k in ("BidOpen", "BidHigh", "BidLow", "BidClose"):
        v = noise[k].to_numpy(float).copy()
        v[cut + 1:] = syn[k].to_numpy()
        noise[k] = v
    for k in strength:
        v = noise[k].to_numpy(float).copy()
        v[cut + 1:] = rng.normal(0, 0.001, len(v) - cut - 1)
        noise[k] = v
    app = synthetic_rows(noise.iloc[-1], n_append, rng, noise["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
    for k in strength:
        app[k] = rng.normal(0, 0.001, n_append)
    pert = pd.concat([noise, app], ignore_index=True)
    F1, _, _ = build(parents(pert), families, np.ones(len(pert), bool), params)
    ctrl = base.copy()
    for k in ("BidOpen", "BidHigh", "BidLow", "BidClose"):
        ctrl.loc[cut, k] = ctrl.loc[cut, k] * np.exp(0.002 if k != "BidLow" else -0.002)
    for k in strength:
        ctrl.loc[cut, k] = (0.0 if not np.isfinite(ctrl.loc[cut, k]) else ctrl.loc[cut, k]) + 0.002
    F2, _, _ = build(parents(ctrl), families, np.ones(len(ctrl), bool), params)
    report = {}
    for fam in families:
        cols = [c for c in F0.columns if _fam_of(c) == fam]
        if not cols:
            report[fam] = {"columns": 0, "passed": None, "status": "no columns produced (family skipped)"}
            continue
        a, b, c2 = F0[cols].to_numpy()[:cut + 1], F1[cols].to_numpy()[:cut + 1], F2[cols].to_numpy()[cut]
        same = np.array_equal(a, b, equal_nan=True)
        diff = np.nanmax(np.abs(a - b)) if not same else 0.0
        changed_at_cut = int(np.sum(~np.isclose(F0[cols].to_numpy()[cut], c2, equal_nan=True)))
        after = int(np.sum(~np.isclose(F0[cols].to_numpy()[cut + 1:], F1[cols].to_numpy()[cut + 1:n_prefix], equal_nan=True)))
        report[fam] = {"columns": len(cols), "rows_at_or_before_cut": cut + 1, "bit_identical_at_or_before_cut": bool(same),
                       "max_abs_diff_at_or_before_cut": float(diff), "cells_changed_after_cut": after,
                       "positive_control_columns_changed_at_cut": changed_at_cut, "positive_control_passed": changed_at_cut > 0,
                       "passed": bool(same and changed_at_cut > 0), "appended_rows": n_append, "prefix_rows": n_prefix}
    return report


def _fam_of(col: str) -> str:
    p = col.split(".")
    if p[0] in ("wv", "hb") and p[1] in STRENGTH_INPUTS:
        return "strength"
    return {"wv": "wavelet", "hb": "hilbert", "mt": "multitaper", "kf": "kalman", "fd": "fracdiff"}.get(p[0]) or ("gmm" if p[1] == "gmm" else "hmm")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--admitted", required=True, help="admitted EURUSD H1 parquet (timestamp = UTC bar start, Bid OHLC)")
    ap.add_argument("--admitted-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-end", default="2024-01-01", help="exclusive end of the fit period for the learned families (fit start 2020-01-01)")
    ap.add_argument("--diag-end", default="2023-01-01", help="exclusive end of the TRAIN diagnostics period (same as layer-1)")
    ap.add_argument("--test-start-closed", default="2025-01-01")
    ap.add_argument("--families", default="all", help="'all', 'learned', 'fixed' or a comma list")
    ap.add_argument("--suffix", default="", help="file suffix, e.g. _fit_end_2023-01-01")
    ap.add_argument("--cross-asset", default=None, help="cross-asset r4 store parquet (fx.usd_strength.ret_1 / fx.eur_strength.ret_1) for the strength family")
    ap.add_argument("--cross-asset-sha256", default=None)
    ap.add_argument("--skip-perturbation", action="store_true")
    a = ap.parse_args(argv)
    if sha(a.admitted) != a.admitted_sha256:
        raise SystemExit("ADMITTED_SHA256_MISMATCH")
    test_start = pd.Timestamp(a.test_start_closed, tz="UTC")
    raw = pd.read_parquet(a.admitted, columns=["timestamp", "BidOpen", "BidHigh", "BidLow", "BidClose"])
    raw["timestamp"] = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"])).tz_convert("UTC")
    n_total = len(raw)
    raw = raw[raw["timestamp"] < test_start].reset_index(drop=True)
    if not raw["timestamp"].is_monotonic_increasing:
        raise SystemExit("GRID_NOT_SORTED")
    cross_asset_info = None
    if a.cross_asset:
        if a.cross_asset_sha256 and sha(a.cross_asset) != a.cross_asset_sha256:
            raise SystemExit("CROSS_ASSET_SHA256_MISMATCH")
        ca = pd.read_parquet(a.cross_asset, columns=["timestamp", *STRENGTH_INPUTS.values()])
        ca.index = pd.DatetimeIndex(pd.to_datetime(ca.pop("timestamp"))).tz_convert("UTC")
        if (ca.index >= test_start).any():
            raise SystemExit("TEST_ROWS_FORBIDDEN")
        ca = ca.reindex(raw["timestamp"])                       # exact bar-start alignment, no fill
        for c in STRENGTH_INPUTS.values():
            raw[c] = ca[c].to_numpy(float)
        cross_asset_info = {"file": Path(a.cross_asset).name, "sha256": sha(a.cross_asset), "columns": list(STRENGTH_INPUTS.values()),
                            "finite_share": {c: float(np.isfinite(raw[c].to_numpy(float)).mean()) for c in STRENGTH_INPUTS.values()}}
    fams = ALL if a.families == "all" else LEARNED if a.families == "learned" else FIXED if a.families == "fixed" else tuple(a.families.split(","))
    P = parents(raw)
    ts = P.index
    fit_end = pd.Timestamp(a.fit_end, tz="UTC")
    fit_mask = np.asarray((ts >= TRAIN_START) & (ts < fit_end))
    X, meta, params = build(P, fams, fit_mask)
    # diagnostics + decisions on TRAIN only (layer-1 rules)
    train = np.asarray((ts >= TRAIN_START) & (ts < pd.Timestamp(a.diag_end, tz="UTC")))
    table, extra, colmeta = [], {}, {}
    for fam in fams:
        for c, cm in meta[fam]["columns"].items():
            colmeta[c] = cm | {"family": fam}
    for c in X.columns:
        x = X[c].to_numpy(float)
        d = diagnostics(x[train])
        dec, why, y = decide(c, x, d)
        table.append({"feature": c, "family": colmeta[c]["family"], **d, "decision": dec, "reason": why})
        if y is not None:
            extra[f"{c}@{dec.lower()}"] = y
    if extra:
        X = pd.concat([X, pd.DataFrame(extra, index=ts)], axis=1)
    pert = None if a.skip_perturbation else perturbation_check(raw, fams, params)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"layer2_pool_eurusd_h1{a.suffix}.parquet"
    store = X.reset_index().rename(columns={"index": "timestamp"})
    store.to_parquet(out / fname, index=False)
    diag = pd.DataFrame(table)
    diag.to_csv(out / f"FEATURE_DIAGNOSTICS{a.suffix}.csv", index=False)
    fam_counts = {f: int(sum(1 for c in colmeta if colmeta[c]["family"] == f)) for f in fams}
    man = {"schema": "layer2_pool_h1.v1", "producer": "tools/layer2_pool_h1.py", "source_sha256": a.admitted_sha256, "rows": int(len(store)),
           "rows_in_source": n_total, "rows_dropped_test": int(n_total - len(store)), "test_start_closed": str(test_start), "test_read": False,
           "span": [str(ts[0]), str(ts[-1])], "grid": "UTC bar start; decision at bar start + 1 h; weekend closures are consecutive bars",
           "fit_period": {"start": str(TRAIN_START), "end_exclusive": str(fit_end), "rows": int(fit_mask.sum()), "applies_to": list(f for f in fams if f in LEARNED)},
           "diagnostics_period": {"start": str(TRAIN_START), "end_exclusive": a.diag_end, "rows": int(train.sum())},
           "families": list(fams), "family_counts": fam_counts, "cross_asset_input": cross_asset_info, "features": list(X.columns), "columns": colmeta | {k: colmeta[k.split("@")[0]] | {"derived_from": k.split("@")[0]} for k in extra},
           "family_meta": {f: {k: v for k, v in meta[f].items() if k != "columns"} for f in fams},
           "parents": {"log_close": "log(BidClose)", "logret_1": "diff(log_close)", "log_range": "log(BidHigh/BidLow)", "hl_pips": "(BidHigh-BidLow)/1e-4",
                       "rv_24": "sqrt(rolling 24-bar sum of logret_1^2), min 24 bars",
                       "usd_strength_ret": "fx.usd_strength.ret_1 from the cross-asset r4 store (as is)", "eur_strength_ret": "fx.eur_strength.ret_1 from the cross-asset r4 store (as is)"},
           "decisions": {r["feature"]: r["decision"] for r in table}, "decision_counts": diag.decision.value_counts().to_dict(),
           "future_perturbation": pert, "file": fname, "file_sha256": sha(out / fname),
           "defects_absent": ["wavelet is PyWavelets wavedec on a trailing window (not a moving-average proxy)",
                              "no interpolate/ffill anywhere: incomplete or non-finite windows emit NaN",
                              "window sizes are declared constants, never derived from len(dataset)",
                              "regime models are fitted on the fit period only and the HMM uses the forward filter (no forward-backward / predict_proba over the sequence)",
                              "no TEST row (>= test_start_closed) is read into any artifact"],
           "scaling_note": "no scaling in the store; the model loader standardizes every feature with TRAIN-only mean/std"}
    (out / f"MANIFEST{a.suffix}.json").write_text(json.dumps(man, indent=1, default=str))
    L = [f"# Layer-2 pool{a.suffix}: per-feature diagnostics (TRAIN {TRAIN_START.date()} .. {a.diag_end}) and preprocessing decisions", "",
         f"Features {len(X.columns)} ({fam_counts}); fit period for learned families {TRAIN_START.date()} .. {a.fit_end} (exclusive); decisions {man['decision_counts']}.", "",
         "Future perturbation (real-data prefix, rows after the cut replaced by noise + 500 appended; positive control = bar at the cut perturbed):", ""]
    if pert:
        L += ["| family | columns | identical <= cut | max abs diff | cells changed after cut | positive control cols changed | passed |", "|---|---|---|---|---|---|---|"]
        L += [f"| {f} | {r['columns']} | {r['bit_identical_at_or_before_cut']} | {r['max_abs_diff_at_or_before_cut']:.3g} | {r['cells_changed_after_cut']} | {r['positive_control_columns_changed_at_cut']} | {r['passed']} |" if r["columns"] else f"| {f} | 0 | — | — | — | — | {r['status']} |" for f, r in pert.items()]
    L += ["", "| feature | family | ADF p | KPSS p | skew | kurt | decision | reason |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"  # noqa: E731
    for r in table:
        L.append(f"| {r['feature']} | {r['family']} | {fmt(r.get('adf_p'))} | {fmt(r.get('kpss_p'))} | {fmt(r.get('skew'))} | {fmt(r.get('kurtosis'))} | {r['decision']} | {r['reason']} |")
    (out / f"FEATURE_DIAGNOSTICS{a.suffix}.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"features": len(X.columns), "family_counts": fam_counts, "decisions": man["decision_counts"], "fit_end": a.fit_end,
                      "perturbation": {f: r.get("passed") for f, r in (pert or {}).items()},
                      "params": {f: {k: v for k, v in params[f].items() if k in ("q_level", "q_slope", "r", "d", "state_rv24_mean_raw", "iterations", "converged")} for f in params}}, default=str))


if __name__ == "__main__":
    main()
