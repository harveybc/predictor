"""Lane H: three paired arms for the causal Kalman family. DEVELOPMENT, nothing here is a confirmed result.

Arms on the SAME rows, targets, scaler, split, learner and budget:
  A  original input;
  B  original + Kalman outputs of the declared features;
  C  Kalman outputs REPLACING only the declared features (everything else stays original);
plus controls: identity (== A, bitwise), comparable causal EWMA (replaces like C), non-causal smoother (a deliberate
rejection: it must look better than any causal arm if it leaks, and is never an eligible input), and equal-capacity
permutation and noise controls of B.

Learner: ridge, closed form through the eigendecomposition of the Gram matrix; alpha chosen ONLY on a chronological
inner hold-out of the TRAIN origins (purged), then refit on all TRAIN origins; the validation split is read once per
arm for the report. numpy float64 with one BLAS thread. Metrics are MAE/MSE in the train-standardized target space
of the lane F2 dataset, beside the same-row naives (zero-return, persistence, seasonal-6, train-mean).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")

ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
FIXED_ALPHAS = (1.0, 1e3, 1e5)


def sha_array(a, dtype="<f8"):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------- windows
def gather(C: np.ndarray, origins: np.ndarray, lags: int) -> np.ndarray:
    """(N, lags*P): lag 0 (the origin row) first, then lag 1 ... lags-1, each block the P channels of that row."""
    blocks = [C[origins - l] for l in range(lags)]
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------------------------- ridge
def _gram_and_rhs(X, Y, mu, sd, block=2048):
    P = X.shape[1]
    G = np.zeros((P, P))
    R = np.zeros((P, Y.shape[1]))
    for i in range(0, X.shape[0], block):
        A = (X[i:i + block] - mu) / sd
        G += A.T @ A
        R += A.T @ Y[i:i + block]
    return G, R


def ridge_fit_predict_multi(Xtr, Ytr, Xev_list, alphas=ALPHAS, fixed=FIXED_ALPHAS, holdout_frac=0.2, purge=30):
    """Choose alpha on a chronological inner hold-out of the train rows (rows are in time order), refit on all train.

    Also returns the predictions of the refit at each ``fixed`` alpha (a capacity-neutral comparison across arms)."""
    N = Xtr.shape[0]
    cut = int(N * (1.0 - holdout_frac))
    inner_tr = slice(0, cut - purge)
    inner_ho = slice(cut, N)
    mu = Xtr[inner_tr].mean(axis=0)
    sd = Xtr[inner_tr].std(axis=0)
    sd[sd == 0] = 1.0
    ym = Ytr[inner_tr].mean(axis=0)
    G, R = _gram_and_rhs(Xtr[inner_tr], Ytr[inner_tr] - ym, mu, sd)
    w, V = np.linalg.eigh(G)
    VtR = V.T @ R
    Aho = (Xtr[inner_ho] - mu) / sd
    AV = Aho @ V
    inner = {}
    for a in alphas:
        pred = AV @ (VtR / (w[:, None] + a)) + ym
        inner[a] = float(np.mean(np.abs(pred - Ytr[inner_ho])))
    best = min(alphas, key=lambda a: (inner[a], a))
    del G, V, AV, Aho
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    ym = Ytr.mean(axis=0)
    G, R = _gram_and_rhs(Xtr, Ytr - ym, mu, sd)
    w, V = np.linalg.eigh(G)
    VtR = V.T @ R
    del G
    evs = [((X - mu) / sd) @ V for X in Xev_list]
    def at(a):
        coef_eig = VtR / (w[:, None] + a)
        return [E @ coef_eig + ym for E in evs]
    return {"chosen": best, "inner": inner, "preds_chosen": at(best), "preds_fixed": {a: at(a) for a in fixed}}


def ridge_fit_predict(Xtr, Ytr, Xev_list, alphas=ALPHAS, holdout_frac=0.2, purge=30):
    r = ridge_fit_predict_multi(Xtr, Ytr, Xev_list, alphas=alphas, fixed=(), holdout_frac=holdout_frac, purge=purge)
    return r["preds_chosen"], r["chosen"], r["inner"]


# --------------------------------------------------------------------------------------------- metrics
def naive_predictions(z_target_rows, origins, horizons, mu, sigma, seasonal_period=6):
    """Dict naive -> (N, H); same definitions as lane F2 (tools/eth_forecast_naives.py)."""
    N, H = len(origins), len(horizons)
    out = {"persistence_last_value": np.repeat(z_target_rows[origins][:, None], H, axis=1),
           "zero_return": np.broadcast_to(np.array([-h * mu / sigma for h in horizons])[None, :], (N, H)).copy(),
           "train_mean": np.zeros((N, H))}
    seas = np.full((N, H), np.nan)
    for k, h in enumerate(horizons):
        if h <= seasonal_period:
            s = np.zeros(N)
            for j in range(1, h + 1):
                s += z_target_rows[origins - seasonal_period + j]
            seas[:, k] = s
    out[f"seasonal_{seasonal_period}"] = seas
    return out


def per_horizon_metrics(pred, Y):
    e = pred - Y
    return {"MAE": np.mean(np.abs(e), axis=0), "MSE": np.mean(e ** 2, axis=0)}


def block_bootstrap_ci(d, L, B=2000, seed=0, level=0.95):
    """Moving-block bootstrap of the mean of a paired per-row difference series (rows are consecutive origins)."""
    n = len(d)
    L = max(1, min(L, n))
    nb = int(math.ceil(n / L))
    rng = np.random.RandomState(seed)
    starts = rng.randint(0, n - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def quarter_deltas(d, k=4):
    edges = np.linspace(0, len(d), k + 1).astype(int)
    return [float(d[edges[i]:edges[i + 1]].mean()) for i in range(k)]


# ------------------------------------------------------------------------------- Kalman channel builders
def log_var(v):
    """log of a strictly positive variance with libm (math.log), element-wise."""
    flat = v.ravel()
    return np.array([math.log(x) if x == x and x > 0 else float("nan") for x in flat.tolist()]).reshape(v.shape)


def kalman_channels(out: "kf.KalmanOutput", names=None):
    """Eligible channel matrix and names for every output; state_var enters as log(state_var)."""
    if not isinstance(out, kf.KalmanOutput) or not out.causal_forward or not out.eligible_for_inputs:
        raise kf.OperatorRefusal("this output is not an eligible input: only the forward causal filter produces "
                                 "eligible inputs; the backward smoother is a non-causal control")
    outputs = [o for o in kf.OUTPUTS[out.kind] if o not in ("obs", "state_var", "slope_var")]
    M, labels = kf.eligible_matrix(out, names=names, outputs=outputs)
    cols = list(range(len(out.column_ids))) if names is None else [out.column_ids.index(n) for n in names]
    extra, extra_labels = [], []
    for j in cols:
        extra.append(log_var(out.arrays["state_var"][:, j]))
        extra_labels.append(f"{out.column_ids[j]}__kf_logvar")
    return np.concatenate([M, np.stack(extra, axis=1)], axis=1), labels + extra_labels


# ------------------------------------------------------------------------------------- diagnostics
def lag1_acf(v):
    v = v[np.isfinite(v)]
    v = v - v.mean()
    d = float(np.dot(v, v))
    return float(np.dot(v[1:], v[:-1]) / d) if d > 0 else float("nan")


def innovation_stability(zinn_train, zinn_val, window=500):
    def block(z):
        z = z[np.isfinite(z)]
        roll = [float(z[i:i + window].std()) for i in range(0, len(z) - window + 1, window)]
        q = 0.0
        n = len(z)
        zc = z - z.mean()
        den = float(np.dot(zc, zc))
        for k in range(1, 11):
            rho = float(np.dot(zc[k:], zc[:-k]) / den) if den > 0 else 0.0
            q += rho * rho / (n - k)
        return {"mean": float(z.mean()), "std": float(z.std()), "acf1": lag1_acf(z), "ljung_box_q10": float(n * (n + 2) * q),
                "rolling_std_min": min(roll) if roll else None, "rolling_std_max": max(roll) if roll else None, "n": int(n)}
    return {"train": block(zinn_train), "validation": block(zinn_val)}


def phase_lag(obs, level, max_lag=6):
    """Lag k >= 0 (bars) maximising corr(diff(level)_t, diff(obs)_{t-k}); 0 means no measurable delay."""
    do, dl = np.diff(obs), np.diff(level)
    m = np.isfinite(do) & np.isfinite(dl)
    best, bestc = 0, -2.0
    cs = []
    for k in range(0, max_lag + 1):
        a, b = dl[k:], do[:len(do) - k]
        mm = m[k:] & m[:len(m) - k]
        if mm.sum() < 50 or a[mm].std() == 0 or b[mm].std() == 0:
            cs.append(None)
            continue
        c = float(np.corrcoef(a[mm], b[mm])[0, 1])
        cs.append(c)
        if c > bestc:
            best, bestc = k, c
    return {"best_lag_bars": best, "corr_at_best": bestc, "corr_by_lag": cs}


def extremes_retained(obs, level, zinnov, train_med, train_std, thr=3.0, zthr=2.0):
    """On observations whose train-scaled deviation exceeds ``thr`` train standard deviations: the median ratio
    |level - median| / |obs - median| (how much of the excursion the state keeps) and the share of those events where
    the standardized innovation exceeds ``zthr`` in magnitude (the excursion is visible in the innovation channel)."""
    dev = np.abs(obs - train_med) / train_std
    m = np.isfinite(dev) & (dev > thr) & np.isfinite(zinnov)
    if m.sum() == 0:
        return {"events": 0}
    ratio = np.abs(level[m] - train_med) / np.abs(obs[m] - train_med)
    return {"events": int(m.sum()), "median_level_retention": float(np.median(ratio)),
            "share_abs_zinnov_over_threshold": float(np.mean(np.abs(zinnov[m]) > zthr))}
