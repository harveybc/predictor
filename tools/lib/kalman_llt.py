"""Causal local-linear-trend Kalman FILTER with (Q, R) by maximum likelihood on the fit period only.

Origin / attribution: the filter follows ``kalman_local_trend`` / ``kalman_q_mle`` of feature-eng
``app/plugins/cross_asset_h1.py`` (branch satoshi/cross-asset-h1-20261020 @c133ccd): state [level, slope],
F = [[1, 1], [0, 1]], prediction/update only, never a smoother, NaN observations skipped (prediction only).
Differences from the origin, both deliberate: (1) the observation noise R and the two state noises
(q_level, q_slope) are all estimated, by Nelder-Mead on the exact innovation log-likelihood
-1/2 sum(log 2 pi S_t + e_t^2 / S_t) instead of a 7-point grid on q with R fixed to 1; (2) the filter is written in
scalar form (2x2 algebra expanded) so the ~30k-bar fit-period likelihood is cheap enough to optimise.

The series is treated as consecutive bars on the admitted grid (a weekend closure is one step, as every trailing
window in layer-1 does). The first ``burn_in`` innovations are excluded from the likelihood (diffuse start).
"""
from __future__ import annotations

import math

import numpy as np


def filter_llt(y: np.ndarray, q_level: float, q_slope: float, r: float, *, p0: float = 1.0):
    """Returns (level, slope, innovation, innovation_variance) per t, each using y[:t+1] only. NaN rows are
    predicted through (no update) and emit NaN innovation; level/slope are the predicted values there."""
    n = len(y)
    lvl = np.full(n, np.nan)
    slp = np.full(n, np.nan)
    inn = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    started = False
    a = b = 0.0
    p11 = p12 = p22 = 0.0
    for t in range(n):
        yt = y[t]
        if not started:
            if yt == yt:              # finite
                a, b = float(yt), 0.0
                p11, p12, p22 = p0, 0.0, p0
                started = True
                lvl[t], slp[t] = a, b
            continue
        # predict: x = F x ; P = F P F' + Q
        a = a + b
        n11 = p11 + 2.0 * p12 + p22 + q_level
        n12 = p12 + p22
        n22 = p22 + q_slope
        p11, p12, p22 = n11, n12, n22
        if yt == yt:
            s = p11 + r
            e = float(yt) - a
            k1 = p11 / s
            k2 = p12 / s
            a += k1 * e
            b += k2 * e
            # P = P - K H P  (H = [1, 0])
            p11, p12, p22 = p11 - k1 * p11, p12 - k1 * p12, p22 - k2 * p12
            inn[t], inv[t] = e, s
        lvl[t], slp[t] = a, b
    return lvl, slp, inn, inv


def negloglik(theta, y, burn_in):
    q_level, q_slope, r = (math.exp(v) for v in theta)
    _, _, e, s = filter_llt(y, q_level, q_slope, r)
    e, s = e[burn_in:], s[burn_in:]
    ok = np.isfinite(e)
    e, s = e[ok], s[ok]
    return 0.5 * float(np.sum(np.log(2 * np.pi * s) + e * e / s))


def fit_llt(y_fit: np.ndarray, *, burn_in: int = 100, maxiter: int = 400) -> dict:
    """Maximum likelihood of (q_level, q_slope, r) on the fit-period series only."""
    from scipy.optimize import minimize
    y = np.asarray(y_fit, float)
    d = np.diff(y[np.isfinite(y)])
    v = float(np.var(d)) or 1e-8
    x0 = np.log([v * 0.5, v * 1e-3, v * 0.5])
    res = minimize(negloglik, x0, args=(y, burn_in), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-3, "adaptive": True})
    q_level, q_slope, r = (float(math.exp(v)) for v in res.x)
    tiny = 1e-6 * max(q_level, r)
    boundary = [k for k, v in (("q_level", q_level), ("q_slope", q_slope), ("r", r)) if v < tiny]
    return {"q_level": q_level, "q_slope": q_slope, "r": r, "negloglik": float(res.fun), "iterations": int(res.nit),
            "boundary_hits": boundary,
            "boundary_note": ("a parameter below 1e-6 x max(q_level, r) is at the zero boundary of the likelihood: with q_slope ~ 0 the "
                              "local linear trend degenerates to a local level with a drift that converges to a constant, so the slope "
                              "output carries little information" if boundary else "no parameter at the zero boundary"),
            "function_evaluations": int(res.nfev), "converged": bool(res.success), "burn_in": int(burn_in),
            "fit_rows": int(np.isfinite(y).sum()), "init_var_diff": v,
            "method": "Nelder-Mead on log(q_level, q_slope, r), exact innovation likelihood"}
