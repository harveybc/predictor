"""C158: an independent, slow, prefix-only reference for the causal operator
bank. Tests only.

`outputs_at(spec, fitted, fit_mode, prefix, t0)` computes the outputs, reason
and (for trailing_haar_threshold) per-level outputs of the LAST row of
`prefix`, a list of rows (lists of Python floats, NaN for missing) holding
exactly the rows visible at that instant. It is called once per t with
X[:t+1]. Nothing else is visible to it.

It does not import df_operators or numpy, and shares no window, padding or
recursion helper with production: windows are plain list slices, medians come
from sorted(), recursions replay the prefix from its first row. Fitted
parameters (Kalman variances, taps, Butterworth coefficients, Hampel
fallback scales, Haar thresholds, frozen seasonal means) are read as data.

Arithmetic order. Sums accumulate from the oldest sample, FIR terms from the
oldest tap, Haar pairs as (newer + older) * 0.5 (IEEE addition is commutative,
so the bits do not depend on the operand order), the expanding seasonal centre
as a left-to-right sum over phases divided by the period. Production uses the
same orders, so equality is expected to be bitwise for every licensed kind;
REFERENCE_TOLERANCE records that declaration per kind.
"""
from __future__ import annotations

import math

NAN = float("nan")
BITWISE = "BITWISE"

REFERENCE_TOLERANCE = {
    "identity": (BITWISE, "copy of the last sample"),
    "ewma": (BITWISE, "same scalar recursion a*x + (1-a)*level"),
    "local_level_kalman": (BITWISE, "same scalar recursion and predict-only step"),
    "local_linear_trend_kalman": (BITWISE, "same scalar recursion and predict-only step"),
    "trailing_mean": (BITWISE, "sum accumulated oldest-first, then divided by the window"),
    "trailing_median": (BITWISE, "odd window: the middle order statistic is an input value"),
    "trailing_hampel": (BITWISE, "odd-window medians are input values; same comparison"),
    "fir_sinc_lowpass": (BITWISE, "terms accumulated oldest-first with the fitted taps"),
    "butterworth2_lowpass": (BITWISE, "same transposed direct-form II recursion"),
    "trailing_haar_threshold": (BITWISE, "pair means by powers of two; same soft threshold"),
    "causal_decomposition": (BITWISE, "same trend/seasonal recursion; phase centre summed left to right"),
}
TOLERANCE_WHEN_NOT_BITWISE = "abs(a - b) <= 1e-12 * max(1, abs(a), abs(b))"


def _isnan(v: float) -> bool:
    return v != v


def warmup(kind: str, params: dict, fit_mode: str) -> int:
    if kind == "identity":
        return 0
    if kind in ("trailing_mean", "trailing_median", "trailing_hampel"):
        return params["window"] - 1
    if kind == "fir_sinc_lowpass":
        return params["taps"] - 1
    if kind == "trailing_haar_threshold":
        return 2 ** params["levels"] - 1
    if kind == "causal_decomposition" and fit_mode == "EXPANDING_PREFIX":
        return params["period"]
    return 0


def _window_length(kind: str, params: dict) -> int:
    return warmup(kind, params, "") + 1


def _median_odd(vals: list) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def _soft(d: float, lam: float) -> float:
    m = abs(d) - lam
    if m < 0.0:
        m = 0.0
    sign = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    return sign * m


def _haar_at(col: list, levels: int, lam: list) -> tuple:
    """col: the last 2^levels samples, oldest first. Returns (y, per-level)."""
    L = len(col)
    t = L - 1
    memo = {}

    def c(j: int, tau: int) -> float:
        if j == 0:
            return col[tau]
        key = (j, tau)
        if key not in memo:
            s = 2 ** (j - 1)
            memo[key] = (c(j - 1, tau) + c(j - 1, tau - s)) * 0.5
        return memo[key]

    acc = 0.0
    per = []
    for j in range(1, levels + 1):
        d = c(j - 1, t) - c(j, t)
        sd = _soft(d, lam[j - 1])
        acc = acc + sd
        per.append({"detail": sd, "approx": c(j, t)})
    return c(levels, t) + acc, per


def _windowed(kind, params, fitted, prefix, t0):
    t = len(prefix) - 1
    V = len(prefix[-1])
    L = _window_length(kind, params)
    w = warmup(kind, params, "")
    values, reason, levels = [], [], []
    for j in range(V):
        xt = prefix[-1][j]
        window = [row[j] for row in prefix[max(0, t - L + 1):]]
        if _isnan(xt):
            reason.append("MISSING_INPUT")
        elif t < w:
            reason.append("WARMUP")
        elif any(_isnan(v) for v in window):
            reason.append("MISSING_INPUT")
        else:
            reason.append("AVAILABLE")
        if reason[-1] != "AVAILABLE":
            values.append(NAN)
            if kind == "trailing_haar_threshold":
                levels.append([{"detail": NAN, "approx": NAN} for _ in range(params["levels"])])
            continue
        if kind == "identity":
            y = xt
        elif kind == "trailing_mean":
            acc = window[0]
            for v in window[1:]:
                acc = acc + v
            y = acc / L
        elif kind == "trailing_median":
            y = _median_odd(window)
        elif kind == "trailing_hampel":
            med = _median_odd(window)
            mad = 1.4826 * _median_odd([abs(v - med) for v in window])
            scale = mad if mad > 0 else fitted["fallback_scale"][j]
            y = med if abs(xt - med) > params["k"] * scale else xt
        elif kind == "fir_sinc_lowpass":
            h = fitted["taps"]
            acc = h[L - 1] * window[0]
            for k in range(L - 2, -1, -1):
                acc = acc + h[k] * window[L - 1 - k]
            y = acc
        elif kind == "trailing_haar_threshold":
            y, per = _haar_at(window, params["levels"], fitted["thresholds"][j])
            levels.append(per)
        else:
            raise ValueError(kind)
        values.append(y)
    out = {"values": {"y": values}, "reason": reason}
    if kind == "trailing_haar_threshold":
        out["levels"] = levels
    return out


def _recursive_column(kind, params, fitted, fit_mode, series, t0, j):
    """Replay one column from the first visible row; outputs of the last row."""
    last = len(series) - 1
    names = ("denoised", "trend", "seasonal", "residual") if kind == "causal_decomposition" else ("y",)
    out = {n: NAN for n in names}
    reason = "AVAILABLE"
    if kind == "ewma":
        a, level = params["alpha"], NAN
        for i, x in enumerate(series):
            if not _isnan(x):
                level = x if _isnan(level) else a * x + (1.0 - a) * level
            if i == last:
                out["y"] = NAN if _isnan(x) else level
    elif kind == "local_level_kalman":
        c = fitted["per_column"][j]
        r, q = c["obs_var"], c["level_var"]
        level = var = NAN
        for i, x in enumerate(series):
            if _isnan(level):
                if _isnan(x):
                    pass
                else:
                    level, var = x, r
                    vp = var + q
                    k = vp / (vp + r)
                    level, var = level + k * (x - level), (1.0 - k) * vp
            elif _isnan(x):
                var = var + q
            else:
                vp = var + q
                k = vp / (vp + r)
                level, var = level + k * (x - level), (1.0 - k) * vp
            if i == last:
                out["y"] = NAN if _isnan(x) else level
    elif kind == "local_linear_trend_kalman":
        c = fitted["per_column"][j]
        r, ql, qs = c["obs_var"], c["level_var"], c["slope_var"]
        lvl = slope = p11 = p12 = p22 = NAN
        for i, x in enumerate(series):
            if _isnan(lvl) and not _isnan(x):
                lvl, slope, p11, p12, p22 = x, 0.0, r, 0.0, r
            if not _isnan(lvl):
                a1 = lvl + slope
                q11 = p11 + 2.0 * p12 + p22 + ql
                q12 = p12 + p22
                q22 = p22 + qs
                if _isnan(x):
                    lvl, p11, p12, p22 = a1, q11, q12, q22
                else:
                    fv = q11 + r
                    k1, k2 = q11 / fv, q12 / fv
                    v = x - a1
                    lvl, slope = a1 + k1 * v, slope + k2 * v
                    p11, p12, p22 = q11 - k1 * q11, q12 - k1 * q12, q22 - k2 * q12
            if i == last:
                out["y"] = NAN if _isnan(x) else lvl
    elif kind == "butterworth2_lowpass":
        b0, b1, b2 = fitted["b"]
        a1, a2 = fitted["a"][1], fitted["a"][2]
        zi0, zi1 = fitted["zi_unit"]
        z0 = z1 = NAN
        for i, x in enumerate(series):
            y = NAN
            if not _isnan(x):
                if _isnan(z0):
                    z0, z1 = zi0 * x, zi1 * x
                y = b0 * x + z0
                z0, z1 = b1 * x - a1 * y + z1, b2 * x - a2 * y
            if i == last:
                out["y"] = y
    elif kind == "causal_decomposition":
        P, aT, aS = params["period"], params["trend_alpha"], params["season_alpha"]
        expanding = fit_mode == "EXPANDING_PREFIX"
        seasonal = [NAN] * P if expanding else list(fitted["seasonal_init"][j])
        ready = not expanding
        trend = raw = NAN
        sums, counts = [0.0] * P, [0] * P
        for i, x in enumerate(series):
            ph = (t0 + i) % P
            row = {n: NAN for n in names}
            rsn = "MISSING_INPUT" if _isnan(x) else "AVAILABLE"
            if not _isnan(x) and ready:
                s_prev = seasonal[ph]
                des = x - s_prev
                tr_new = des if _isnan(trend) else trend + aT * (des - trend)
                row = {"denoised": tr_new + s_prev, "trend": tr_new, "seasonal": s_prev,
                       "residual": x - tr_new - s_prev}
                seasonal[ph] = s_prev + aS * ((x - tr_new) - s_prev)
                trend = tr_new
            elif not _isnan(x):
                rsn = "WARMUP"
                raw = x if _isnan(raw) else aT * x + (1.0 - aT) * raw
                sums[ph] = sums[ph] + (x - raw)
                counts[ph] += 1
                if all(n > 0 for n in counts):
                    means = [sums[p] / counts[p] for p in range(P)]
                    acc = means[0]
                    for m in means[1:]:
                        acc = acc + m
                    centre = acc / P
                    seasonal = [m - centre for m in means]
                    trend = raw
                    ready = True
            if i == last:
                out, reason = row, rsn
        return out, reason
    else:
        raise ValueError(kind)
    if _isnan(series[last]):
        reason = "MISSING_INPUT"
    return out, reason


def outputs_at(spec: dict, fitted: dict, fit_mode: str, prefix: list, t0: int = 0) -> dict:
    kind, params = spec["kind"], spec["params"]
    if kind in ("identity", "trailing_mean", "trailing_median", "trailing_hampel", "fir_sinc_lowpass",
                "trailing_haar_threshold"):
        return _windowed(kind, params, fitted, prefix, t0)
    V = len(prefix[-1])
    values, reason = {}, []
    for j in range(V):
        col = [row[j] for row in prefix]
        o, r = _recursive_column(kind, params, fitted, fit_mode, col, t0, j)
        for n, v in o.items():
            values.setdefault(n, []).append(v)
        reason.append(r)
    return {"values": values, "reason": reason}
