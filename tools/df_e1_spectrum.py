#!/usr/bin/env python3
"""RP28: train-only spectra with a DECLARED resolution. Welch with a declared segment length gives a
frequency grid whose slowest non-zero bin is nperseg * delta_t: any band beyond that support has no bin
and is NO_RESUELTO, never a measured zero. Segment, resolution, observable range, normalisation, weighting,
column rule and the treatment of missing values are recorded with every result; per-variable and aggregate
powers are different grains and are reported apart; sensitivity to the train support is measured by
recomputing on a shorter suffix of train. Batch characterisation of train; never an online filter.
"""

from __future__ import annotations

import math

import numpy as np


def welch_declared(x: np.ndarray, delta_seconds: float, nperseg: int, missing: str = "mean_impute_and_count") -> dict:
    """One column, train block only. Missing values: mean-imputed (train mean) and counted; the count is part of the result."""
    from scipy.signal import welch
    v = np.asarray(x, dtype=float)
    n_missing = int((~np.isfinite(v)).sum())
    if v.size < 16 or np.isfinite(v).sum() < 16:
        return {"state": "NO_RESUELTO", "why": "fewer than 16 finite samples"}
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    v = v - v.mean()
    nperseg = int(min(nperseg, v.size))
    f, pxx = welch(v, fs=1.0 / delta_seconds, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density")
    pxx = pxx.copy()
    pxx[0] = 0.0
    return {"state": "MEDIDO", "f": f, "pxx": pxx, "n_missing_imputed": n_missing, "n_samples": int(v.size), "nperseg": nperseg,
            "delta_f_hz": float(1.0 / (nperseg * delta_seconds)), "slowest_resolvable_period_seconds": float(nperseg * delta_seconds),
            "fastest_period_seconds": float(2 * delta_seconds), "declaration": {"window": "hann", "overlap": "50 %", "detrend": "constant (mean removed)",
                                                                                   "scaling": "density (V**2/Hz)", "dc_bin": "excluded", "segments": int(max(1, (v.size - nperseg) // (nperseg // 2) + 1))}}


def band_share(spec: dict, period_seconds: float, tol: float = 0.08) -> dict:
    """Share of total power in bins whose period is within tol of the target period; NO_RESUELTO when the
    period is outside the spectral support (slower than nperseg * delta or faster than 2 * delta)."""
    if spec.get("state") != "MEDIDO":
        return {"state": "NO_RESUELTO", "why": spec.get("why")}
    if period_seconds > spec["slowest_resolvable_period_seconds"] or period_seconds < spec["fastest_period_seconds"]:
        return {"state": "NO_RESUELTO", "why": f"period {period_seconds:.0f} s outside the spectral support [{spec['fastest_period_seconds']:.0f}, {spec['slowest_resolvable_period_seconds']:.0f}] s"}
    f = spec["f"]
    periods = np.where(f > 0, 1.0 / np.maximum(f, 1e-18), np.inf)
    sel = np.abs(periods - period_seconds) <= tol * period_seconds
    if not sel.any():
        return {"state": "NO_RESUELTO", "why": "no bin within the tolerance of that period (resolution too coarse there)"}
    tot = float(spec["pxx"].sum())
    return {"state": "MEDIDO", "share": float(spec["pxx"][sel].sum() / tot) if tot > 0 else None, "bins": int(sel.sum()),
            "period_seconds": period_seconds, "tolerance": tol}


def peaks(spec: dict, k: int = 6) -> list:
    if spec.get("state") != "MEDIDO":
        return []
    f, p = spec["f"], spec["pxx"]
    order = [int(i) for i in np.argsort(p)[::-1] if f[i] > 0][:k]
    tot = float(p.sum())
    return [{"period_seconds": float(1.0 / f[i]), "share": float(p[i] / tot) if tot > 0 else None} for i in order]


def per_variable_and_aggregate(X: np.ndarray, columns: list, delta_seconds: float, nperseg: int, bands_seconds: dict, column_rule: str) -> dict:
    """Two grains: per-variable spectra (each normalised by its own total power) and the aggregate of the
    per-variable normalised spectra (equal weight per variable, declared); the column rule is recorded."""
    per, agg = {}, None
    for j, c in enumerate(columns):
        s = welch_declared(X[:, j], delta_seconds, nperseg)
        entry = {"state": s["state"], "n_missing_imputed": s.get("n_missing_imputed"), "bands": {name: band_share(s, sec) for name, sec in bands_seconds.items()}, "peaks": peaks(s, 3)}
        per[c] = entry
        if s["state"] == "MEDIDO":
            norm = s["pxx"] / s["pxx"].sum() if s["pxx"].sum() > 0 else s["pxx"]
            agg = norm if agg is None else agg + norm
            ref = s
    out = {"columns_used": list(columns), "column_rule": column_rule, "per_variable": per, "aggregate": None}
    if agg is not None:
        aspec = dict(ref, pxx=agg / len([1 for v in per.values() if v["state"] == "MEDIDO"]))
        out["aggregate"] = {"weighting": "equal weight per variable after normalising each variable's spectrum to unit power (a scale change of one column does not change it)",
                            "bands": {name: band_share(aspec, sec) for name, sec in bands_seconds.items()}, "peaks": peaks(aspec, 6),
                            "resolution": {k: aspec[k] for k in ("nperseg", "delta_f_hz", "slowest_resolvable_period_seconds", "fastest_period_seconds", "declaration")}}
    return out
