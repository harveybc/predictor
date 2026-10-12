"""Exact physical-hour labels, four-year weekly partitions and original prices."""

from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def load_input(path, expected, n_features=20):
    """Authenticate bytes before interpreting them; reject TEST and invalid clocks.

    ``n_features`` is the bound manifest length (20 for the legacy manifest).
    """
    if file_hash(path) != expected:
        raise ValueError("input digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        ts, x, price = (archive[k].copy() for k in ("ts", "x", "price"))
    if (ts.dtype.kind != "i" or ts.ndim != 1 or not len(ts)
            or np.any(np.diff(ts) <= 0) or np.any(ts % 3600)
            or ts[-1] >= epoch("2025-01-01T00:00:00Z")
            or x.shape != (len(ts), n_features) or price.shape != ts.shape
            or not np.isfinite(price).all() or np.any(price <= 0)):
        raise ValueError("invalid hourly price clock/features or forbidden TEST rows")
    return ts, x, price


def targets(ts, price, horizon):
    """Return log returns and exact endpoint indices; missing endpoints remain NaN."""
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("positive integer physical hours required")
    end = np.searchsorted(ts, ts + horizon*3600)
    valid = (end < len(ts)) & (ts[np.minimum(end, len(ts)-1)] == ts+horizon*3600)
    y = np.full(len(ts), np.nan)
    y[valid] = np.log(price[end[valid]]/price[valid])
    return y, np.where(valid, end, -1)


def partitions(ts, y, week, horizon, require_fit=True):
    """Purge label endpoints and require all physical input support inside TRAIN."""
    start, cut = epoch(week["fit_start"]), epoch(week["fit_cutoff"])
    inner = cut-28*86400
    valid = np.isfinite(y)
    support = ts-2*horizon*3600 >= start
    fit = np.flatnonzero(valid & support & (ts+horizon*3600 < inner))
    vi = np.flatnonzero(valid & support & (ts >= inner) & (ts+horizon*3600 < cut))
    si = np.flatnonzero(valid & (ts >= epoch(week["score_start"]))
                        & (ts < epoch(week["score_end"]))
                        & (ts+horizon*3600 < epoch("2025-01-01T00:00:00Z")))
    if require_fit and (not len(fit) or not len(vi)):
        raise ValueError("empty TRAIN or inner-validation partition")
    return fit, vi, si


def price_prediction(current, z, center, scale):
    """Undo target scaling in float64 BEFORE exponentiation and price scoring."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(current, np.float64)*np.exp(np.asarray(z, np.float64)*scale+center)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("nonfinite/invalid reconstructed prediction")
    return result


def metrics(current, future, prediction, horizon):
    """User skill convention: negative means lower error than that horizon's naive."""
    from tools.i6e_horizon_plan import compare_prices
    result = compare_prices(current, future, prediction, horizon_hours=horizon, naive_horizon_hours=horizon)
    result["skill_mae_conventional"] = result["skill_mae"]
    result["skill_mae"] = None if result["naive_mae"] == 0 else (result["mae"]-result["naive_mae"])/result["naive_mae"]
    result["skill_formula"] = "(MAE_h - naive_h) / naive_h"
    return result
