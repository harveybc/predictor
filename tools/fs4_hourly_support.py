"""Causal elapsed-hour windows for FS4's hourly encoder grid and RAW controls."""

from __future__ import annotations

import numpy as np


HOUR = 3600


def hourly_windows(values, timestamps, origins, *, mean, sd, window_hours=24, lag_hours=0, min_timestamp=None):
    """Return (windows, retained origins) on an hourly grid, with value/mask channels per feature.

    A lag is elapsed time, never a count of dataset rows. Missing grid hours and nonfinite
    observations have value=0, mask=0 after TRAIN-fitted standardization. No slot after
    ``origin - lag_hours`` is read, and no window may cross ``min_timestamp``.
    """
    x = np.asarray(values, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64)
    idx = np.asarray(origins, dtype=np.int64)
    mu = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(sd, dtype=np.float64)
    if x.ndim != 2 or ts.ndim != 1 or x.shape[0] != ts.size or ts.size == 0:
        raise ValueError("SERIES_SHAPE_MISMATCH")
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= ts.size):
        raise ValueError("ORIGIN_INDEX_OUT_OF_BOUNDS")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("TIMESTAMPS_NOT_STRICTLY_INCREASING")
    if np.any((ts - ts[0]) % HOUR):
        raise ValueError("HOURLY_GRID_MISALIGNED")
    if mu.shape != (x.shape[1],) or scale.shape != mu.shape or not np.all(np.isfinite(mu)) \
            or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("TRAIN_STANDARDISER_INVALID")
    if type(window_hours) is not int or window_hours < 1 or type(lag_hours) is not int or lag_hours < 0:
        raise ValueError("WINDOW_OR_LAG_INVALID")
    earliest = ts[0] if min_timestamp is None else max(ts[0], int(min_timestamp))
    end = ts[idx] - lag_hours * HOUR
    keep = end - (window_hours - 1) * HOUR >= earliest
    retained = idx[keep]
    if retained.size == 0:
        return np.empty((0, window_hours, 2 * x.shape[1]), dtype=np.float32), retained
    slots = end[keep, None] - np.arange(window_hours - 1, -1, -1, dtype=np.int64)[None, :] * HOUR
    positions = np.searchsorted(ts, slots)
    safe = np.minimum(positions, ts.size - 1)
    present = (positions < ts.size) & (ts[safe] == slots)
    raw = x[safe]
    observed = present[:, :, None] & np.isfinite(raw)
    out = np.zeros((retained.size, window_hours, 2 * x.shape[1]), dtype=np.float32)
    out[:, :, 0::2] = np.where(observed, (raw - mu) / scale, 0).astype(np.float32)
    out[:, :, 1::2] = observed.astype(np.float32)
    return out, retained
