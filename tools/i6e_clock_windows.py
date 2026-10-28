"""Origin-anchored sparse clock windows for the compact resolution experiment."""

import numpy as np


def clock_windows(timestamps, scaled_values, origins, spacing_hours, samples):
    """Return chronological values and market-presence masks, without imputation.

    Parameters
    ----------
    timestamps : array-like
        Strictly increasing observed decision timestamps in UTC seconds.
    scaled_values : array-like
        Finite rank-two values, already processed by TRAIN-only scaling and
        source-availability handling. This function fits no preprocessing.
    origins : array-like
        Indices of observed current bars, not arbitrary clock timestamps.
    spacing_hours, samples : int
        Positive spacing and sample count, including the observed origin.

    Returns
    -------
    values, observed : ndarray, ndarray
        Shapes (origins, samples, channels) and (origins, samples). A zero at
        an absent slot is a numeric placeholder, not a market quote. Models
        must consume the mask; dropping it invalidates the experiment.
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    values = np.asarray(scaled_values, dtype=np.float32)
    oi = np.asarray(origins)
    if (ts.ndim != 1 or not len(ts) or np.any(np.diff(ts) <= 0)
            or values.ndim != 2 or len(values) != len(ts)
            or not np.isfinite(values).all()):
        raise ValueError("invalid observed timestamps or scaled values")
    if (oi.ndim != 1 or oi.dtype.kind not in "iu" or np.any(oi < 0)
            or np.any(oi >= len(ts))):
        raise ValueError("origins must index observed bars")
    if (type(spacing_hours) is not int or spacing_hours <= 0
            or type(samples) is not int or samples < 2):
        raise ValueError("positive integer spacing and at least two samples required")
    required = ts[oi, None] - np.arange(samples - 1, -1, -1) * spacing_hours * 3600
    if np.any(required < ts[0]):
        raise ValueError("window crosses history boundary")
    indices = np.searchsorted(ts, required)
    observed = (indices < len(ts)) & (ts[np.minimum(indices, len(ts) - 1)] == required)
    result = np.zeros((*required.shape, values.shape[1]), dtype=np.float32)
    result[observed] = values[indices[observed]]
    return result, observed
