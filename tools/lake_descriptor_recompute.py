#!/usr/bin/env python3
"""C72 (order 2026-09-12): recompute the published lake descriptors from
source bytes, with formulas written HERE from the published contracts.

Independence is structural: this module imports nothing from
`characterize_lake`, `run_characterization` or any producer module, and
its test asserts that. Every formula below was written from the text of
the `descriptor_contract` the cube publishes for that descriptor.

Specificity is declared BEFORE any comparison and never adjusted after
seeing a result:

  FULLY_SPECIFIED   the contract text fixes the number; a recomputed
                    value that disagrees beyond tolerance is a
                    divergence.
  CONDITIONAL       fully specified only when the window holds no
                    non-finite value, because the contract does not say
                    whether NaN or inf enter the computation. With
                    non-finite values present it is UNDERSPECIFIED.
  UNDERSPECIFIED    the contract leaves a choice open (interpolation,
                    scale constant, degrees of freedom, window count,
                    frequency units). A value is still computed under a
                    DECLARED reading when one exists, and its agreement
                    is reported, but it is never counted as
                    independently verified.

Tolerance, also declared in advance: counts compare exactly; real values
agree when |a - b| <= ABS_TOL + REL_TOL * max(|a|, |b|).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

CONTRACT = "crispdm.independent_descriptor_recompute.v1"
ROW_CAP = 200_000
#: the published window rule is "first 70% of rows in file order" after
#: the row cap; integer arithmetic so no float rounding can move it.
WINDOW_NUMERATOR, WINDOW_DENOMINATOR = 7, 10
REL_TOL = 1e-9
ABS_TOL = 1e-12
MIN_FINITE = 3
ENTROPY_BINS = 64

FULLY = "FULLY_SPECIFIED"
CONDITIONAL = "CONDITIONAL"
UNDERSPECIFIED = "UNDERSPECIFIED"

#: descriptor -> (specificity, reason when not fully specified)
SPECIFICITY = {
    "n_observations": (FULLY, None),
    "missing_count": (FULLY, None),
    "non_finite_count": (FULLY, None),
    "missingness_fraction": (FULLY, None),
    "insufficient_finite_observations": (FULLY, None),
    "noise_estimate": (FULLY, None),
    "characterization_disposition": (FULLY, None),
    "constant": (FULLY, None),
    "autocorrelation_lag1": (FULLY, None),
    "autocorrelation_lag5": (FULLY, None),
    "autocorrelation_lag20": (FULLY, None),
    "mean": (CONDITIONAL, "the contract does not say whether non-finite "
                          "values are excluded"),
    "std": (CONDITIONAL, "the contract does not say whether non-finite "
                         "values are excluded"),
    "min": (CONDITIONAL, "non-finite handling unstated"),
    "max": (CONDITIONAL, "non-finite handling unstated"),
    "median": (CONDITIONAL, "non-finite handling unstated"),
    "duplicate_count": (CONDITIONAL, "whether NaN entries count as one "
                                     "distinct value is unstated"),
    "compressed_length_ratio": (CONDITIONAL, "whether non-finite values "
                                             "enter the compressed bytes "
                                             "is unstated"),
    "discrete_entropy_bits": (CONDITIONAL, "non-finite handling unstated"),
    "p01": (UNDERSPECIFIED, "percentile interpolation method unstated"),
    "p99": (UNDERSPECIFIED, "percentile interpolation method unstated"),
    "iqr": (UNDERSPECIFIED, "percentile interpolation method unstated"),
    "mad": (UNDERSPECIFIED, "whether the consistency constant 1.4826 is "
                            "applied is unstated"),
    "difference_to_level_dispersion": (UNDERSPECIFIED,
                                       "degrees of freedom of both "
                                       "standard deviations unstated"),
    "spectral_centroid": (UNDERSPECIFIED, "frequency unit, amplitude "
                                          "definition and treatment of "
                                          "the zero frequency unstated"),
    "window_mean_dispersion": (UNDERSPECIFIED, "the number of equal "
                                               "windows is unstated"),
}

#: readings declared for UNDERSPECIFIED descriptors that admit one. The
#: two with no defensible reading are not computed at all.
DECLARED_READINGS = {
    "p01": "numpy linear interpolation over finite values",
    "p99": "numpy linear interpolation over finite values",
    "iqr": "numpy linear interpolation over finite values",
    "mad": "unscaled median of absolute deviations from the median",
    "difference_to_level_dispersion": "ddof=1 for both standard deviations",
}

COUNT_DESCRIPTORS = frozenset({"n_observations", "missing_count",
                               "non_finite_count", "duplicate_count"})


def window_rows(rows_in_file: int) -> dict:
    rows_total = min(int(rows_in_file), ROW_CAP)
    return {"capped": int(rows_in_file) > ROW_CAP,
            "row_cap": ROW_CAP,
            "rows_total": rows_total,
            "rows_used": rows_total * WINDOW_NUMERATOR // WINDOW_DENOMINATOR,
            "rule": "first 70% of rows in file order"}


def specificity(descriptor: str, non_finite: int) -> tuple[str, str | None]:
    if descriptor not in SPECIFICITY:
        return UNDERSPECIFIED, "no contract known to this verifier"
    level, why = SPECIFICITY[descriptor]
    if level == CONDITIONAL:
        return (FULLY, None) if non_finite == 0 else (UNDERSPECIFIED, why)
    return level, why


def agree(descriptor: str, published, recomputed) -> bool:
    if published is None or recomputed is None:
        return published is None and recomputed is None
    if descriptor in COUNT_DESCRIPTORS:
        return float(published) == float(recomputed)
    a, b = float(published), float(recomputed)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))


def _pearson(x: np.ndarray, y: np.ndarray):
    if len(x) < MIN_FINITE:
        return None
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _autocorrelation(w: np.ndarray, lag: int):
    if len(w) <= lag:
        return None
    a, b = w[:-lag], w[lag:]
    keep = np.isfinite(a) & np.isfinite(b)
    return _pearson(a[keep], b[keep])


def recompute(window: np.ndarray) -> dict:
    """Every descriptor this verifier can compute, as
    {descriptor: (value or None, identifiable)}."""
    w = np.asarray(window, dtype=np.float64)
    finite_mask = np.isfinite(w)
    f = w[finite_mask]
    n = len(w)
    out: dict[str, tuple] = {
        "n_observations": (float(n), True),
        "missing_count": (float(np.isnan(w).sum()), True),
        "non_finite_count": (float((~finite_mask).sum()), True),
        "missingness_fraction": ((1.0 - len(f) / n) if n else None, n > 0),
        "duplicate_count": (float(n - len(np.unique(w))), True),
        "noise_estimate": (None, False),
    }
    if len(f) < MIN_FINITE:
        out["insufficient_finite_observations"] = (None, False)
        return out
    out["constant"] = (1.0 if f.min() == f.max() else 0.0, True)
    out["mean"] = (float(f.mean()), True)
    out["std"] = (float(f.std(ddof=1)), True)
    out["min"] = (float(f.min()), True)
    out["max"] = (float(f.max()), True)
    out["median"] = (float(np.median(f)), True)
    p = np.percentile(f, [1, 25, 75, 99])
    out["p01"] = (float(p[0]), True)
    out["p99"] = (float(p[3]), True)
    out["iqr"] = (float(p[2] - p[1]), True)
    out["mad"] = (float(np.median(np.abs(f - np.median(f)))), True)
    for lag in (1, 5, 20):
        v = _autocorrelation(w, lag)
        out[f"autocorrelation_lag{lag}"] = (v, v is not None)
    raw = f.astype("<f8").tobytes()
    out["compressed_length_ratio"] = (len(zlib.compress(raw, 9)) / len(raw),
                                      True)
    if f.min() == f.max():
        out["discrete_entropy_bits"] = (0.0, True)
    else:
        counts, _ = np.histogram(f, bins=ENTROPY_BINS,
                                 range=(f.min(), f.max()))
        prob = counts[counts > 0] / counts.sum()
        out["discrete_entropy_bits"] = (float(-(prob * np.log2(prob)).sum()),
                                        True)
    a, b = w[:-1], w[1:]
    keep = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[keep]
    level_sd = f.std(ddof=1)
    if len(d) >= 2 and level_sd > 0:
        out["difference_to_level_dispersion"] = (float(d.std(ddof=1)
                                                       / level_sd), True)
    else:
        out["difference_to_level_dispersion"] = (None, False)
    return out
