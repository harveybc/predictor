"""Declared multi-scale PRICE candidate pool, computed causally from hourly OHLC, plus the availability /
causality gate and the within-day-variance report used by the timing selection.

Every value at bar t is a function of bars <= t only (prefix invariance is a test).  Calendar encodings are
KNOWN_IN_ADVANCE; everything else is OBSERVED at the bar label.  No centred, two-sided or decomposition
filter is produced or admitted (the gate refuses by name, the same rule as causal-inference's
event_impact.FORBIDDEN_TARGET_PATTERN).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 4, 6, 12, 24, 48)
ATR_SHORT, ATR_LONG = 14, 120
FORBIDDEN_TRANSFORM = re.compile(
    r"(stl|loess|lowess|centred|centered|two[_-]?sided|seasonal|trend[_-]?comp|resid|smooth|savgol|hp[_-]?filter)",
    re.IGNORECASE)
# Session blocks on the LAKE BAR CLOCK AS LABELLED (vendor clock; its UTC mapping is under repair and is NOT
# applied here, no hand shift): hours 00-07, 07-13, 13-21 of the label.  Names are conventions only; they are
# not certified UTC sessions.  The same caveat applies to every cal.* hour / weekday encoding.
SESSIONS = {"cal.session_asia": (0, 7), "cal.session_europe": (7, 13), "cal.session_us": (13, 21)}


class AvailabilityError(ValueError):
    """A typed refusal; the message starts with its code."""


def _true_range(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    return np.nanmax(np.column_stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=1)


def price_pool(bars: pd.DataFrame) -> pd.DataFrame:
    """The pool on the bar grid of ``bars`` (OPEN/HIGH/LOW/CLOSE, UTC index)."""
    h, l, c = (bars[k].to_numpy(float) for k in ("HIGH", "LOW", "CLOSE"))
    H, L, C = pd.Series(h), pd.Series(l), pd.Series(c)
    out = {}
    r1 = np.log(C / C.shift(1))
    for k in HORIZONS:
        out[f"px.logret_{k}"] = np.log(C / C.shift(k))
        hi, lo = H.rolling(k).max(), L.rolling(k).min()
        out[f"px.range_{k}"] = np.log(hi / lo)
        span = (hi - lo).replace(0.0, np.nan)
        out[f"px.close_loc_{k}"] = (C - lo) / span
        path = r1.abs().rolling(k).sum().replace(0.0, np.nan)
        out[f"px.er_{k}"] = (np.log(C / C.shift(k))).abs() / path
        out[f"px.rv_{k}"] = np.sqrt((r1 ** 2).rolling(k).sum())
    tr = pd.Series(_true_range(h, l, c))
    atr_s = tr.rolling(ATR_SHORT).mean()
    atr_l = tr.rolling(ATR_LONG).mean()
    out[f"px.atr_{ATR_SHORT}"] = atr_s
    out[f"px.atr_ratio_{ATR_SHORT}_{ATR_LONG}"] = atr_s / atr_l.replace(0.0, np.nan)
    out["px.range_ratio_1_24"] = out["px.range_1"] / out["px.range_24"].replace(0.0, np.nan)
    out["px.bar_body"] = np.log(C / pd.Series(bars["OPEN"].to_numpy(float)))
    idx = bars.index
    hour = np.asarray(idx.hour, float)
    dow = np.asarray(idx.dayofweek, float)
    out["cal.hod_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal.hod_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal.dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["cal.dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for name, (a, b) in SESSIONS.items():
        out[name] = ((hour >= a) & (hour < b)).astype(float)
    out["px.elapsed_hours"] = np.r_[np.nan, np.diff(idx.asi8) / 3.6e12]
    frame = pd.DataFrame(out)
    frame.index = idx
    return frame


def pool_metadata(frame: pd.DataFrame) -> list[dict]:
    meta = []
    for col in frame.columns:
        known = col.startswith("cal.")
        meta.append({"feature_id": col, "family": "calendar_known" if known else "price_pool",
                     "source": "derived:clock" if known else "derived:lake_ohlc_h1",
                     "transform": "trailing-window function of bars <= t" if not known else "bar label encoding",
                     "unit": "dimensionless", "clock": "KNOWN_IN_ADVANCE" if known else "OBSERVED",
                     "availability_time": "known in advance" if known else "timestamp",
                     "support_h": 0 if known else 1, "train_coverage": float(frame[col].notna().mean()),
                     "source_bytes": int(frame[col].notna().sum()) * 8})
    return meta


def availability_gate(features: list[dict]) -> list[str]:
    """Admit only features available at the decision bar with no two-sided transform; return the admitted ids."""
    admitted = []
    for f in features:
        fid = f.get("feature_id")
        text = " ".join(str(f.get(k, "")) for k in ("feature_id", "transform"))
        m = FORBIDDEN_TRANSFORM.search(text)
        if m:
            raise AvailabilityError(f"FORBIDDEN_TRANSFORM: {fid} names a two-sided filter ({m.group(0)!r})")
        avail = str(f.get("availability_time", ""))
        if avail not in ("timestamp", "known in advance", "KNOWN_IN_ADVANCE"):
            raise AvailabilityError(f"FUTURE_AVAILABILITY: {fid} declares availability_time={avail!r}; only "
                                    f"'timestamp' (<= t) or 'known in advance' is admissible")
        admitted.append(fid)
    return admitted


UNITS = {"px.logret_": "market bars", "px.range_": "market bars", "px.close_loc_": "market bars", "px.er_": "market bars",
         "px.rv_": "market bars", "px.atr_": "market bars", "px.range_ratio": "market bars", "px.bar_body": "one bar",
         "px.elapsed_hours": "elapsed hours", "cal.": "label clock"}


def window_unit(feature_id: str) -> str:
    """Declared window unit of a pool feature: rolling windows count OBSERVED market bars, never elapsed hours."""
    for pre, unit in UNITS.items():
        if feature_id.startswith(pre):
            return unit
    return "declared by producer"


def quality_profile(frame: pd.DataFrame) -> dict:
    """Phase-1 style quality per column on the bar grid: coverage, duplicate timestamps, finite share, warm-up,
    extrema (genuine values kept; no clipping), constants, and declared availability/unit."""
    idx = frame.index
    dup = int(pd.Index(idx).duplicated().sum())
    out = {"rows": int(len(frame)), "duplicate_timestamps": dup, "columns": {}}
    for col in frame.columns:
        v = frame[col].to_numpy(float)
        fin = np.isfinite(v)
        first = int(np.argmax(fin)) if fin.any() else None
        tail = v[first:] if first is not None else v[:0]
        interior_nonfinite = int((~np.isfinite(tail)).sum()) if first is not None else None
        fv = v[fin]
        out["columns"][col] = {
            "finite_share": float(fin.mean()) if len(v) else 0.0,
            "warmup_rows": first, "warmup_until": str(idx[first]) if first is not None else None,
            "interior_nonfinite": interior_nonfinite,
            "min": float(fv.min()) if fv.size else None, "max": float(fv.max()) if fv.size else None,
            "p001": float(np.quantile(fv, 0.001)) if fv.size else None, "p999": float(np.quantile(fv, 0.999)) if fv.size else None,
            "distinct_values": int(np.unique(fv).size) if fv.size else 0,
            "constant": bool(fv.size and np.nanstd(fv) == 0), "window_unit": window_unit(col)}
    return out


def perturbation_check(fn, bars: pd.DataFrame, cut: int, *, seed: int = 0, rtol: float = 0.0) -> dict:
    """Prefix invariance + future perturbation for a generator ``fn(bars) -> DataFrame`` on the same index:
    (a) fn(bars[:cut]) equals fn(bars)[:cut]; (b) replacing bars[cut:] by a seeded random walk leaves
    fn(...)[:cut] unchanged.  Exact equality (rtol 0) unless declared."""
    full = fn(bars)
    pre = fn(bars.iloc[:cut])
    rng = np.random.default_rng(seed)
    pert = bars.copy()
    m = len(bars) - cut
    base = float(bars.CLOSE.iloc[cut - 1])
    c = base * np.exp(rng.normal(0, 8e-4, m).cumsum())
    w = np.abs(rng.normal(0, 4e-4, m)) * base
    pert.iloc[cut:, pert.columns.get_loc("CLOSE")] = c
    pert.iloc[cut:, pert.columns.get_loc("OPEN")] = np.r_[base, c[:-1]]
    pert.iloc[cut:, pert.columns.get_loc("HIGH")] = np.maximum(c, np.r_[base, c[:-1]]) + w
    pert.iloc[cut:, pert.columns.get_loc("LOW")] = np.minimum(c, np.r_[base, c[:-1]]) - w
    perturbed = fn(pert)

    def diff(a, b):
        a, b = a.to_numpy(float), b.to_numpy(float)
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= rtol * np.maximum(np.abs(a), 1e-300)) | (a == b)
        bad = ~same
        return {"cells_differing": int(bad.sum()), "columns_differing": sorted(set(np.array(list(range(a.shape[1])))[bad.any(axis=0)].tolist()))}

    a = diff(full.iloc[:cut], pre)
    b = diff(full.iloc[:cut], perturbed.iloc[:cut])
    cols = list(full.columns)
    return {"cut_position": int(cut), "prefix": {"cells_differing": a["cells_differing"], "columns": [cols[i] for i in a["columns_differing"]]},
            "future_perturbation": {"cells_differing": b["cells_differing"], "columns": [cols[i] for i in b["columns_differing"]]},
            "passed": a["cells_differing"] == 0 and b["cells_differing"] == 0}


def within_day_variance(frame: pd.DataFrame, *, rel_tol: float = 1e-9) -> dict:
    """Per feature: the mean within-day variance relative to the total variance, and a constant-within-day flag
    (the 14 daily macro columns of the old manifest were constant across the hours of a day)."""
    day = frame.index.tz_convert("UTC").floor("D") if frame.index.tz is not None else frame.index.floor("D")
    out = {}
    for col in frame.columns:
        s = frame[col].astype(float)
        total = float(np.nanvar(s.to_numpy()))
        within = float(s.groupby(day).var(ddof=0).mean()) if total > 0 else 0.0
        ratio = within / total if total > 0 else 0.0
        out[col] = {"within_day_variance_ratio": ratio, "constant_within_day": bool(total > 0 and ratio <= rel_tol)}
    return out
