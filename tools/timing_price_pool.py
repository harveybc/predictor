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
# UTC session blocks (ASSUMED, not a broker calendar): Asia 00-07, Europe 07-13, US 13-21, off 21-24
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
