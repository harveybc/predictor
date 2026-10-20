"""One shared business-eligibility function for the timing selection (plan audit v3, V3-01).

Every stage that uses an origin (target labels, the univariate screen and the phase-3 engines, the causal
stage, classifier fitting and early stopping, calibration, thresholds and strategy scoring) uses the SAME
mask computed here once by the driver and written with a digest (``ELIGIBILITY.parquet``).  The evaluator
consumes that file and refuses a digest mismatch; nothing recomputes its own calendar.

Clock: the admitted source's UTC bar START; a bar is complete (and its values available) at start + 1 h.
Sessions are the complements of the admitted ``CLOSURE_INTERVALS.json`` ([start, stop) UTC, weekends and
holidays, including short sessions).  For origin t (bar start s_t, decision at s_t + 1 h, executable fill at
the next bar's open):

  BEFORE_REOPEN_12H        decision time < session open + reopen_hours (no entry in the first 12 h)
  WINDOW_CROSSES_CLOSURE   a bar of (t, t+k] lies in another session (the window spans a closure)
  AFTER_FORCE_FLAT_12H     the label window's end (bar t+k complete) > session close - preclose_hours
  NO_FUTURE_BARS           fewer than k observed bars after t
  INSIDE_CLOSURE           the origin bar itself falls inside a declared closure (a source defect)
  then per target: the label ledger's status (CENSORED_*, PURGED_*) when it is not LABELLED

The first matching reason in that order is the row's primary reason; every flag is also kept.  Historical
INPUT bars are never removed: features at t may use bars from earlier sessions (the prior week included).
k counts observed bars.  The 4/day cap of this timing lane is a strategy rule applied in scoring, declared
separately from the SAC 1/day contract; it is not part of eligibility.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

BAR = pd.Timedelta(hours=1)
REASONS = ("ELIGIBLE", "INSIDE_CLOSURE", "BEFORE_REOPEN_12H", "WINDOW_CROSSES_CLOSURE", "AFTER_FORCE_FLAT_12H",
           "NO_FUTURE_BARS", "LABEL_NOT_USABLE")
SCHEMA = "timing_selection.eligibility.v1"


class EligibilityError(ValueError):
    pass


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def parse_closures(intervals) -> pd.DataFrame:
    """[[start, stop), ...] UTC -> sorted, non-overlapping frame; refuses malformed or overlapping intervals."""
    rows = sorted((_utc(a), _utc(b)) for a, b in intervals)
    for a, b in rows:
        if not a < b:
            raise EligibilityError(f"CLOSURE_MALFORMED: {a} !< {b}")
    for (a0, b0), (a1, b1) in zip(rows, rows[1:]):
        if a1 < b0:
            raise EligibilityError(f"CLOSURE_OVERLAP: {a0}-{b0} and {a1}-{b1}")
    return pd.DataFrame(rows, columns=["start", "stop"])


def sessions(index: pd.DatetimeIndex, closures: pd.DataFrame):
    """Per bar: session id, session open and session close (the closures that bound it); inside-closure flag."""
    v = index.asi8
    lo = closures["start"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    hi = closures["stop"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    pos = np.searchsorted(lo, v, side="right") - 1                   # last closure starting at or before the bar
    inside = (pos >= 0) & (v < hi[np.maximum(pos, 0)])
    # session k = between closure k-1 end and closure k start; for a bar after closure pos: open = hi[pos],
    # close = lo[pos+1]
    sess = pos + 1
    open_ns = np.where(pos >= 0, hi[np.maximum(pos, 0)], np.iinfo(np.int64).min)
    close_ns = np.where(pos + 1 < len(lo), lo[np.minimum(pos + 1, len(lo) - 1)], np.iinfo(np.int64).max)
    return sess, open_ns, close_ns, inside


def eligibility(index: pd.DatetimeIndex, closures, *, k_bars: int, reopen_hours: float = 12.0,
                preclose_hours: float = 12.0, label_status: dict | None = None) -> pd.DataFrame:
    """The shared mask.  ``index`` = UTC bar starts (observed bars); ``label_status`` = {target: status array}
    from the label ledger (LABELLED or a typed reason)."""
    index = pd.DatetimeIndex(index)
    if index.tz is None or str(index.tz) != "UTC":
        raise EligibilityError("INDEX_NOT_UTC: eligibility needs UTC bar starts")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise EligibilityError("INDEX_NOT_STRICTLY_INCREASING")
    cl = closures if isinstance(closures, pd.DataFrame) else parse_closures(closures)
    n, k = len(index), int(k_bars)
    sess, open_ns, close_ns, inside = sessions(index, cl)
    v = index.asi8
    decision = v + BAR.value
    reopen = (decision < open_ns + int(reopen_hours * 3.6e12))
    t = np.arange(n)
    end_pos = t + k
    no_future = end_pos >= n
    ep = np.minimum(end_pos, n - 1)
    window_end = v[ep] + BAR.value                                    # bar t+k complete
    after_flat = window_end > close_ns - int(preclose_hours * 3.6e12)
    crosses = np.zeros(n, bool)
    for j in range(1, k + 1):
        jj = np.minimum(t + j, n - 1)
        crosses |= (sess[jj] != sess) | inside[jj]
    crosses &= ~no_future
    flags = {"INSIDE_CLOSURE": inside, "BEFORE_REOPEN_12H": reopen, "AFTER_FORCE_FLAT_12H": after_flat & ~no_future,
             "WINDOW_CROSSES_CLOSURE": crosses, "NO_FUTURE_BARS": no_future}
    base = np.full(n, "ELIGIBLE", object)
    for name in REASONS[1:6]:
        base[(base == "ELIGIBLE") & flags[name]] = name
    out = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "timestamp": index, "session_id": sess,
                        "session_open": pd.to_datetime(np.where(open_ns == np.iinfo(np.int64).min, 0, open_ns), utc=True),
                        "session_close": pd.to_datetime(np.where(close_ns == np.iinfo(np.int64).max, 0, close_ns), utc=True),
                        "business_reason": base, "business_eligible": base == "ELIGIBLE"})
    for name, f in flags.items():
        out[f"flag_{name.lower()}"] = f
    for tgt, st in (label_status or {}).items():
        st = np.asarray(st, object)
        r = base.copy()
        r[(r == "ELIGIBLE") & (st != "LABELLED")] = "LABEL_NOT_USABLE"
        out[f"{tgt}_reason"] = r
        out[f"{tgt}_eligible"] = r == "ELIGIBLE"
    return out


def counts(mask: pd.DataFrame, periods: dict, column: str) -> dict:
    """Origins by period x reason for one reason column (every origin accounted for)."""
    ts = pd.DatetimeIndex(mask["timestamp"])
    res = {}
    for name, (a, b) in periods.items():
        m = (ts >= _utc(a)) & (ts < _utc(b))
        vc = mask.loc[m, column].value_counts()
        res[name] = {"origins": int(m.sum()), **{r: int(vc.get(r, 0)) for r in REASONS}}
        if sum(v for r, v in res[name].items() if r != "origins") != res[name]["origins"]:
            raise EligibilityError("ORIGINS_UNACCOUNTED")
    return res


def digest(mask: pd.DataFrame) -> str:
    cols = [c for c in mask.columns if c.endswith("_eligible") or c.endswith("_reason") or c == "business_reason"]
    body = {"schema": SCHEMA, "row_ids": mask["row_id"].astype(int).tolist(),
            "ts": pd.DatetimeIndex(mask["timestamp"]).asi8.tolist(), **{c: mask[c].astype(str).tolist() for c in sorted(cols)}}
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
