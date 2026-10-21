"""D1 event clock for EURUSD H1 (2026-10-21, literature-guided): hours since / to the daily FX fixes and cuts.

Krohn, Mueller & Whelan 2024 (JF 79(1)) document USD appreciation INTO the Tokyo and London fixes and reversal
after them; the New York 10:00 option cut is the market's other daily reference. Each event is a local wall-clock
time in its own zone, so its UTC time moves with that zone's DST (London and New York change on different dates;
Tokyo has no DST). All values are functions of the decision instant (bar start + 1 h) and the published fix
conventions only: KNOWN IN ADVANCE, no data read, no possible leakage.

  ev.tokyo_fix   09:55 Asia/Tokyo       (00:55 UTC all year)
  ev.london_fix  16:00 Europe/London    (WM/Reuters 4 pm fix; 15:00 UTC in BST, 16:00 UTC in GMT)
  ev.ny_cut      10:00 America/New_York (option expiry cut)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENTS = {"tokyo_fix": ("Asia/Tokyo", 9, 55), "london_fix": ("Europe/London", 16, 0), "ny_cut": ("America/New_York", 10, 0)}
HOUR = pd.Timedelta(hours=1)


def _since_to(decision_utc: pd.DatetimeIndex, tz: str, hh: int, mm: int):
    loc = decision_utc.tz_convert(tz)
    today = loc.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    today = pd.DatetimeIndex([t.tz_localize(None) for t in today]).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    last = today.where(loc >= today, today - pd.Timedelta(days=1))
    last = pd.DatetimeIndex([t.tz_localize(None) for t in last]).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    nxt = pd.DatetimeIndex([t.tz_localize(None) for t in (last + pd.Timedelta(days=1))]).tz_localize(tz, ambiguous="NaT",
                                                                                                    nonexistent="shift_forward")
    since = (decision_utc - last.tz_convert("UTC")) / HOUR
    to = (nxt.tz_convert("UTC") - decision_utc) / HOUR
    return np.asarray(since, float), np.asarray(to, float)


def event_clock(bar_start_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Hours since the last and to the next occurrence of each event, measured from the DECISION instant."""
    d = pd.DatetimeIndex(bar_start_utc).tz_convert("UTC") + HOUR
    out = {}
    for name, (tz, hh, mm) in EVENTS.items():
        s, t = _since_to(d, tz, hh, mm)
        out[f"ev.{name}.hours_since"] = s
        out[f"ev.{name}.hours_to"] = t
    return pd.DataFrame(out, index=bar_start_utc)


def event_meta(columns) -> list[dict]:
    return [{"feature_id": c, "family": "event_clock", "availability_time": "known in advance",
             "transform": "hours between the decision instant and a scheduled daily fix/cut in its own time zone (DST per tz database)",
             "unit": "hours"} for c in columns]
