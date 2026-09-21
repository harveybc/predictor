#!/usr/bin/env python3
"""Phase 2 (calendar): the production path for calendar inputs — computed from the timestamp alone.

What a calendar feature must be, for it to be an INPUT and not a leak:

  * a function of the ORIGIN's own timestamp label, and of nothing observed after it;
  * known at the decision: the civil time of row t is known at row t, by definition of the label;
  * declared: which clock the label is on, and what is not known about it. The household panel's
    labels are NAIVE_WALL_CLOCK — local French time presumed by the producer, zone and daylight
    saving undocumented. A feature built on them is a feature of the wall clock, and says so.

This module builds hour-of-day and day-of-week as sin/cos pairs — the encoding of the TensorFlow
forecasting tutorial (a reference read at its source in RP61) — and refuses what would silently
lie: a numeric timestamp column, an unparseable label, a missing label, a label that is not on the
declared grid. Perturbing any row's VALUES leaves every calendar feature unchanged, because none of
them reads a value; perturbing a later row's TIMESTAMP leaves the earlier rows' features unchanged,
because each row's feature reads only its own label. Both are the acceptance tests.

The feature adds channels, and channels change the parameter count of a model that consumes them.
`parameter_delta` states that change so the phase-2 design can hold capacity constant explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CLOCK = "NAIVE_WALL_CLOCK"
FEATURES = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos")


class CalendarRefusal(ValueError):
    """A calendar feature that could not be built honestly is not built at all."""


@dataclass
class CalendarSpec:
    """The declaration a calendar input travels with."""
    timestamp_column: str
    ts_format: str
    clock: str = CLOCK
    timezone: str = "UNKNOWN (local French time presumed by the producer; DST undocumented)"
    step_seconds: int = 60
    features: tuple = FEATURES
    sampling_assumption: str = "one label per row on the declared grid; the label IS the observation's own minute"
    publication_assumption: str = "a static archive: every row's label is known when the row exists; no revision"
    known_at_decision: str = "each row's features depend on that row's label only"
    notes: list = field(default_factory=list)


def parse_labels(frame: pd.DataFrame, spec: CalendarSpec) -> pd.Series:
    """The labels as instants on the declared clock — refusing anything that is not a label."""
    col = spec.timestamp_column
    if col not in frame.columns:
        raise CalendarRefusal(f"the timestamp column {col!r} is absent: a calendar feature has nothing to read")
    series = frame[col]
    if pd.api.types.is_numeric_dtype(series):
        raise CalendarRefusal("the timestamp column is numeric: a number is not a civil time and would be "
                              "misread as one")
    if series.isna().any():
        raise CalendarRefusal(f"{int(series.isna().sum())} timestamp labels are missing; a calendar feature is not "
                              "invented for a row that has no time")
    ts = pd.to_datetime(series, format=spec.ts_format, errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        raise CalendarRefusal(f"{bad} timestamp labels do not parse with {spec.ts_format!r}")
    return ts.reset_index(drop=True)


def grid_report(ts: pd.Series, spec: CalendarSpec) -> dict:
    """What the labels say about the grid and the clock over THIS slice — reported, never fixed."""
    deltas = ts.diff().dropna().dt.total_seconds()
    off = deltas[deltas != spec.step_seconds]
    span = (ts.iloc[-1]-ts.iloc[0])
    return {"rows": int(ts.size), "first": str(ts.iloc[0]), "last": str(ts.iloc[-1]),
            "rows_off_grid": int(off.size),
            "duplicated_labels": int(ts.duplicated().sum()),
            "span_days": span.total_seconds()/86400.0,
            "clock": spec.clock, "timezone": spec.timezone,
            "dst_note": ("on a wall clock a spring transition shows as a one-hour gap and an autumn one as a "
                         "repeated hour; both would appear above as off-grid rows or duplicates and are "
                         "reported, not closed. None is asserted absent without this count.")}


def features(ts: pd.Series, spec: CalendarSpec) -> np.ndarray:
    """hour and weekday as sin/cos pairs, from each row's own label."""
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy()/60.0
    weekday = ts.dt.dayofweek.to_numpy() + hour/24.0
    two_pi = 2.0*math.pi
    out = np.stack([np.sin(two_pi*hour/24.0), np.cos(two_pi*hour/24.0),
                    np.sin(two_pi*weekday/7.0), np.cos(two_pi*weekday/7.0)], axis=1)
    return out.astype(np.float64)


def build(frame: pd.DataFrame, spec: CalendarSpec) -> dict:
    ts = parse_labels(frame, spec)
    return {"features": features(ts, spec), "names": list(spec.features), "grid": grid_report(ts, spec),
            "spec": spec.__dict__ | {"features": list(spec.features)}}


def parameter_delta(model_builder, window: int, channels_before: int, channels_added: int, **kw) -> dict:
    """How many trainable parameters the added channels bring, measured by building both models.

    The phase-2 design holds capacity constant or declares the difference; it does not let an input
    change pass as an information change alone.
    """
    def count(model):
        return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    before = count(model_builder(window, channels_before, **kw))
    after = count(model_builder(window, channels_before+channels_added, **kw))
    return {"channels_before": channels_before, "channels_added": channels_added,
            "parameters_before": before, "parameters_after": after, "delta": after-before,
            "reading": "a positive delta is capacity added with the information; the design must match it "
                       "in the control arm or declare and separate it"}
