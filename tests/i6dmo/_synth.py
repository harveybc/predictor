"""Synthetic I6-DMO data fixtures (no real market data, no TEST year).

The calendar mimics the SHAPE of the assumed FX calendar used by the real
bundle (open Sunday 17:00 New York, close Friday 17:00 New York, Thursday
00:00 New York entry cutoff) so DST weeks are exercised. It is a fixture,
not a broker calendar. Bars use the UTC bar-END clock: one bar per whole hour
in (open, close].
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
REPO = Path(__file__).resolve().parents[2]
DESIGN_PATH = REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json"


def load_design() -> dict:
    return json.loads(DESIGN_PATH.read_text())


def _ny(day, hour):
    return datetime(day.year, day.month, day.day, hour, tzinfo=NY).astimezone(UTC)


def calendar_weeks(first_sunday: datetime, last_sunday: datetime) -> list:
    """Weekly entries between two Sundays (inclusive)."""
    weeks = []
    day = first_sunday.date()
    while day <= last_sunday.date():
        weeks.append({
            "week_id": day.isoformat(),
            "market_open": _ny(day, 17).isoformat(),
            "entry_cutoff": _ny(day + timedelta(days=4), 0).isoformat(),
            "market_close": _ny(day + timedelta(days=5), 17).isoformat(),
        })
        day += timedelta(days=7)
    return weeks


def calendar_config(weeks) -> dict:
    return {"entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12,
            "market_calendar": {"calendar_id": "SYNTH_FX_NY17_FIXTURE", "version": "synth-v1",
                                "weeks": weeks}}


def bar_ends(weeks) -> np.ndarray:
    out = []
    for w in weeks:
        o = int(datetime.fromisoformat(w["market_open"]).timestamp())
        c = int(datetime.fromisoformat(w["market_close"]).timestamp())
        out.extend(range(o + 3600, c + 1, 3600))
    return np.asarray(out, dtype=np.int64)


def make_input(directory: Path, *, first_sunday=datetime(2019, 12, 1, tzinfo=UTC),
               last_sunday=datetime(2024, 12, 29, tzinfo=UTC), data_end=None,
               missing=(), nonfinite=(), features=None, seed=7) -> Path:
    """Write INPUT.npz + CLOCK.json + CALENDAR.json into ``directory``.

    ``missing``: UTC epoch seconds of bars to drop; ``nonfinite``: (ts, column)
    pairs set to NaN. ``data_end`` (exclusive epoch seconds) truncates bars,
    default 2025-01-01 so no TEST bar exists.
    """
    directory.mkdir(parents=True, exist_ok=True)
    features = list(features or MANIFEST_NO_STLFSI4)
    weeks = calendar_weeks(first_sunday, last_sunday)
    ts = bar_ends(weeks)
    end = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()) if data_end is None else data_end
    ts = ts[ts < end]
    if missing:
        ts = ts[~np.isin(ts, np.asarray(sorted(missing), dtype=np.int64))]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(ts), len(features)))
    price = 1.10 + np.cumsum(rng.standard_normal(len(ts)) * 1e-3)
    index = {int(v): i for i, v in enumerate(ts)}
    for stamp, col in nonfinite:
        x[index[int(stamp)], col] = np.nan
    np.savez(directory / "INPUT.npz", ts=ts, x=x, price=price)
    digest = hashlib.sha256((directory / "INPUT.npz").read_bytes()).hexdigest()
    (directory / "CLOCK.json").write_text(json.dumps({
        "schema": "i6n.source_clock.v1", "status": "RESOLVED", "features": features,
        "input_sha256": digest, "timestamp_semantics": "UTC_BAR_END",
        "availability_policy": "AT_TIMESTAMP", "data_kind": "synthetic_fixture"}, indent=1))
    (directory / "CALENDAR.json").write_text(json.dumps(calendar_config(weeks), indent=1))
    return directory


def epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp())


def rewrite_input(directory: Path, *, ts=None, x=None, price=None) -> None:
    """Rewrite INPUT.npz in place (keeps CLOCK.json input_sha256 consistent)."""
    with np.load(directory / "INPUT.npz") as z:
        arrays = {k: z[k] for k in z.files}
    for key, value in (("ts", ts), ("x", x), ("price", price)):
        if value is not None:
            arrays[key] = value
    np.savez(directory / "INPUT.npz", **arrays)
    clock = json.loads((directory / "CLOCK.json").read_text())
    clock["input_sha256"] = hashlib.sha256((directory / "INPUT.npz").read_bytes()).hexdigest()
    (directory / "CLOCK.json").write_text(json.dumps(clock, indent=1))
