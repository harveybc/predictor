"""Invented dense CPU fixture for lane A trajectory regressions.

Not historical data, not clock evidence, not cost evidence. A complete
four-year nominal calendar with fully observed hourly bars in three weeks:
week 10 (TRAIN), 207 (inner) and 210 (development evaluation week only;
the bridge never scores it). Optional DST-like shift: weeks >= ``dst_from``
open one UTC hour earlier, as a northern summer FX week does.
"""

from datetime import datetime, timedelta, timezone
import json

import numpy as np

from tools.i6m_data import file_hash
from tools.i6n_contracts import eligibility as c1
from tools.i6n_model import DEFAULT_CONFIG

START = datetime(2019, 12, 29, 22, tzinfo=timezone.utc)
PLAN_TRAINING = dict(max_epochs=50, patience=5, min_delta=1e-6, batch_size=64,
                     learning_rate=.001, weight_decay=0., huber_delta=1., seed=6102026,
                     head_weights=[1.])


def week_open(i, dst_from=None):
    shift = timedelta(hours=-1) if dst_from is not None and i >= dst_from else timedelta(0)
    return START + timedelta(days=7 * i) + shift


def write_json(path, value):
    path.write_text(json.dumps(value))
    return file_hash(path)


def build(tmp_path, *, horizon, samples, spacing, observed_weeks=(10, 207, 210),
          drop=(), keep=None, dst_from=None, n_weeks=211):
    """Return an authenticated endpoint base cell config (plan stopping).

    ``drop``: iterable of (week, hour_from_open) bars removed. ``keep``:
    optional predicate(week, hour) -> bool restricting observed bars.
    """
    weeks = []
    for i in range(n_weeks):
        opened = week_open(i, dst_from)
        weeks.append(dict(week_id=str(i), market_open=c1.iso(opened),
                          market_close=c1.iso(opened + timedelta(days=5)),
                          entry_cutoff=c1.iso(opened + timedelta(days=4))))
    calendar = dict(market_calendar=dict(calendar_id="INVENTED_LANE_A_FIXTURE", version="1", weeks=weeks),
                    entry_delay_after_open_hours=1, force_flat_before_close_hours=1)
    dropped = set(drop)
    points = [START - timedelta(days=30)]
    for w in observed_weeks:
        for h in range(120):
            if (w, h) in dropped or (keep is not None and not keep(w, h)):
                continue
            points.append(week_open(w, dst_from) + timedelta(hours=h))
    ts = np.array([int(t.timestamp()) for t in points], dtype=np.int64)
    rng = np.random.default_rng(6102026)
    x = rng.normal(size=(len(ts), 20))
    price = 1.1 + np.cumsum(rng.normal(scale=1e-3, size=len(ts))) + 1e-4 * np.arange(len(ts))
    path = tmp_path / "input.npz"
    np.savez(path, ts=ts, x=x, price=price)
    features = [f"selected_{i}" for i in range(20)]
    manifest = dict(schema="i6n.source_clock.v1", status="RESOLVED", coverage_mode="complete",
                    data_kind="historical", input_sha256=file_hash(path), features=features,
                    calendar_sha256=c1.canonical_digest(calendar), timestamp_semantics="UTC_BAR_END",
                    availability_policy="AT_TIMESTAMP", clock_policy="invented lane A fixture only")
    clock = tmp_path / "clock.json"
    return dict(
        schema="i6n.historical_cell.v1", input=dict(path=str(path), sha256=file_hash(path)),
        source_clock=dict(path=str(clock), sha256=write_json(clock, manifest)), calendar=calendar,
        evaluation_week_id="210", evaluation_split=dict(split_id="DEVELOPMENT",
        start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", label_available_by=None),
        coverage_mode="complete", role="MANAGEMENT", horizon_hours=horizon, samples=samples,
        spacing_hours=spacing, features=features,
        model={**DEFAULT_CONFIG, "width": 4, "reducer_filters": [4], "reducer_strides": [1]},
        device="cpu", asset="UNIT_FIXTURE", unit="PRICE", training=dict(PLAN_TRAINING),
        resources=dict(max_input_bytes=4_000_000, max_rows=50_000, max_samples=433,
                       max_materialized_bytes=50_000_000, max_batch_bytes=50_000_000,
                       max_updates=10_000, max_parameters=200_000, max_seconds=110))


def genome(search, horizon, *, dx, dy=None, multiplier=1, core=None):
    g = search.default_genome(horizon)
    g.update(delta_x=dx, multiplier=multiplier)
    if dy is not None:
        g.update(output_mode="trajectory", delta_y=dy)
    if core is not None:
        g["core"] = dict(core)
    return g
