"""Invented CPU fixtures only: not historical fit/clock/cost evidence."""

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from tools import i6n_historical_cell as cell
from tools.i6n_model import DEFAULT_CONFIG
from tools.i6m_data import file_hash
from tools.i6n_contracts import eligibility as c1


def write_json(path, value):
    path.write_text(json.dumps(value))
    return file_hash(path)


@pytest.fixture
def fixture(tmp_path):
    # A complete invented four-year calendar, with only a few observed bars.
    start = datetime(2019, 12, 29, 22, tzinfo=timezone.utc)
    weeks = []
    for i in range(211):
        opened = start + timedelta(days=7*i)
        weeks.append(dict(week_id=str(i), market_open=c1.iso(opened),
                          market_close=c1.iso(opened + timedelta(days=5)),
                          entry_cutoff=c1.iso(opened + timedelta(days=4))))
    calendar = dict(market_calendar=dict(calendar_id="INVENTED_UNIT_FIXTURE", version="1", weeks=weeks),
                    entry_delay_after_open_hours=1, force_flat_before_close_hours=1)
    points = [start - timedelta(days=10)]
    for i in (10, 207, 210):
        points.extend(start + timedelta(days=7*i, hours=h) for h in range(10))
    ts = np.array([int(t.timestamp()) for t in points], dtype=np.int64)
    x = np.arange(len(ts)*20, dtype=float).reshape(-1, 20)
    x[3, 0] = np.nan
    price = 1.2 + .001*np.arange(len(ts))**2
    path = tmp_path / "input.npz"
    np.savez(path, ts=ts, x=x, price=price)
    features = [f"selected_{i}" for i in range(20)]
    manifest = dict(schema="i6n.source_clock.v1", status="RESOLVED", coverage_mode="complete",
                    data_kind="historical", input_sha256=file_hash(path), features=features,
                    calendar_sha256=c1.canonical_digest(calendar), timestamp_semantics="UTC_BAR_END",
                    availability_policy="AT_TIMESTAMP", clock_policy="invented unit fixture only")
    mp = tmp_path / "clock.json"
    cfg = dict(schema="i6n.historical_cell.v1", input=dict(path=str(path), sha256=file_hash(path)),
               source_clock=dict(path=str(mp), sha256=write_json(mp, manifest)), calendar=calendar,
               evaluation_week_id="210", evaluation_split=dict(split_id="DEVELOPMENT",
               start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", label_available_by=None),
               coverage_mode="complete", role="MANAGEMENT", horizon_hours=1, samples=3,
               spacing_hours=1, features=features, model={**DEFAULT_CONFIG, "width": 2,
               "reducer_filters": [2], "reducer_strides": [1]}, device="cpu", asset="UNIT_FIXTURE",
               unit="PRICE", training=dict(max_epochs=1, patience=1, min_delta=0., batch_size=4,
               learning_rate=.001, weight_decay=0., huber_delta=1., seed=42, head_weights=[1.]),
               resources=dict(max_input_bytes=2_000_000, max_rows=50_000, max_samples=8,
               max_materialized_bytes=2_000_000, max_batch_bytes=1_000_000,
               max_updates=100, max_parameters=100_000, max_seconds=90))
    return cfg, manifest


def test_complete_grid_cutoff_and_train_only_scales(fixture):
    cfg, _ = fixture
    data = cell.prepare(cfg)
    parts = data["parts"]
    assert parts.coverage_complete and parts.coverage_mode == "complete"
    assert parts.fit_cutoff == c1.parse_instant("close", cfg["calendar"]["market_calendar"]["weeks"][-2]["market_close"])
    assert parts.history_start == parts.fit_cutoff.replace(year=parts.fit_cutoff.year-4)
    assert parts.inner_start == parts.fit_cutoff - timedelta(days=28)
    assert len(parts.train.origins) > 24000 > len(data["ts"])
    assert parts.train.exclusions["INPUT_UNAVAILABLE"] > 20000
    assert len(parts.evaluation.origins) == 120 and parts.evaluation.n_eligible == 8
    fit = data["fit_indices"]
    assert all(parts.history_start <= datetime.fromtimestamp(int(data["ts"][i]), timezone.utc) < parts.inner_start for i in fit)
    assert data["target_scale"].shape == (1,) and data["target_scale"][0] > 0
    before = data["scaler"].sha256(), data["target_center"].copy(), data["target_scale"].copy()
    with np.load(cfg["input"]["path"]) as ar:
        ts, x, p = ar["ts"], ar["x"], ar["price"]
    late = ts >= parts.inner_start.timestamp()
    x[late] += 10000
    p[late] *= 2
    np.savez(cfg["input"]["path"], ts=ts, x=x, price=p)
    repin(cfg)
    changed = cell.prepare(cfg)
    assert changed["scaler"].sha256() == before[0]
    np.testing.assert_array_equal(changed["target_center"], before[1])
    np.testing.assert_array_equal(changed["target_scale"], before[2])


def repin(cfg, **clock_changes):
    cfg["input"]["sha256"] = file_hash(cfg["input"]["path"])
    path = cell.Path(cfg["source_clock"]["path"])
    clock = json.loads(path.read_text())
    clock.update(input_sha256=cfg["input"]["sha256"], **clock_changes)
    cfg["source_clock"]["sha256"] = write_json(path, clock)


@pytest.mark.parametrize("change,match", [
    ("unresolved", "clock"), ("synthetic", "complete"), ("test", "TEST"),
    ("features", "features"), ("digest", "digest"), ("calendar", "CALENDAR"),
    ("samples", "samples"), ("updates", "updates"), ("memory", "bytes"),
    ("multihead", "one horizon"), ("clock_binding", "clock"),
])
def test_fail_closed_before_fitting(fixture, change, match):
    cfg, _ = fixture
    if change == "unresolved":
        repin(cfg, status="UNRESOLVED")
    elif change == "synthetic":
        cfg["coverage_mode"] = "synthetic_sparse"
    elif change == "test":
        cfg["evaluation_split"]["split_id"] = "TEST"
    elif change == "features":
        cfg["features"][1] = cfg["features"][0]
    elif change == "digest":
        cfg["input"]["sha256"] = "0"*64
    elif change == "calendar":
        cfg["calendar"]["market_calendar"]["weeks"].pop(20)
        repin(cfg, calendar_sha256=c1.canonical_digest(cfg["calendar"]))
    elif change == "samples":
        cfg["samples"] = 100
    elif change == "updates":
        cfg["resources"]["max_updates"] = 1
    elif change == "memory":
        cfg["resources"]["max_batch_bytes"] = 1
    elif change == "multihead":
        cfg["model"]["n_horizons"] = 2
    else:
        repin(cfg, timestamp_semantics="UNRESOLVED_LOCAL")
    with pytest.raises(ValueError, match=match):
        cell.prepare(cfg)


def test_npz_test_rows_and_unresolved_not_decoded(fixture, monkeypatch):
    cfg, _ = fixture
    repin(cfg, status="UNRESOLVED")
    monkeypatch.setattr(cell, "load_input", lambda *a: pytest.fail("decoded unresolved input"))
    with pytest.raises(ValueError, match="clock"):
        cell.prepare(cfg)


def test_npz_test_rows_are_rejected(fixture):
    cfg, _ = fixture
    with np.load(cfg["input"]["path"]) as ar:
        ts, x, p = ar["ts"], ar["x"], ar["price"]
    ts[-1] = 1735689600
    np.savez(cfg["input"]["path"], ts=ts, x=x, price=p)
    repin(cfg)
    with pytest.raises(ValueError, match="TEST"):
        cell.prepare(cfg)


def test_sample_spacing_not_limited_to_old_compact_pilot(fixture):
    cfg, _ = fixture
    cfg.update(samples=19, spacing_hours=2)
    cfg["resources"]["max_samples"] = 19
    data = cell.prepare(cfg)
    x, _, _, origins = cell.materialize(data, data["parts"].train, cfg)
    assert x.shape[1:] == (19, 41)
    np.testing.assert_array_equal(x[:, -1, :-1], data["scaled"][np.searchsorted(data["ts"], origins)])


@pytest.mark.parametrize("limit", ["max_input_bytes", "max_rows", "max_materialized_bytes"])
def test_additional_resource_limits_are_admission_guards(fixture, limit):
    cfg, _ = fixture
    cfg["resources"][limit] = 1
    with pytest.raises(ValueError, match="bytes|rows"):
        cell.prepare(cfg)


def test_gpu_unavailable_is_refused_not_retried_on_cpu(monkeypatch):
    import tensorflow as tf
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda _: [])
    with pytest.raises(ValueError, match="no fallback"):
        cell.device_context("gpu")


def test_windows_elapsed_time_current_origin_gap_and_source_mask():
    ts = np.array([0, 3600, 10800, 14400], dtype=np.int64)
    scaled = np.ones((4, 40), np.float32)
    scaled[:, 0] = [1, 2, 3, 4]
    scaled[2, 1] = 0
    x = cell.windows(ts, scaled, np.array([10800, 14400, 7200]), 1, 3)
    np.testing.assert_array_equal(x[0, :, 0], [2, 0, 3])
    np.testing.assert_array_equal(x[0, :, -1], [1, 0, 1])
    assert x[0, -1, 1] == 0 and x[1, -1, 0] == 4
    assert not x[2].any()  # Absent current origin has no admissible input.


def test_materialization_is_bounded_and_masks_match(fixture, monkeypatch):
    cfg, _ = fixture
    data = cell.prepare(cfg)
    real, calls = cell.windows, []
    def observed(*args):
        calls.append(len(args[2]))
        return real(*args)
    monkeypatch.setattr(cell, "windows", observed)
    x, y, mask, origins = cell.materialize(data, data["parts"].train, cfg)
    assert max(calls) <= cfg["training"]["batch_size"]
    assert len(origins) == data["parts"].train.n_eligible
    assert x.shape == (len(origins), 3, 41) and mask.all()
    assert np.isfinite(y).all()


def test_tiny_cpu_fit_checkpoint_c2_and_verified_reuse(fixture, tmp_path, monkeypatch):
    cfg, _ = fixture
    config_path = tmp_path / "config.json"
    write_json(config_path, cfg)
    output = tmp_path / "run"
    events, real_writer = [], cell.atomic_json
    def capture(path, value):
        real_writer(path, value)
        if path.name == "STATUS.json" and "progress" in value:
            events.append((value["progress"]["event"], value["eta_seconds"]))
    monkeypatch.setattr(cell, "atomic_json", capture)
    result = cell.run(config_path, output)
    assert result["state"] == "COMPLETED" and result["coverage_complete"]
    assert result["training"]["observed_updates"] > 0
    assert result["training"]["initial_weights_sha256"] != result["training"]["restored_weights_sha256"]
    assert result["checkpoint_reload_verified"]
    groups = [g for g in result["metrics"]["groups"] if g["population_kind"] == "S"]
    assert all(g["coverage"] == 1 and g["scale"] == "ORIGINAL_PRICE" for g in groups)
    assert result["evaluation_pairs"] == groups[0]["n"]
    rows = json.loads((output / "metric_input.json").read_text())["forecasts"]
    assert groups[0]["mae"] == pytest.approx(np.mean([abs(r["prediction"]-r["price_target"]) for r in rows]))
    assert groups[0]["naive_mae"] == pytest.approx(np.mean([abs(r["price_origin"]-r["price_target"]) for r in rows]))
    status = json.loads((output / "STATUS.json").read_text())
    assert status["eta_seconds"] == 0
    assert any(e == "update" for e, _ in events)
    assert any(e == "epoch" and eta is not None for e, eta in events)
    assert cell.status(config_path, output) == (result, 0)
    monkeypatch.setattr(cell, "execute", lambda *a: pytest.fail("repeated fit"))
    assert cell.run(config_path, output) == result
    metric_path = output / "metric_input.json"
    original = metric_path.read_bytes()
    doc = json.loads(original)
    doc["forecasts"][0]["prediction"] += .1
    changed_digest = write_json(metric_path, doc)
    receipt = dict(result, artifacts={**result["artifacts"], "metric_input.json": changed_digest})
    status["result_sha256"] = write_json(output / "RESULT.json", receipt)
    write_json(output / "STATUS.json", status)
    with pytest.raises(ValueError, match="RESULT_MISMATCH"):
        cell.run(config_path, output)
    metric_path.write_bytes(original)
    status["result_sha256"] = write_json(output / "RESULT.json", result)
    write_json(output / "STATUS.json", status)
    cfg["training"]["seed"] += 1
    write_json(config_path, cfg)
    with pytest.raises(ValueError, match="identity"):
        cell.run(config_path, output)
    cfg["training"]["seed"] -= 1
    write_json(config_path, cfg)
    (output / "model.keras").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="digest"):
        cell.run(config_path, output)


def test_failure_and_orphan_never_retry(fixture, tmp_path, monkeypatch):
    cfg, _ = fixture
    config_path = tmp_path / "config.json"
    write_json(config_path, cfg)
    output = tmp_path / "run"
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(cell, "execute", fail)
    with pytest.raises(RuntimeError):
        cell.run(config_path, output)
    assert json.loads((output / "STATUS.json").read_text())["state"] == "FAILED"
    failed, code = cell.status(config_path, output)
    assert failed["state"] == "FAILED" and code != 0
    with pytest.raises(ValueError, match="retry"):
        cell.run(config_path, output)
    assert len(calls) == 1
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "model.keras").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="orphan"):
        cell.run(config_path, orphan)


def test_status_interrupted_and_wall_budget_terminal(fixture, tmp_path, monkeypatch):
    cfg, _ = fixture
    cfg["resources"]["max_seconds"] = 1
    path = tmp_path / "config.json"
    write_json(path, cfg)
    output = tmp_path / "run"
    def pause(*args):
        cell.time.sleep(2)
    monkeypatch.setattr(cell, "execute", pause)
    with pytest.raises(TimeoutError, match="wall budget"):
        cell.run(path, output)
    record = json.loads((output / "STATUS.json").read_text())
    assert record["state"] == "FAILED"
    record["state"] = "RUNNING"
    write_json(output / "STATUS.json", record)
    assert cell.status(path, output)[0]["state"] == "INTERRUPTED"
    assert json.loads((output / "STATUS.json").read_text())["state"] == "RUNNING"
    with pytest.raises(ValueError, match="retry"):
        cell.run(path, output)
