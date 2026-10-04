"""Incremental PS4 profile contract. These tests are the specification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps4_transform_profile"))

import run_ps4_incremental_profiles as inc  # noqa: E402
import run_ps4_transform_profiles as published  # noqa: E402

INNER = [
    {"name": "inner_2019", "train_rows": [0, 120], "val_rows": [140, 160]},
    {"name": "inner_2020", "train_rows": [0, 150], "val_rows": [170, 190]},
    {"name": "inner_2021", "train_rows": [0, 180], "val_rows": [200, 220]},
    {"name": "inner_2022", "train_rows": [0, 210], "val_rows": [230, 250]},
    {"name": "inner_2023", "train_rows": [0, 240], "val_rows": [260, 280]},
]


def _folds(path: Path, folds=None) -> None:
    path.write_text(json.dumps({"schema": "laneA_folds.v1", "folds": folds or INNER}))


def _series(path: Path, columns: dict) -> None:
    np.savez(path, **columns)


def _terminal(path: Path, series: Path, feature: str, **overrides) -> Path:
    payload = {
        "schema": "ut_pilot_run.v1",
        "status": "COMPLETED",
        "features": [feature],
        "seed": 0,
        "code_commit": "1" * 40,
        "families": ["identity", "random", "ae", "dae"],
        "results_sha256": "2" * 64,
        "series_sha256": inc.sha256_file(series),
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _column(n: int, scale: float) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, n, dtype=np.float32) * np.float32(scale)
    values[3] = np.nan
    return values


def _boom(*_args, **_kwargs):
    raise AssertionError("series bytes or the profiler ran")


def _metrics(rows: list[dict]) -> bytes:
    cleaned = []
    for row in rows:
        item = dict(row)
        item.pop("source_digests", None)
        cleaned.append(item)
    return published.scientific_bytes(cleaned)


def test_fixture_profiles_prefixed_and_bare_columns(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {
        "x__px.logret_1h": _column(300, 1.0),
        "px.range_oc": _column(300, 2.0),
        "Y_l": _column(300, 9.0),
    })
    folds = tmp_path / "folds.json"
    _folds(folds)
    out = tmp_path / "out"
    first_terminal = _terminal(tmp_path / "first_terminal.json", series, "px.logret_1h")
    second_terminal = _terminal(tmp_path / "second_terminal.json", series, "px.range_oc")
    prefixed = inc.execute_unit(
        series, folds, "px.logret_1h", "inner_2019", out,
        dataset_id="fixture", terminal_manifest=first_terminal,
    )
    bare = inc.execute_unit(
        series, folds, "px.range_oc", "inner_2020", out,
        dataset_id="fixture", terminal_manifest=second_terminal,
    )
    assert prefixed["unit_status"] == "MEASURED"
    assert bare["unit_status"] == "MEASURED"
    assert prefixed["metric_rows"] > 0 and bare["metric_rows"] > 0
    left = json.loads(inc.unit_path(out, "px.logret_1h", "inner_2019").read_text())
    right = json.loads(inc.unit_path(out, "px.range_oc", "inner_2020").read_text())
    assert left["train_rows"] == [0, 120]
    assert right["train_rows"] == [0, 150]
    assert {row["fold"] for row in left["rows"]} == {"inner_2019"}
    assert inc.classify_unit(inc.unit_path(out, "px.logret_1h", "inner_2019")) == "MEASURED"
    assert _metrics(left["rows"]) != _metrics(right["rows"])


def test_rows_outside_train_do_not_change_the_unit(tmp_path: Path) -> None:
    folds = tmp_path / "folds.json"
    _folds(folds)
    base = _column(300, 1.0)
    mutated = base.copy()
    mutated[120:] = np.float32(999.0)
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _series(first, {"x__px.logret_1h": base})
    _series(second, {"x__px.logret_1h": mutated})
    terminal_a = _terminal(tmp_path / "terminal_a.json", first, "px.logret_1h")
    terminal_b = _terminal(tmp_path / "terminal_b.json", second, "px.logret_1h")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    inc.execute_unit(first, folds, "px.logret_1h", "inner_2019", out_a,
                     dataset_id="fixture", terminal_manifest=terminal_a)
    inc.execute_unit(second, folds, "px.logret_1h", "inner_2019", out_b,
                     dataset_id="fixture", terminal_manifest=terminal_b)
    rows_a = json.loads(inc.unit_path(out_a, "px.logret_1h", "inner_2019").read_text())["rows"]
    rows_b = json.loads(inc.unit_path(out_b, "px.logret_1h", "inner_2019").read_text())["rows"]
    assert _metrics(rows_a) == _metrics(rows_b)
    mutated[10] = np.float32(-50.0)
    third = tmp_path / "c.npz"
    _series(third, {"x__px.logret_1h": mutated})
    terminal_c = _terminal(tmp_path / "terminal_c.json", third, "px.logret_1h")
    out_c = tmp_path / "out_c"
    inc.execute_unit(third, folds, "px.logret_1h", "inner_2019", out_c,
                     dataset_id="fixture", terminal_manifest=terminal_c)
    rows_c = json.loads(inc.unit_path(out_c, "px.logret_1h", "inner_2019").read_text())["rows"]
    assert _metrics(rows_c) != _metrics(rows_a)


def test_resume_skips_completed_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    out = tmp_path / "out"
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    digest = inc.sha256_file(series)
    first = inc.execute_unit(
        series, folds, "px.logret_1h", "inner_2019", out,
        dataset_id="fixture", expected_digest=digest, terminal_manifest=terminal,
    )
    assert first["resumed"] is False
    stored = inc.unit_path(out, "px.logret_1h", "inner_2019")
    snapshot = stored.read_bytes()
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    second = inc.execute_unit(
        series, folds, "px.logret_1h", "inner_2019", out,
        dataset_id="fixture", expected_digest=digest, terminal_manifest=terminal,
    )
    assert second["resumed"] is True
    assert second["unit_status"] == "MEASURED"
    assert stored.read_bytes() == snapshot


def test_resume_checks_source_digest_before_returning_measured(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    expected = inc.sha256_file(series)
    out = tmp_path / "out"
    inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                     expected_digest=expected, terminal_manifest=terminal)
    _series(series, {"x__px.logret_1h": _column(300, 9.0)})
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                         expected_digest=expected, terminal_manifest=terminal)
    assert caught.value.code == "DIGEST_MISMATCH"


def test_partial_unit_is_not_measured_and_is_recomputed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"px.range_oc": _column(300, 1.5)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    out = tmp_path / "out"
    terminal = _terminal(tmp_path / "terminal.json", series, "px.range_oc")
    path = inc.unit_path(out, "px.range_oc", "inner_2019")
    path.parent.mkdir(parents=True)
    path.write_text("{\"unit_status\": \"PARTIAL\"")
    assert inc.classify_unit(path) == "PENDING"
    assert inc.classify_unit(path) != "MEASURED"
    leftover = path.with_suffix(path.suffix + ".tmp")
    leftover.write_text("{")
    assert inc.classify_unit(leftover) == "PENDING"
    calls = {"n": 0}
    real = inc.profile_fold

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(inc, "profile_fold", _count)
    result = inc.execute_unit(series, folds, "px.range_oc", "inner_2019", out,
                              dataset_id="fixture", terminal_manifest=terminal)
    assert calls["n"] == 1
    assert result["unit_status"] == "MEASURED"
    assert inc.classify_unit(path) == "MEASURED"
    assert not leftover.exists()


def test_target_column_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(30, 1.0), "Y_l_24h": _column(30, 1.0)})
    targets = tmp_path / "targets.npz"
    _series(targets, {"Y_l": _column(30, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "Y_l_24h", "inner_2019", tmp_path / "out")
    assert caught.value.code == "TARGET_COLUMN"
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(targets, folds, "px.logret_1h", "inner_2019", tmp_path / "out2")
    assert caught.value.code == "TARGET_COLUMN"
    assert list((tmp_path / "out").glob("units/*")) == []


def test_external_validation_and_test_splits_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    series = tmp_path / "series.npz"
    folds = tmp_path / "folds.json"
    for split, code in (("external_validation", "EXTERNAL_VALIDATION_SPLIT"), ("test", "TEST_SPLIT")):
        with pytest.raises(inc.ProfileRefusal) as caught:
            inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", tmp_path / "out", split=split)
        assert caught.value.code == code
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "outer_test", tmp_path / "out")
    assert caught.value.code == "TEST_SPLIT"
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "external_validation", tmp_path / "out")
    assert caught.value.code == "EXTERNAL_VALIDATION_SPLIT"


def test_already_measured_transforms_are_not_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inc, "execute_unit", _boom)
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    names = list(inc.ALREADY_MEASURED_TRANSFORMS)
    assert set(names) == {
        "tv.hilbert_amp", "tv.kalman_dev", "tv.stl_dev", "tv.stl_seasonal",
        "tv.wav_d1", "tv.wav_d2", "tv.wav_d3", "tv.wav_d4", "tv.wav_d5",
        "tv.mt_band_6_48h",
    }
    result = inc.drive(names, Path("missing.npz"), Path("folds.json"), Path("out"), fold="inner_2019")
    assert result["skipped_already_measured"] == names
    assert result["units"] == []
    monkeypatch.undo()
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    for name in names:
        with pytest.raises(inc.ProfileRefusal) as caught:
            inc.execute_unit(Path("missing.npz"), Path("folds.json"), name, "inner_2019", Path("out"))
        assert caught.value.code == "ALREADY_MEASURED"


def test_digest_mismatch_does_not_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    monkeypatch.setattr(inc, "profile_fold", _boom)
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(
            series, folds, "px.logret_1h", "inner_2019", tmp_path / "out",
            expected_digest="0" * 64, terminal_manifest=terminal,
        )
    assert caught.value.code == "DIGEST_MISMATCH"
    assert inc.classify_unit(inc.unit_path(tmp_path / "out", "px.logret_1h", "inner_2019")) == "ABSENT"


def test_failed_unit_is_not_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    out = tmp_path / "out"
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")

    def _fail(*_args, **_kwargs):
        raise RuntimeError("profiler failed")

    monkeypatch.setattr(inc, "profile_fold", _fail)
    result = inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                              dataset_id="fixture", terminal_manifest=terminal)
    path = inc.unit_path(out, "px.logret_1h", "inner_2019")
    assert result["unit_status"] == "FAILED"
    assert inc.classify_unit(path) == "FAILED"
    assert inc.classify_unit(path) != "MEASURED"
    monkeypatch.setattr(inc, "profile_fold", _boom)
    resumed = inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                               dataset_id="fixture", terminal_manifest=terminal)
    assert resumed["resumed"] is True
    assert resumed["unit_status"] == "FAILED"
    tallied = inc.count_units(out / "units")
    assert tallied["counts"] == {"MEASURED": 0, "PENDING": 0, "FAILED": 1}


def test_unit_identity_and_terminal_are_authenticated(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    out = tmp_path / "out"
    inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                     terminal_manifest=terminal)
    unit = inc.unit_path(out, "px.logret_1h", "inner_2019")
    original = json.loads(unit.read_text())
    assert inc.classify_unit(unit) == "MEASURED"

    mutations = {
        "feature_id": "other",
        "fold_id": "inner_2020",
        "train_rows": [0, 150],
        "source_sha256": {"series.npz": "9" * 64, "folds.json": original["source_sha256"]["folds.json"]},
    }
    for field, value in mutations.items():
        changed = dict(original)
        changed[field] = value
        unit.write_text(json.dumps(changed))
        assert inc.classify_unit(unit) == "PENDING", field

    changed = json.loads(json.dumps(original))
    changed["rows"][0]["feature_id"] = "other"
    changed["rows_sha256"] = inc._rows_sha256(changed["rows"])
    unit.write_text(json.dumps(changed))
    assert inc.classify_unit(unit) == "PENDING"

    changed = json.loads(json.dumps(original))
    changed["terminal_identity"]["seed"] = 7
    unit.write_text(json.dumps(changed))
    assert inc.classify_unit(unit) == "PENDING"
    unit.write_text(json.dumps(original))


def test_terminal_manifest_is_required_and_must_match(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", tmp_path / "out")
    assert caught.value.code == "TERMINAL_MANIFEST_ABSENT"
    wrong = _terminal(tmp_path / "wrong.json", series, "other")
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", tmp_path / "out",
                         terminal_manifest=wrong)
    assert caught.value.code == "TERMINAL_IDENTITY_MISMATCH"


def test_completed_unit_refuses_a_different_terminal(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    first = _terminal(tmp_path / "first.json", series, "px.logret_1h")
    second = _terminal(
        tmp_path / "second.json", series, "px.logret_1h",
        code_commit="3" * 40, results_sha256="4" * 64,
    )
    out = tmp_path / "out"
    inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                     terminal_manifest=first)
    with pytest.raises(inc.ProfileRefusal) as caught:
        inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                         terminal_manifest=second)
    assert caught.value.code == "UNIT_IDENTITY_MISMATCH"


def test_legacy_binding_preserves_scientific_rows_and_adds_terminal_identity(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    folds = tmp_path / "folds.json"
    _folds(folds)
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    out = tmp_path / "out"
    inc.execute_unit(series, folds, "px.logret_1h", "inner_2019", out,
                     terminal_manifest=terminal)
    unit = inc.unit_path(out, "px.logret_1h", "inner_2019")
    current = json.loads(unit.read_text())
    rows_before = json.dumps(current["rows"], sort_keys=True)
    legacy = dict(current)
    legacy["schema"] = inc.LEGACY_UNIT_SCHEMA
    legacy["fold"] = legacy.pop("fold_id")
    legacy.pop("terminal_identity")
    legacy.pop("identity_sha256")
    unit.write_text(json.dumps(legacy))
    assert inc.classify_unit(unit) == "PENDING"
    result = inc.bind_legacy_unit(unit, terminal)
    rebound = json.loads(unit.read_text())
    assert result["scientific_rows_recomputed"] is False
    assert json.dumps(rebound["rows"], sort_keys=True) == rows_before
    assert rebound["terminal_identity"]["results_sha256"] == "2" * 64
    assert inc.classify_unit(unit) == "MEASURED"


def _evidence_with_terminal(root: Path, *, write_series: bool, features: list[str]) -> None:
    batch = root / "laneB" / "batch_001"
    batch.mkdir(parents=True)
    columns = {"x__" + feature: _column(32, 1.0) for feature in features}
    series = batch / "series.npz"
    if write_series:
        _series(series, columns)
        digest = inc.sha256_file(series)
    else:
        digest = "ab" * 32
    manifest = {"schema": "ps2_batch.v1", "features": features, "series": {"file": "series.npz", "sha256": digest}}
    (batch / "batch_manifest.json").write_text(json.dumps(manifest))
    cost = root / "laneE" / "feature_cost.csv"
    cost.parent.mkdir(parents=True)
    lines = ["batch,feature,stage,status,series_sha256"]
    for feature in features:
        lines.append(f"batch_001,{feature},tier_1,DONE,{digest}")
    cost.write_text("\n".join(lines) + "\n")


def test_selector_skips_measured_names_and_missing_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    absent = tmp_path / "absent"
    _evidence_with_terminal(absent, write_series=False, features=["px.logret_1h"])
    missing = inc.select_one_feature_fold(absent)
    assert missing["state"] == "PENDING"
    assert missing["reason"] == "NO_LOCAL_TRAIN_SERIES_IN_WORKTREE_EVIDENCE"
    blocked = tmp_path / "blocked"
    _evidence_with_terminal(blocked, write_series=True, features=list(inc.ALREADY_MEASURED_TRANSFORMS))
    skipped = inc.select_one_feature_fold(blocked)
    assert skipped["state"] == "PENDING"
    assert skipped["reason"] == "NO_AUTHENTIC_TERMINAL_SERIES_COLUMN"
    ready_root = tmp_path / "ready"
    _evidence_with_terminal(
        ready_root, write_series=True, features=["tv.hilbert_amp", "px.logret_1h"],
    )
    chosen = inc.select_one_feature_fold(ready_root)
    assert chosen["state"] == "READY"
    assert chosen["feature_id"] == "px.logret_1h"
    assert chosen["fold"] == "inner_2019"
    assert chosen["feature_id"] not in inc.ALREADY_MEASURED_TRANSFORMS


def test_publish_pending_when_series_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inc, "profile_fold", _boom)
    monkeypatch.setattr(inc, "load_series_column", _boom)
    evidence = tmp_path / "evidence"
    _evidence_with_terminal(evidence, write_series=False, features=["px.logret_1h"])
    folds = tmp_path / "folds.json"
    _folds(folds)
    destination = tmp_path / "REPORT.json"
    report = inc.publish_report(evidence, folds, tmp_path / "units", destination, canonical_bounds=False)
    assert report["counts"] == {"MEASURED": 0, "PENDING": 1, "FAILED": 0}
    assert report["selection_denominator"] == 366
    assert report["selection_closed"] is False
    assert report["ps4_selects_features"] is False
    assert report["candidate"]["state"] == "PENDING"
    assert destination.is_file()
    assert "NO_FEATURE_SELECTION_DECISION" in report["flags"]
    assert report["execution_receipt"]["status"] == "NOT_RETAINED"


def test_pending_candidate_is_included_in_report_counts(tmp_path: Path) -> None:
    series = tmp_path / "series.npz"
    _series(series, {"x__px.logret_1h": _column(300, 1.0)})
    terminal = _terminal(tmp_path / "terminal.json", series, "px.logret_1h")
    folds = tmp_path / "folds.json"
    _folds(folds)
    out = tmp_path / "out"
    for fold in inc.INNER_FOLDS:
        inc.execute_unit(series, folds, "px.logret_1h", fold, out,
                         terminal_manifest=terminal)
    evidence = tmp_path / "evidence"
    _evidence_with_terminal(evidence, write_series=False, features=["other"])
    report = inc.publish_report(evidence, folds, out / "units", tmp_path / "REPORT.json")
    assert report["counts"] == {"MEASURED": 5, "PENDING": 1, "FAILED": 0}
    assert report["candidate"]["state"] == "PENDING"
