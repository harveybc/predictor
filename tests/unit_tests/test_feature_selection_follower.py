from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import feature_selection_follower as follower


FOLDS = ["inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _runner(path: Path) -> None:
    path.write_text(
        '''import hashlib, json
from pathlib import Path
INNER_FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unit_path(out, feature, fold):
    return Path(out) / "units" / f"{feature.replace('/', '_')}__{fold}.json"

def classify_unit(path, expected_feature_id="", expected_fold_id="", expected_acceptance=None, **kwargs):
    path = Path(path)
    try:
        body = json.loads(path.read_text())
    except Exception:
        return "PENDING"
    valid = (body.get("unit_status") == "MEASURED"
             and body.get("feature_id") == expected_feature_id
             and body.get("fold_id") == expected_fold_id
             and expected_acceptance
             and expected_acceptance.get("unit_sha256") == sha(path)
             and expected_acceptance.get("source_sha256") == body.get("source_sha256")
             and expected_acceptance.get("terminal_identity") == body.get("terminal_identity")
             and (kwargs.get("expected_source_sha256") is None
                  or kwargs["expected_source_sha256"] == body.get("source_sha256"))
             and (kwargs.get("expected_terminal_identity") is None
                  or kwargs["expected_terminal_identity"] == body.get("terminal_identity")))
    return "MEASURED" if valid else "PENDING"

def execute_unit(series, folds_path, feature_id, fold_name, out, **kwargs):
    Path(out).mkdir(parents=True, exist_ok=True)
    log = Path(out) / "calls.log"
    with log.open("a") as handle:
        handle.write(f"{feature_id}:{fold_name}\\n")
    terminal = json.loads(Path(kwargs["terminal_manifest"]).read_text())
    payload = {
        "unit_status": "MEASURED", "feature_id": feature_id, "fold_id": fold_name,
        "source_sha256": {"series.npz": sha(series), "folds.json": sha(folds_path)},
        "terminal_identity": {"results_sha256": terminal["results_sha256"]},
    }
    destination = unit_path(out, feature_id, fold_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    temporary.replace(destination)
    return {"unit_status": "MEASURED", "unit": destination.name}
'''
    )


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    runner = tmp_path / "runner.py"
    _runner(runner)
    folds = tmp_path / "folds.json"
    _write_json(folds, {"folds": FOLDS})
    batch = tmp_path / "batches" / "batch_001"
    batch.mkdir(parents=True)
    series = batch / "series.npz"
    series.write_bytes(b"TRAIN-SERIES-ONLY")
    manifest = {
        "schema": "ps2_batch.v1",
        "batch_id": "batch_001",
        "features": ["feature.one"],
        "series": {"file": "series.npz", "sha256": _sha(series)},
    }
    _write_json(batch / "batch_manifest.json", manifest)
    result = tmp_path / "results" / "feature.one"
    result.mkdir(parents=True)
    rows = result / "results.jsonl"
    rows.write_bytes(b'{"metric": 1}\n')
    terminal = {
        "schema": "ut_pilot_run.v1",
        "status": "COMPLETED",
        "features": ["feature.one"],
        "batch_id": "batch_001",
        "batch_manifest_sha256": _sha(batch / "batch_manifest.json"),
        "series_sha256": _sha(series),
        "results_sha256": _sha(rows),
        "seed": 0,
        "code_commit": "abc",
        "families": ["identity", "ae"],
    }
    _write_json(result / "run_manifest.json", terminal)
    output = tmp_path / "ps4"
    config = {
        "result_roots": [str(tmp_path / "results")],
        "ps2_batch_dirs": [str(tmp_path / "batches")],
        "folds_file": str(folds),
        "ps4_runner": str(runner),
        "ps4_output_dir": str(output),
        "ps4_units_dir": str(output / "units"),
        "accepted_units_path": str(output / "accepted_units.json"),
        "status_path": str(output / "STATUS.json"),
        "poll_interval_seconds": 0.01,
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return config_path, result


def _run_once(config_path: Path) -> dict:
    settings = follower.load_settings(config_path)
    runner = follower._load_runner(settings.ps4_runner)
    return follower.run_cycle(settings, runner, started_at="2026-01-01T00:00:00Z")


def test_tampered_results_are_rejected_before_ps4(tmp_path: Path) -> None:
    config, result = _campaign(tmp_path)
    (result / "results.jsonl").write_bytes(b"tampered")

    status = _run_once(config)

    assert status["counts"] == {
        "discovered": 1, "accepted": 0, "profiled": 0, "rejected": 1, "pending": 0
    }
    assert "digest mismatch" in status["rejected"][0]["reason"]
    assert not (tmp_path / "ps4" / "calls.log").exists()


def test_second_cycle_is_idempotent(tmp_path: Path) -> None:
    config, _ = _campaign(tmp_path)
    first = _run_once(config)
    calls = (tmp_path / "ps4" / "calls.log").read_text().splitlines()

    second = _run_once(config)

    assert first["profiled"] == second["profiled"] == ["feature.one"]
    assert len(calls) == 5
    assert (tmp_path / "ps4" / "calls.log").read_text().splitlines() == calls
    accepted = json.loads((tmp_path / "ps4" / "accepted_units.json").read_text())
    assert len(accepted["units"]) == 5


def test_partial_units_resume_only_missing_folds(tmp_path: Path) -> None:
    config, _ = _campaign(tmp_path)
    settings = follower.load_settings(config)
    runner = follower._load_runner(settings.ps4_runner)
    batches, _ = follower.load_batches(settings)
    terminals, _, _ = follower.discover_terminals(settings)
    terminal = terminals[0]
    for fold in FOLDS[:2]:
        runner.execute_unit(
            batches["feature.one"].series_path,
            settings.folds_file,
            "feature.one",
            fold,
            settings.ps4_output_dir,
            terminal_manifest=terminal.manifest_path,
        )
    before = (tmp_path / "ps4" / "calls.log").read_text().splitlines()

    status = follower.run_cycle(settings, runner, started_at="2026-01-01T00:00:00Z")

    after = (tmp_path / "ps4" / "calls.log").read_text().splitlines()
    assert status["profiled"] == ["feature.one"]
    assert len(after) - len(before) == 3


def test_status_is_published_by_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = _campaign(tmp_path)
    real_replace = follower.os.replace
    replacements: list[tuple[Path, Path]] = []

    def observed_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(follower.os, "replace", observed_replace)
    status = _run_once(config)
    status_path = tmp_path / "ps4" / "STATUS.json"

    assert json.loads(status_path.read_text()) == status
    assert any(destination == status_path and source.suffix == ".tmp" for source, destination in replacements)
    assert not list(status_path.parent.glob(".STATUS.json.*.tmp"))
