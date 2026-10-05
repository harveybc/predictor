from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from tools.phase1_deployment.prepare import (
    DeploymentError,
    build_combined_bundle,
    build_eth_bundle,
    build_eurusd_bundle,
    run_disposable_fixture,
    verify_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(path: Path, names: list[str], *, batch: str) -> None:
    fields = [
        "feature_id", "family", "source", "unit", "frequency", "support_h",
        "event_time", "availability_time", "transform", "licence", "role",
        "note", "admissibility", "train_rows", "train_finite", "train_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow({
                "feature_id": name,
                "family": f"family-{batch}",
                "source": f"fixture/{batch}",
                "unit": "ratio",
                "frequency": "1h",
                "event_time": "bar end",
                "availability_time": "bar end",
                "role": "feature",
                "admissibility": "ADMISSIBLE",
                "train_rows": "8",
                "train_finite": "8",
                "train_coverage": "1.0",
            })


def _eurusd_sources(root: Path) -> tuple[Path, Path, Path]:
    evidence = root / "evidence"
    retained = root / "retained"
    causal = root / "causal"
    identity = pd.DataFrame({
        "t_decision_utc": pd.date_range("2023-01-02", periods=8, freq="h", tz="UTC"),
        "row_id": range(8),
    })
    batches = {
        "batch_001": ["eur.a", "eur.b"],
        "batch_002": ["eur.c"],
        "batch_003": ["eur.d"],
    }
    for number, (batch, names) in enumerate(batches.items(), start=1):
        source = evidence / batch
        data = retained / batch
        source.mkdir(parents=True)
        data.mkdir(parents=True)
        _write_inventory(source / "admissible_features.csv", names, batch=batch)
        frame = identity.copy()
        for index, name in enumerate(names):
            frame[name] = [float(index + number + value) for value in range(8)]
        frame.to_parquet(data / "features_train.parquet", index=False)
    targets = identity.copy()
    for hours in (1, 2, 3, 4, 5, 6):
        targets[f"Y_s_{hours}h"] = [value / (100 * hours) for value in range(8)]
    for hours in (24, 48, 72, 96, 120, 144):
        targets[f"Y_l_{hours}h"] = [value / (100 * hours) for value in range(8)]
    targets["Y_b_s6"] = [0, 1, 0, -1, 0, 1, 0, -1]
    targets["Y_b_l144"] = [-1, 0, 1, 0, -1, 0, 1, 0]
    targets.to_parquet(retained / "batch_001" / "targets_train.parquet", index=False)
    folds = {
        "schema": "fixture-folds.v1",
        "folds": [{"train_rows": [0, 5], "val_rows": [5, 8]}],
    }
    (retained / "batch_001" / "folds.json").write_text(json.dumps(folds), encoding="utf-8")

    profiles = evidence / "profiles"
    profiles.mkdir()
    for batch, names in batches.items():
        with (profiles / f"{batch}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["feature_id", "metric", "state", "value", "reason", "metrics_version"],
            )
            writer.writeheader()
            for name in names:
                writer.writerow({
                    "feature_id": name,
                    "metric": "missingness",
                    "state": "MEASURED",
                    "value": json.dumps({"n_rows": 8, "missing_fraction": 0.0}),
                    "metrics_version": "fixture.v1",
                })

    causal.mkdir()
    causal_rows = []
    for name in sum(batches.values(), []):
        causal_rows.append({
            "feature_id": name,
            "target": "Y_s_1h",
            "horizon_h": 1,
            "rung1": {"state": "NOT_IDENTIFIED", "raw_state": "NO_ASSOCIATION"},
            "rung2": {"state": "NOT_IDENTIFIED", "raw_state": "NOT_EVALUATED"},
            "rung3": {"state": "NOT_IDENTIFIED", "raw_state": "NOT_EVALUATED"},
        })
    causal_file = causal / "causal_evidence.jsonl"
    causal_file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in causal_rows), encoding="utf-8")
    digest_doc = {"artifacts_sha256": {"causal_evidence.jsonl": _sha(causal_file)}}
    (causal / "digests.json").write_text(json.dumps(digest_doc), encoding="utf-8")
    return evidence, retained, causal


def test_eurusd_bundle_merges_rows_and_adopts_verified_evidence(tmp_path: Path) -> None:
    evidence, retained, causal = _eurusd_sources(tmp_path)
    result = build_eurusd_bundle(
        evidence_root=evidence,
        retained_root=retained,
        causal_root=causal,
        output_root=tmp_path / "bundle",
        expected_features=4,
        expected_targets=1,
        predictor_revision="174e7412",
        causal_revision="5e88fc6",
    )

    merged = pd.read_parquet(result["features_path"])
    assert list(merged.columns) == [
        "t_decision_utc", "row_id", "eur.a", "eur.b", "eur.c", "eur.d",
    ]
    assert result["feature_count"] == 4
    assert result["adoption"]["state"] == "ADOPTED_VERIFIED_EVIDENCE"
    assert result["adoption"]["causal_rows"] == 4
    assert len(result["adoption"]["envelopes"]) == 4
    assert verify_bundle(Path(result["bundle_manifest"]))["state"] == "VERIFIED"


def test_eurusd_bundle_rejects_row_identity_and_duplicate_features(tmp_path: Path) -> None:
    evidence, retained, causal = _eurusd_sources(tmp_path)
    path = retained / "batch_002" / "features_train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[3, "row_id"] = 99
    frame.to_parquet(path, index=False)
    with pytest.raises(DeploymentError, match="row identity"):
        build_eurusd_bundle(
            evidence, retained, causal, tmp_path / "bad-rows",
            expected_features=4, expected_targets=1,
        )

    evidence, retained, causal = _eurusd_sources(tmp_path / "duplicate")
    duplicate = evidence / "batch_003" / "admissible_features.csv"
    _write_inventory(duplicate, ["eur.c"], batch="batch_003")
    frame = pd.read_parquet(retained / "batch_003" / "features_train.parquet")
    frame = frame.rename(columns={"eur.d": "eur.c"})
    frame.to_parquet(retained / "batch_003" / "features_train.parquet", index=False)
    with pytest.raises(DeploymentError, match="duplicate feature"):
        build_eurusd_bundle(
            evidence, retained, causal, tmp_path / "bad-duplicate",
            expected_features=4, expected_targets=1,
        )


def test_eurusd_adoption_rejects_altered_causal_bytes(tmp_path: Path) -> None:
    evidence, retained, causal = _eurusd_sources(tmp_path)
    with (causal / "causal_evidence.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(DeploymentError, match="digest"):
        build_eurusd_bundle(
            evidence, retained, causal, tmp_path / "bad-adoption",
            expected_features=4, expected_targets=1,
        )


def test_eth_bundle_is_separate_provisional_and_exactly_declared(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range("2023-01-01", periods=12, freq="4h"),
        "log_return_1": [value / 1000 for value in range(12)],
        "eth.a": range(12),
        "eth.b": range(12, 24),
        "CLOSE": range(100, 112),
    })
    data = tmp_path / "eth.csv"
    frame.to_csv(data, index=False)
    manifest = {
        "schema": "selected_feature_manifest.v1",
        "status": "FROZEN_DEVELOPMENT",
        "resource": {"path": str(data), "sha256": _sha(data), "timestamp_column": "DATE_TIME"},
        "split": {"train": {"rows": [0, 10]}},
        "features": ["eth.a", "eth.b"],
        "feature_count": 2,
    }
    manifest_path = tmp_path / "eth-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = build_eth_bundle(
        manifest_path=manifest_path,
        dataset_path=data,
        output_root=tmp_path / "eth-bundle",
        expected_features=2,
    )
    inventory = pd.read_csv(result["inventory_path"])
    assert inventory.feature_id.tolist() == ["eth.a", "eth.b"]
    assert set(inventory.population_id) == {"ETH"}
    assert result["point_in_time_state"] == "PROVISIONAL_DEVELOPMENT"
    targets = pd.read_parquet(result["targets_path"])
    assert [f"Y_eth_{hours}h" for hours in (4, 8, 12, 16, 20, 24)] == [
        column for column in targets if column.startswith("Y_eth_")
    ]
    assert targets.loc[9, "Y_eth_4h"] != targets.loc[9, "Y_eth_4h"]
    assert targets.loc[4, "Y_eth_24h"] != targets.loc[4, "Y_eth_24h"]


def test_disposable_three_column_fixture_exercises_stdio_and_orchestrator(tmp_path: Path) -> None:
    result = run_disposable_fixture(tmp_path / "fixture")
    assert result["state"] == "COMPLETE"
    assert result["inventory_total"] == 3
    assert result["worker_stdio_observed"] is True
    assert result["network_calls"] == 0
    assert result["completed"] == 3


def test_disposable_fixture_cli_runs_from_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable, "tools/phase1_deployment/prepare.py",
            "integration-fixture", "--output", str(tmp_path / "cli-fixture"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["state"] == "COMPLETE"


def test_combined_bundle_is_host_neutral_and_contains_runtime_tools(tmp_path: Path) -> None:
    evidence, retained, causal = _eurusd_sources(tmp_path / "eur")
    eth_frame = pd.DataFrame({
        "DATE_TIME": pd.date_range("2023-01-01", periods=12, freq="4h"),
        "log_return_1": [value / 1000 for value in range(12)],
        "eth.a": range(12),
        "eth.b": range(12, 24),
        "CLOSE": range(100, 112),
    })
    eth_data = tmp_path / "eth.csv"
    eth_frame.to_csv(eth_data, index=False)
    eth_manifest = tmp_path / "eth-manifest.json"
    eth_manifest.write_text(json.dumps({
        "schema": "selected_feature_manifest.v1",
        "status": "FROZEN_DEVELOPMENT",
        "resource": {
            "path": str(eth_data), "sha256": _sha(eth_data),
            "timestamp_column": "DATE_TIME",
        },
        "split": {"train": {"rows": [0, 10]}},
        "features": ["eth.a", "eth.b"],
        "feature_count": 2,
    }), encoding="utf-8")

    result = build_combined_bundle(
        evidence_root=evidence,
        retained_root=retained,
        causal_root=causal,
        eth_manifest_path=eth_manifest,
        eth_dataset_path=eth_data,
        output_root=tmp_path / "combined",
        expected_eurusd_features=4,
        expected_eurusd_targets=1,
        expected_eth_features=2,
    )

    root = Path(result["bundle_manifest"]).parent
    manifest = json.loads(Path(result["bundle_manifest"]).read_text())
    assert set(manifest["populations"]) == {"EURUSD", "ETH"}
    assert (root / "tools" / "finalize_bridge.py").is_file()
    assert (root / "tools" / "prepare.py").is_file()
    assert (root / "tools" / "phase1_inventory_orchestrator.py").is_file()
    assert len(list((root / "hosts").glob("*.local.json"))) == 3
    assert verify_bundle(Path(result["bundle_manifest"]))["state"] == "VERIFIED"
    serialized = json.dumps(manifest)
    for forbidden in ("omega", "gamma", "dragon", "password", "api_key"):
        assert forbidden not in serialized.lower()
    adoption = json.loads((root / "adoption" / "ADOPTION.json").read_text())
    assert all(not Path(path).is_absolute() for path in adoption["envelopes"])
