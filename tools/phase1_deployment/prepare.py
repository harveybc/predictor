#!/usr/bin/env python3
"""Build reproducible EURUSD/ETH phase-1 bundles without launching live work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PREDICTOR_REVISION = "174e7412"
CAUSAL_REVISION = "5e88fc6"
BATCHES = ("batch_001", "batch_002", "batch_003")
IDENTITY_COLUMNS = ("t_decision_utc", "row_id")
ROW_FAMILIES = (
    "sampling_quality",
    "variable_profiles",
    "information_metrics",
    "pair_relations",
    "causal_evidence",
    "selection_decisions",
)
EURUSD_TARGETS = (
    *((f"Y_s_{hours}h", "Y_s", "short", hours) for hours in (1, 2, 3, 4, 5, 6)),
    *((f"Y_l_{hours}h", "Y_l", "long", hours) for hours in (24, 48, 72, 96, 120, 144)),
    ("Y_b_s6", "Y_b", "barrier", 6),
    ("Y_b_l144", "Y_b", "barrier", 144),
)
EURUSD_HISTORY = (
    "px.logret_24h", "px.logret_120h", "px.ewma_vol_24", "px.ewma_vol_168",
    "cal.hour_sin", "cal.hour_cos", "cal.dow_sin", "cal.dow_cos",
)
EURUSD_PRE_RETURNS = ("px.logret_1h", "px.logret_6h")
EURUSD_LOCATORS = (
    "ev.USD.high.count_24h", "ev.EUR.high.count_24h",
    "ev.USD.high.hours_since", "ev.EUR.high.hours_since",
)
STANDARD_BUNDLE_ROOT = "~/.local/share/predictor/phase1/phase1-eurusd-eth-v1"
STANDARD_STATE_ROOT = "~/.local/state/predictor/phase1"


class DeploymentError(RuntimeError):
    """The requested bundle cannot be produced without guessing."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _copy_verified(source: Path, target: Path, expected: str | None = None) -> str:
    if not source.is_file():
        raise DeploymentError(f"required file is absent: {source}")
    observed = _sha(source)
    if expected is not None and observed != expected:
        raise DeploymentError(f"digest mismatch for {source}: {observed} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha(target) != observed:
        raise DeploymentError(f"copy verification failed for {target}")
    return observed


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise DeploymentError(f"empty inventory: {path}")
    return rows


def _load_batch_rows(
    evidence_root: Path, expected_features: int, excluded_features: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    rows: list[dict[str, str]] = []
    by_batch: dict[str, list[str]] = {}
    seen: dict[str, dict[str, str]] = {}
    excluded_features = excluded_features or set()
    for batch in BATCHES:
        batch_rows = _read_inventory(evidence_root / batch / "admissible_features.csv")
        names: list[str] = []
        for row in batch_rows:
            if row.get("feature_id") in excluded_features:
                continue
            if row.get("role") != "feature":
                continue
            if row.get("admissibility") != "ADMISSIBLE":
                raise DeploymentError(f"{batch}: a role=feature row is not admissible")
            feature = (row.get("feature_id") or "").strip()
            if not feature:
                raise DeploymentError(f"{batch}: feature_id is empty")
            if feature in seen:
                qualifier = "conflicting" if seen[feature] != row else "duplicate"
                raise DeploymentError(f"{qualifier} duplicate feature {feature!r}")
            seen[feature] = row
            rows.append({**row, "batch": batch})
            names.append(feature)
        by_batch[batch] = names
    if len(rows) != expected_features:
        raise DeploymentError(f"EURUSD inventory expected {expected_features}, observed {len(rows)}")
    return rows, by_batch


def _same_identity(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if len(left) != len(right):
        raise DeploymentError(f"{label}: row identity length differs")
    if not left["row_id"].equals(right["row_id"]):
        raise DeploymentError(f"{label}: row identity differs")
    a = pd.to_datetime(left["t_decision_utc"], utc=True)
    b = pd.to_datetime(right["t_decision_utc"], utc=True)
    if not a.equals(b):
        raise DeploymentError(f"{label}: decision timestamps differ")


def _episode_context(evidence_root: Path) -> list[str]:
    path = evidence_root / "batch_002" / "role_overlay_batch_001.json"
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "selector_episode_source_features", "episode_sources",
        "overlay_episode_sources", "excluded_episode_sources",
    ):
        values = document.get(key)
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return values
    text = json.dumps(document)
    candidates = sorted({
        token.strip('"') for token in text.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if token.strip('"').startswith("ev.")
    })
    return candidates


def _artifact_digest(evidence_root: Path, batch: str, name: str) -> str | None:
    path = evidence_root / batch / "digests.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    return (document.get("artifacts_sha256") or {}).get(name)


def _seal_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["row_sha256"] = _digest(result)
    return result


def _envelope(run: dict[str, Any], rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    document = {
        "schema_version": "feature_selection_envelope.v1",
        "run": run,
        "rows": {family: [_seal_row(row) for row in rows.get(family, [])] for family in ROW_FAMILIES},
    }
    document["envelope_sha256"] = _digest(document)
    return document


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)) and np.isfinite(value):
        yield prefix, float(value)
    elif isinstance(value, dict):
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(value[key], name)


def _profile_rows(profile_files: list[Path], features: set[str], population_id: str) -> dict[str, list[dict[str, Any]]]:
    output = {family: [] for family in ROW_FAMILIES}
    observed: set[str] = set()
    for path in profile_files:
        for row in _read_inventory(path):
            feature = row["feature_id"]
            if feature not in features:
                continue
            observed.add(feature)
            try:
                value = json.loads(row.get("value") or "null")
            except json.JSONDecodeError as error:
                raise DeploymentError(f"invalid profile JSON for {feature}: {path}") from error
            family = "sampling_quality" if row["metric"] in {"missingness", "timestamp_gaps"} else "variable_profiles"
            leaves = list(_numeric_leaves(value))
            if not leaves:
                output[family].append({
                    "feature_id": feature, "split": "train", "metric_name": row["metric"],
                    "state": row["state"], "population_id": population_id, "fold": None,
                })
            for suffix, numeric in leaves:
                output[family].append({
                    "feature_id": feature, "split": "train",
                    "metric_name": f"{row['metric']}.{suffix}", "metric_value": numeric,
                    "state": row["state"], "population_id": population_id, "fold": None,
                })
    if observed != features:
        missing = sorted(features - observed)
        raise DeploymentError(f"retained profiles do not cover inventory: {missing[:5]}")
    return output


def _causal_and_decision_rows(cells: list[dict[str, Any]], population_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    causal_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for cell in cells:
        supported = contradicted = 0
        for number, name in enumerate(("rung1", "rung2", "rung3"), start=1):
            evidence = cell.get(name) or {}
            state = str(evidence.get("state") or "NOT_IDENTIFIED")
            supported += state == "SUPPORTED"
            contradicted += state == "CONTRADICTED" and bool(evidence.get("robust"))
            causal_rows.append({
                "feature_id": cell["feature_id"], "target_id": cell["target"],
                "horizon": int(cell["horizon_h"]), "split": "train", "rung": number,
                "estimand": str(evidence.get("estimand") or f"causal_ladder_rung_{number}"),
                "estimator": str(evidence.get("estimator") or "retained_verified_evidence"),
                "state": "NOT_SUPPORTED" if state == "CONTRADICTED" else state,
                "effect": None, "lower": None, "upper": None,
                "support_n": int(evidence.get("effective_rows") or evidence.get("episodes_n") or 0),
                "assumptions": list(map(str, evidence.get("assumptions") or [])),
                "adjustment_set": list(map(str, evidence.get("adjustment_set") or [])),
                "evidence_sha256": _digest(evidence), "population_id": population_id, "fold": None,
            })
        decision = "REJECTED" if contradicted else "SELECTED" if supported else "NEUTRAL"
        rule = (
            "at least one robust CONTRADICTED causal rung" if contradicted
            else "at least one SUPPORTED rung and no robust contradiction" if supported
            else "causal ladder did not identify support or robust contradiction"
        )
        decisions.append({
            "feature_id": cell["feature_id"], "target_id": cell["target"],
            "horizon": int(cell["horizon_h"]), "method": "causal_ladder_global_fdr",
            "score": float(supported - contradicted), "rank": None, "decision": decision,
            "rule": rule, "evidence_sha256": _digest(cell), "population_id": population_id,
        })
    return causal_rows, decisions


def _adopt_eurusd(
    evidence_root: Path, causal_root: Path, output: Path, rows: list[dict[str, str]],
    predictor_revision: str, causal_revision: str, inventory_sha256: str,
    expected_targets: int,
) -> dict[str, Any]:
    digest_path = causal_root / "digests.json"
    causal_path = causal_root / "causal_evidence.jsonl"
    if not digest_path.is_file() or not causal_path.is_file():
        return {"state": "NOT_AVAILABLE", "reason": "retained causal evidence is absent", "envelopes": []}
    digest_document = json.loads(digest_path.read_text(encoding="utf-8"))
    expected = (digest_document.get("artifacts_sha256") or {}).get("causal_evidence.jsonl")
    if not expected or _sha(causal_path) != expected:
        raise DeploymentError("retained causal evidence digest does not reconcile")
    cells = [json.loads(line) for line in causal_path.read_text(encoding="utf-8").splitlines() if line]
    features = {row["feature_id"] for row in rows}
    cell_features = {cell.get("feature_id") for cell in cells}
    if cell_features != features:
        raise DeploymentError("retained causal evidence population does not reconcile")
    pairs = {(cell.get("feature_id"), cell.get("target")) for cell in cells}
    if len(cells) != len(features) * expected_targets or len(pairs) != len(cells):
        raise DeploymentError(
            "retained causal evidence does not contain the complete feature-target denominator"
        )
    profile_files = [evidence_root / "profiles" / f"{batch}.csv" for batch in BATCHES]
    if not all(path.is_file() for path in profile_files):
        profile_files = [evidence_root / batch / "profile_cells.csv" for batch in BATCHES]
    population_id = _digest({"population": "EURUSD", "inventory_sha256": inventory_sha256})
    profiles = _profile_rows(profile_files, features, population_id)
    causal_rows, decisions = _causal_and_decision_rows(cells, population_id)
    grouped_cells: dict[str, list[dict[str, Any]]] = {feature: [] for feature in features}
    grouped_decisions: dict[str, list[dict[str, Any]]] = {feature: [] for feature in features}
    for row in causal_rows:
        grouped_cells[row["feature_id"]].append(row)
    for row in decisions:
        grouped_decisions[row["feature_id"]].append(row)
    by_profile: dict[str, dict[str, list[dict[str, Any]]]] = {
        feature: {family: [] for family in ROW_FAMILIES} for feature in features
    }
    for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
        for row in profiles[family]:
            by_profile[row["feature_id"]][family].append(row)
    envelope_dir = output / "adoption" / "warehouse_envelopes"
    envelope_dir.mkdir(parents=True, exist_ok=True)
    envelope_paths = []
    causal_sha = _sha(causal_path)
    for feature in sorted(features):
        rows_for_feature = by_profile[feature]
        rows_for_feature["causal_evidence"] = grouped_cells[feature]
        rows_for_feature["selection_decisions"] = grouped_decisions[feature]
        run = {
            "run_id": f"phase1-eurusd-adopted:{hashlib.sha256(feature.encode()).hexdigest()[:16]}",
            "campaign_sha256": _digest({"predictor": predictor_revision, "causal": causal_revision}),
            "code_sha256": _digest({"adopter": PREDICTOR_REVISION}),
            "input_sha256": causal_sha,
            "inventory_sha256": inventory_sha256,
            "created_at": "2026-10-05T00:00:00Z",
        }
        path = envelope_dir / f"{hashlib.sha256(feature.encode()).hexdigest()}.json"
        _write_json(path, _envelope(run, rows_for_feature))
        envelope_paths.append(str(path.relative_to(output)))
    report = {
        "schema": "phase1.evidence_adoption.v1", "state": "ADOPTED_VERIFIED_EVIDENCE",
        "population_id": "EURUSD", "feature_count": len(features), "causal_rows": len(cells),
        "source_sha256": causal_sha, "source_digest_manifest_sha256": _sha(digest_path),
        "predictor_revision": predictor_revision, "causal_revision": causal_revision,
        "envelopes": envelope_paths,
    }
    report["adoption_sha256"] = _digest(report)
    _write_json(output / "adoption" / "ADOPTION.json", report)
    return report


def _folds(path: Path) -> list[dict[str, list[int]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for fold in document.get("folds", []):
        result.append({
            "train_rows": fold["train_rows"],
            "validation_rows": fold.get("validation_rows", fold.get("val_rows")),
        })
    if not result or any(item["validation_rows"] is None for item in result):
        raise DeploymentError("folds do not define train and validation row ranges")
    return result


def _target_definitions(targets: Iterable[tuple[str, str, str, int]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "column": name, "family": family, "head": head, "horizon_hours": hours}
        for name, family, head, hours in targets
    ]


def _write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _deployment_document(population: str, target_pack_id: str, target_pack: dict[str, Any], folds: list[dict[str, Any]], campaign_sha: str) -> dict[str, Any]:
    root = STANDARD_BUNDLE_ROOT
    document = {
        "schema": "phase1.column_worker_deployment.v1",
        "version": "phase1-eurusd-eth-v1",
        "output_root": f"{STANDARD_STATE_ROOT}/worker",
        "resources": {
            f"{population.lower()}.train": {
                "resource_id": f"phase1/{population.lower()}/train",
                "path": f"{root}/data/{population.lower()}_features_train.parquet",
                "format": "parquet",
                "timestamp_column": "t_decision_utc",
                "frequency": "1h" if population == "EURUSD" else "4h",
                "calendar": "24x5" if population == "EURUSD" else "24x7",
            }
        },
        "populations": {
            population: {
                "target_pack_id": target_pack_id,
                "resource_key_field": "resource_key",
                "feature_column_field": "feature_column",
                "feature_family_field": "family",
                "feature_clock_field": "clock",
                "target_pack": target_pack,
                "folds": folds,
                "permutations": 200,
                "campaign_sha256": campaign_sha,
                "created_at": "2026-10-05T00:00:00Z",
                "train_period": ["2012-05-01T00:00:00Z", "2024-01-01T00:00:00Z"] if population == "EURUSD" else [None, "2024-01-01T00:00:00Z"],
            }
        },
    }
    document["deployment_sha256"] = _digest(document)
    return document


def _orchestrator_document(population: str, expected: int, target_pack_id: str) -> dict[str, Any]:
    root = STANDARD_BUNDLE_ROOT
    return {
        "schema": "phase1.inventory_orchestrator.v2",
        "phase": "PHASE_1",
        "population": {"id": population, "target_pack": target_pack_id},
        "state_root": f"{STANDARD_STATE_ROOT}/{population.lower()}",
        "inventory": {
            "files": [f"{root}/inventory/{population.lower()}_inventory.csv"],
            "id_field": "feature_id", "row_count_field": "train_finite",
            "bytes_field": "source_bytes", "availability_field": "input_state",
            "available_values": ["AVAILABLE"],
            "include": {"role": "feature", "admissibility": "ADMISSIBLE"},
            "expected_total": expected, "bytes_per_row": 8, "bytes_per_cost_unit": 8,
        },
        "assignment": {"smallest_fraction": 0.2},
        "hosts": [
            {"id": "small-worker", "size_class": "small", "command_prefix": []},
            {"id": "large-worker-a", "size_class": "large", "command_prefix": ["ssh", "phase1-large-a"]},
            {"id": "large-worker-b", "size_class": "large", "command_prefix": ["ssh", "phase1-large-b"]},
        ],
        "worker": {
            "transport": "stdio-json-v1",
            "command": ["feature-selection-column", "--stdio", "--deployment-manifest", f"{root}/worker/{population.lower()}_deployment.json"],
            "timeout_seconds": 21600,
            "environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        },
        "retries": {
            "max_attempts": 3, "retryable_exit_codes": [75, 137, -9],
            "retryable_result_states": ["FAILED"],
            "retryable_failure_classes": ["TIMEOUT", "ADMISSION", "OOM"],
        },
        "warehouse": {
            "submit_url": "http://127.0.0.1:8110/api/v2/feature-selection-envelopes",
            "reconcile_url": "http://127.0.0.1:8110/api/v2/feature-selection-reconcile",
            "timeout_seconds": 30, "token_env": "PHASE1_WAREHOUSE_TOKEN",
            "auth_profile": "data-gov-service-token",
        },
        "finalizer": {
            "command": ["python", f"{root}/tools/finalize_bridge.py", "--population", population,
                        "--plan", "{plan_path}", "--terminals", "{terminals_dir}",
                        "--output", "{finalizer_output_path}"],
            "timeout_seconds": 21600,
        },
        "claims": {"stale_after_seconds": 900},
    }


def _bundle_manifest(output: Path, metadata: dict[str, Any]) -> Path:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "BUNDLE_MANIFEST.json":
            files.append({
                "path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": _sha(path),
            })
    document = {
        "schema": "phase1.deployment_bundle.v1", **metadata, "files": files,
        "standard_install_root": STANDARD_BUNDLE_ROOT,
    }
    document["bundle_sha256"] = _digest(document)
    path = output / "BUNDLE_MANIFEST.json"
    _write_json(path, document)
    return path


def _copy_runtime_tools(output: Path) -> None:
    source_root = Path(__file__).resolve().parent
    for name in ("prepare.py", "finalize_bridge.py"):
        _copy_verified(source_root / name, output / "tools" / name)
    _copy_verified(
        source_root.parent / "phase1_inventory_orchestrator.py",
        output / "tools" / "phase1_inventory_orchestrator.py",
    )


def _write_host_templates(output: Path) -> None:
    roles = (
        ("small-worker", "small"),
        ("large-worker-a", "large"),
        ("large-worker-b", "large"),
    )
    populations = [
        name for name in ("eurusd", "eth")
        if (output / "orchestrator" / f"{name}_config.json").is_file()
    ]
    for host_id, size_class in roles:
        document = {
            "schema": "phase1.host_local_deployment.v1",
            "host_id": host_id,
            "size_class": size_class,
            "bundle_root": STANDARD_BUNDLE_ROOT,
            "state_root": STANDARD_STATE_ROOT,
            "populations": populations,
            "environment": {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "run_command_template": (
                "python ~/.local/share/predictor/phase1/phase1-eurusd-eth-v1/"
                "tools/phase1_inventory_orchestrator.py --config <config> "
                f"run-host --host {host_id}"
            ),
        }
        _write_json(output / "hosts" / f"{host_id}.local.json", document)


def build_eurusd_bundle(
    evidence_root: Path, retained_root: Path, causal_root: Path, output_root: Path,
    *, expected_features: int = 366, predictor_revision: str = PREDICTOR_REVISION,
    causal_revision: str = CAUSAL_REVISION, expected_targets: int = 14,
) -> dict[str, Any]:
    """Build the exact EURUSD deployment inputs and adopt reconciled evidence."""
    evidence_root, retained_root, causal_root, output_root = map(
        lambda path: Path(path).expanduser().resolve(),
        (evidence_root, retained_root, causal_root, output_root),
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise DeploymentError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    context = _episode_context(evidence_root)
    rows, by_batch = _load_batch_rows(evidence_root, expected_features, set(context))
    base_identity: pd.DataFrame | None = None
    frames: list[pd.DataFrame] = []
    all_columns: set[str] = set()
    for batch in BATCHES:
        source = retained_root / batch / "features_train.parquet"
        expected = _artifact_digest(evidence_root, batch, "features_train.parquet")
        if expected is not None and _sha(source) != expected:
            raise DeploymentError(f"digest mismatch for {source}")
        frame = pd.read_parquet(source)
        if not set(IDENTITY_COLUMNS) <= set(frame):
            raise DeploymentError(f"{batch}: row identity columns are absent")
        identity = frame[list(IDENTITY_COLUMNS)].copy()
        if base_identity is None:
            base_identity = identity
        else:
            _same_identity(base_identity, identity, batch)
        selected = by_batch[batch]
        missing = sorted(set(selected) - set(frame.columns))
        if missing:
            raise DeploymentError(f"{batch}: declared feature columns absent: {missing[:5]}")
        extras = [name for name in context if batch == "batch_001" and name in frame and name not in selected]
        data_columns = selected + extras
        duplicated = sorted(set(data_columns) & all_columns)
        if duplicated:
            raise DeploymentError(f"duplicate feature columns across batches: {duplicated[:5]}")
        all_columns.update(data_columns)
        frames.append(frame[data_columns])
    assert base_identity is not None
    merged = pd.concat([base_identity, *frames], axis=1)
    features_path = output_root / "data" / "eurusd_features_train.parquet"
    features_path.parent.mkdir(parents=True)
    merged.to_parquet(features_path, index=False)
    targets_source = retained_root / "batch_001" / "targets_train.parquet"
    targets_expected = _artifact_digest(evidence_root, "batch_001", "targets_train.parquet")
    targets = pd.read_parquet(targets_source)
    _same_identity(base_identity, targets, "EURUSD targets")
    missing_targets = sorted({name for name, *_ in EURUSD_TARGETS} - set(targets.columns))
    if missing_targets:
        raise DeploymentError(f"EURUSD target columns absent: {missing_targets}")
    targets_path = output_root / "data" / "eurusd_targets_train.parquet"
    _copy_verified(targets_source, targets_path, targets_expected)
    folds_source = retained_root / "batch_001" / "folds.json"
    folds_expected = _artifact_digest(evidence_root, "batch_001", "folds.json")
    folds_path = output_root / "data" / "eurusd_folds.json"
    _copy_verified(folds_source, folds_path, folds_expected)
    inventory_rows = []
    for row in rows:
        clock = "KNOWN_IN_ADVANCE" if row["feature_id"].startswith("cal.") else (
            "ASSUMED" if "assum" in row.get("availability_time", "").lower() else "OBSERVED"
        )
        inventory_rows.append({
            **row, "population_id": "EURUSD", "resource_key": "eurusd.train",
            "feature_column": row["feature_id"], "clock": clock,
            "source_bytes": int(row["train_rows"]) * 8, "input_state": "AVAILABLE",
        })
    inventory_path = output_root / "inventory" / "eurusd_inventory.csv"
    _write_inventory_csv(inventory_path, inventory_rows)
    inventory_sha = _digest([
        {"feature_id": row["feature_id"], "row": row} for row in inventory_rows
    ])
    campaign_sha = _digest({
        "population": "EURUSD", "predictor": predictor_revision,
        "causal": causal_revision, "inventory_sha256": inventory_sha,
    })
    target_pack = {
        "id": "EURUSD", "path": f"{STANDARD_BUNDLE_ROOT}/data/eurusd_targets_train.parquet",
        "format": "parquet", "timestamp_column": "t_decision_utc",
        "definitions": _target_definitions(EURUSD_TARGETS),
        "history_columns": list(EURUSD_HISTORY), "pre_return_columns": list(EURUSD_PRE_RETURNS),
        "calendar_locator_columns": list(EURUSD_LOCATORS), "mediator_target": "Y_s_1h",
        "volatility_regime_column": "px.ewma_vol_168", "placebo_outcome_column": "px.logret_24h",
    }
    deployment = _deployment_document("EURUSD", "eurusd-short-long-v1", target_pack, _folds(folds_path), campaign_sha)
    deployment_path = output_root / "worker" / "eurusd_deployment.json"
    _write_json(deployment_path, deployment)
    orchestrator_path = output_root / "orchestrator" / "eurusd_config.json"
    _write_json(orchestrator_path, _orchestrator_document("EURUSD", expected_features, "eurusd-short-long-v1"))
    adoption = _adopt_eurusd(
        evidence_root, causal_root, output_root, rows,
        predictor_revision, causal_revision, inventory_sha, expected_targets,
    )
    finalizer_manifest = {
        "schema": "feature_selection_inventory.v1", "inventory_id": "EURUSD",
        "units": [
            {"unit_id": row["feature_id"],
             "terminal_path": f"{STANDARD_STATE_ROOT}/worker/EURUSD/evidence/units/<unit-key>/terminal.json"}
            for row in rows
        ],
        "run": {"run_id": "phase1-eurusd", "campaign_sha256": campaign_sha,
                "created_at": "2026-10-05T00:00:00Z"},
        "output_dir": f"{STANDARD_STATE_ROOT}/final/EURUSD",
    }
    _write_json(output_root / "finalizer" / "eurusd_inventory.template.json", finalizer_manifest)
    _copy_runtime_tools(output_root)
    _write_host_templates(output_root)
    manifest = _bundle_manifest(output_root, {
        "predictor_revision": predictor_revision, "causal_revision": causal_revision,
        "populations": {"EURUSD": {"features": expected_features, "status": "DEPLOYABLE"}},
    })
    return {
        "population": "EURUSD", "feature_count": expected_features,
        "features_path": str(features_path), "targets_path": str(targets_path),
        "inventory_path": str(inventory_path), "worker_deployment": str(deployment_path),
        "orchestrator_config": str(orchestrator_path), "bundle_manifest": str(manifest),
        "adoption": adoption,
    }


def _eth_targets(frame: pd.DataFrame, train_end: int) -> pd.DataFrame:
    close = pd.to_numeric(frame["CLOSE"], errors="coerce") if "CLOSE" in frame else None
    if close is None:
        one = pd.to_numeric(frame["log_return_1"], errors="coerce")
        close = np.exp(one.fillna(0).cumsum())
    log_close = np.log(close.astype(float))
    result = pd.DataFrame({
        "t_decision_utc": pd.to_datetime(frame["DATE_TIME"], utc=True),
        "row_id": np.arange(len(frame), dtype=np.int64),
    })
    for bars in range(1, 7):
        column = f"Y_eth_{bars * 4}h"
        result[column] = log_close.shift(-bars) - log_close
        result.loc[max(0, train_end - bars):, column] = np.nan
    return result.iloc[:train_end].reset_index(drop=True)


def _eth_folds(train_end: int) -> list[dict[str, list[int]]]:
    validation = max(2, int(train_end * 0.15))
    starts = [max(2, train_end - validation * multiplier) for multiplier in (3, 2, 1)]
    result = []
    for start in starts:
        end = min(train_end, start + validation)
        purge_end = max(1, start - 60)
        if end > start:
            result.append({"train_rows": [0, purge_end], "validation_rows": [start, end]})
    if not result:
        raise DeploymentError("ETH TRAIN is too short for three chronological folds")
    return result


def build_eth_bundle(
    manifest_path: Path, dataset_path: Path, output_root: Path, *, expected_features: int = 83,
    predictor_revision: str = PREDICTOR_REVISION, causal_revision: str = CAUSAL_REVISION,
) -> dict[str, Any]:
    """Build a separate provisional ETH inventory and six-horizon target pack."""
    manifest_path, dataset_path, output_root = map(
        lambda path: Path(path).expanduser().resolve(), (manifest_path, dataset_path, output_root)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_DEVELOPMENT":
        raise DeploymentError("ETH input must be the frozen development manifest")
    if _sha(dataset_path) != (manifest.get("resource") or {}).get("sha256"):
        raise DeploymentError("ETH dataset digest differs from its frozen manifest")
    features = manifest.get("features") or []
    if len(features) != expected_features or len(set(features)) != expected_features:
        raise DeploymentError(f"ETH inventory expected {expected_features} unique features")
    frame = pd.read_csv(dataset_path)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise DeploymentError(f"ETH declared feature columns absent: {missing[:5]}")
    train_range = ((manifest.get("split") or {}).get("train") or {}).get("rows")
    if not isinstance(train_range, list) or len(train_range) != 2 or train_range[0] != 0:
        raise DeploymentError("ETH TRAIN row range is missing or unsupported")
    train_end = int(train_range[1])
    train = frame.iloc[:train_end].copy()
    train.insert(1, "row_id", np.arange(train_end, dtype=np.int64))
    train = train.rename(columns={"DATE_TIME": "t_decision_utc"})
    keep = ["t_decision_utc", "row_id", *features]
    output_root.mkdir(parents=True, exist_ok=True)
    features_path = output_root / "data" / "eth_features_train.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    train[keep].to_parquet(features_path, index=False)
    targets = _eth_targets(frame, train_end)
    targets_path = output_root / "data" / "eth_targets_train.parquet"
    targets.to_parquet(targets_path, index=False)
    folds = _eth_folds(train_end)
    folds_path = output_root / "data" / "eth_folds.json"
    _write_json(folds_path, {"schema": "phase1.eth_folds.v1", "folds": folds})
    inventory_rows = [
        {
            "feature_id": feature, "feature_column": feature, "population_id": "ETH",
            "family": "eth_technical_statistical", "source": "git-pinned-development",
            "resource_key": "eth.train", "frequency": "4h", "clock": "DEVELOPMENT_UNCERTIFIED",
            "role": "feature", "admissibility": "ADMISSIBLE", "train_rows": train_end,
            "train_finite": int(pd.to_numeric(train[feature], errors="coerce").notna().sum()),
            "source_bytes": train_end * 8, "input_state": "AVAILABLE",
            "point_in_time_state": "PROVISIONAL_DEVELOPMENT",
        }
        for feature in features
    ]
    inventory_path = output_root / "inventory" / "eth_inventory.csv"
    _write_inventory_csv(inventory_path, inventory_rows)
    inventory_sha = _digest(inventory_rows)
    campaign_sha = _digest({
        "population": "ETH", "predictor": predictor_revision, "causal": causal_revision,
        "inventory_sha256": inventory_sha, "point_in_time": "PROVISIONAL_DEVELOPMENT",
    })
    definitions = [
        {"name": f"Y_eth_{bars * 4}h", "column": f"Y_eth_{bars * 4}h",
         "family": "Y_eth", "head": "short" if bars == 1 else "long", "horizon_hours": bars * 4}
        for bars in range(1, 7)
    ]
    target_pack = {
        "id": "ETH", "path": f"{STANDARD_BUNDLE_ROOT}/data/eth_targets_train.parquet",
        "format": "parquet", "timestamp_column": "t_decision_utc", "definitions": definitions,
        "history_columns": [], "pre_return_columns": [], "calendar_locator_columns": [],
        "mediator_target": "Y_eth_4h", "volatility_regime_column": None,
        "placebo_outcome_column": None,
    }
    deployment = _deployment_document("ETH", "eth-4h-h1-h6-v1", target_pack, folds, campaign_sha)
    deployment_path = output_root / "worker" / "eth_deployment.json"
    _write_json(deployment_path, deployment)
    orchestrator_path = output_root / "orchestrator" / "eth_config.json"
    _write_json(orchestrator_path, _orchestrator_document("ETH", expected_features, "eth-4h-h1-h6-v1"))
    caveat = {
        "schema": "phase1.eth_point_in_time_caveat.v1",
        "state": "PROVISIONAL_DEVELOPMENT",
        "reason": "git commit time is not an observed publication or receipt clock",
        "forbidden_claims": ["point-in-time admissibility", "production selection", "EURUSD comparability"],
        "source_manifest_sha256": _sha(manifest_path), "dataset_sha256": _sha(dataset_path),
    }
    caveat["caveat_sha256"] = _digest(caveat)
    _write_json(output_root / "governance" / "ETH_POINT_IN_TIME_CAVEAT.json", caveat)
    _copy_runtime_tools(output_root)
    _write_host_templates(output_root)
    bundle = _bundle_manifest(output_root, {
        "predictor_revision": predictor_revision, "causal_revision": causal_revision,
        "populations": {"ETH": {"features": expected_features, "status": "PROVISIONAL_DEVELOPMENT"}},
    })
    return {
        "population": "ETH", "feature_count": expected_features,
        "point_in_time_state": "PROVISIONAL_DEVELOPMENT",
        "features_path": str(features_path), "targets_path": str(targets_path),
        "inventory_path": str(inventory_path), "worker_deployment": str(deployment_path),
        "orchestrator_config": str(orchestrator_path), "bundle_manifest": str(bundle),
    }


def build_combined_bundle(
    evidence_root: Path, retained_root: Path, causal_root: Path,
    eth_manifest_path: Path, eth_dataset_path: Path, output_root: Path, *,
    expected_eurusd_features: int = 366, expected_eurusd_targets: int = 14,
    expected_eth_features: int = 83,
    predictor_revision: str = PREDICTOR_REVISION,
    causal_revision: str = CAUSAL_REVISION,
) -> dict[str, Any]:
    """Build one host-neutral bundle containing both isolated populations."""
    output_root = Path(output_root).expanduser().resolve()
    eurusd = build_eurusd_bundle(
        evidence_root, retained_root, causal_root, output_root,
        expected_features=expected_eurusd_features,
        expected_targets=expected_eurusd_targets,
        predictor_revision=predictor_revision,
        causal_revision=causal_revision,
    )
    eth = build_eth_bundle(
        eth_manifest_path, eth_dataset_path, output_root,
        expected_features=expected_eth_features,
        predictor_revision=predictor_revision,
        causal_revision=causal_revision,
    )
    manifest = _bundle_manifest(output_root, {
        "predictor_revision": predictor_revision,
        "causal_revision": causal_revision,
        "populations": {
            "EURUSD": {"features": expected_eurusd_features, "status": "DEPLOYABLE"},
            "ETH": {"features": expected_eth_features, "status": "PROVISIONAL_DEVELOPMENT"},
        },
    })
    return {
        "bundle_manifest": str(manifest),
        "eurusd": eurusd,
        "eth": eth,
    }


def verify_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = document.pop("bundle_sha256", None)
    if claimed != _digest(document):
        raise DeploymentError("bundle manifest digest is invalid")
    root = manifest_path.parent
    for item in document.get("files", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise DeploymentError(f"bundle member digest mismatch: {item['path']}")
    return {"state": "VERIFIED", "files": len(document.get("files", [])), "bundle_sha256": claimed}


def run_disposable_fixture(output_root: Path) -> dict[str, Any]:
    """Exercise orchestrator-to-worker stdio with three columns and no network."""
    candidates = (
        Path(__file__).resolve().parent / "phase1_inventory_orchestrator.py",
        Path(__file__).resolve().parent.parent / "phase1_inventory_orchestrator.py",
    )
    orchestrator_path = next((path for path in candidates if path.is_file()), None)
    if orchestrator_path is None:
        raise DeploymentError("phase1_inventory_orchestrator.py is absent")
    spec = importlib.util.spec_from_file_location("phase1_inventory_orchestrator", orchestrator_path)
    if spec is None or spec.loader is None:
        raise DeploymentError("cannot load phase-1 orchestrator")
    orchestrator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = orchestrator
    spec.loader.exec_module(orchestrator)

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = output_root / "inventory.csv"
    _write_inventory_csv(inventory, [
        {"feature_id": f"fixture.{index}", "role": "feature", "admissibility": "ADMISSIBLE",
         "train_rows": 8, "source_bytes": 64, "input_state": "AVAILABLE"}
        for index in range(3)
    ])
    worker = output_root / "stdio_worker.py"
    worker.write_text(
        "import hashlib,json,sys\n"
        "r=json.load(sys.stdin)\n"
        "rows={k:[] for k in ('sampling_quality','variable_profiles','information_metrics','pair_relations','causal_evidence','selection_decisions')}\n"
        "run={'run_id':r['feature_id'],'campaign_sha256':'a'*64,'code_sha256':'b'*64,'input_sha256':'c'*64,'inventory_sha256':r['inventory_sha256'],'created_at':'2026-10-05T00:00:00Z'}\n"
        "e={'schema_version':'feature_selection_envelope.v1','run':run,'rows':rows}\n"
        "e['envelope_sha256']=hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
        "print(json.dumps({'schema':'phase1.column_result.v1','feature_id':r['feature_id'],'state':'COMPLETED','request_sha256':r['request_sha256'],'worker_stdio_observed':True,'envelope':e},sort_keys=True))\n",
        encoding="utf-8",
    )
    finalizer = output_root / "finalizer.py"
    finalizer.write_text(
        "import argparse,json,pathlib\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--terminals'); a=p.parse_args()\n"
        "n=len(list(pathlib.Path(a.terminals).glob('*.json')))\n"
        "pathlib.Path(a.output).write_text(json.dumps({'schema':'phase1.finalizer_result.v1','state':'PHASE_1_COMPLETE','terminal_count':n}))\n",
        encoding="utf-8",
    )
    config_doc = {
        "schema": "phase1.inventory_orchestrator.v2", "phase": "PHASE_1",
        "population": {"id": "FIXTURE", "target_pack": "fixture-v1"},
        "state_root": str(output_root / "state"),
        "inventory": {
            "files": [str(inventory)], "id_field": "feature_id", "row_count_field": "train_rows",
            "bytes_field": "source_bytes", "availability_field": "input_state",
            "available_values": ["AVAILABLE"],
            "include": {"role": "feature", "admissibility": "ADMISSIBLE"},
            "expected_total": 3, "bytes_per_row": 8, "bytes_per_cost_unit": 8,
        },
        "assignment": {"smallest_fraction": 0.34},
        "hosts": [
            {"id": "small", "size_class": "small", "command_prefix": []},
            {"id": "large-a", "size_class": "large", "command_prefix": []},
            {"id": "large-b", "size_class": "large", "command_prefix": []},
        ],
        "worker": {"transport": "stdio-json-v1", "command": [sys.executable, str(worker)],
                   "timeout_seconds": 30, "environment": {"CUDA_VISIBLE_DEVICES": ""}},
        "retries": {"max_attempts": 1, "retryable_exit_codes": [75, 137, -9],
                    "retryable_result_states": ["FAILED"],
                    "retryable_failure_classes": ["TIMEOUT", "ADMISSION", "OOM"]},
        "warehouse": {"submit_url": "http://offline.invalid", "reconcile_url": "http://offline.invalid",
                      "timeout_seconds": 1, "token_env": "FIXTURE_TOKEN", "auth_profile": "fixture"},
        "finalizer": {"command": [sys.executable, str(finalizer), "--output", "{finalizer_output_path}",
                                  "--terminals", "{terminals_dir}"], "timeout_seconds": 30},
        "claims": {"stale_after_seconds": 30},
    }
    config_path = output_root / "config.json"
    _write_json(config_path, config_doc)
    config = orchestrator.load_config(config_path)
    plan = orchestrator.build_plan(config)
    network_calls = 0

    def submit(envelope: dict[str, Any], _warehouse: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, "envelope_sha256": envelope["envelope_sha256"]}

    def reconcile(request: dict[str, Any], _warehouse: dict[str, Any]) -> dict[str, Any]:
        unsigned = {
            "schema": "phase1.warehouse_reconciliation.v1", "state": "RECONCILED",
            "plan_sha256": request["plan_sha256"], "expected_count": request["expected_count"],
            "identities_sha256": request["identities_sha256"],
            "authentication_profile": request["authentication_profile"],
        }
        return {**unsigned, "reconciliation_sha256": orchestrator._sha(unsigned)}

    for host in sorted(host["id"] for host in config["hosts"]):
        while orchestrator.run_host_once(config, host, submit=submit)["action"] != "IDLE":
            pass
    final = orchestrator.finalize_if_ready(config, submit=submit, reconcile=reconcile)
    terminals = list((Path(config["state_root"]) / "terminals").glob("*.json"))
    observed_stdio = all(json.loads(path.read_text())["result"].get("worker_stdio_observed") for path in terminals)
    return {
        "state": "COMPLETE" if final["action"] == "PHASE_1_COMPLETE" else final["action"],
        "inventory_total": plan["inventory_total"], "completed": len(terminals),
        "worker_stdio_observed": observed_stdio, "network_calls": network_calls,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    eur = commands.add_parser("prepare-eurusd")
    eur.add_argument("--evidence-root", type=Path, required=True)
    eur.add_argument("--retained-root", type=Path, required=True)
    eur.add_argument("--causal-root", type=Path, required=True)
    eur.add_argument("--output", type=Path, required=True)
    eth = commands.add_parser("prepare-eth")
    eth.add_argument("--manifest", type=Path, required=True)
    eth.add_argument("--dataset", type=Path, required=True)
    eth.add_argument("--output", type=Path, required=True)
    both = commands.add_parser("prepare-all")
    both.add_argument("--eurusd-evidence-root", type=Path, required=True)
    both.add_argument("--eurusd-retained-root", type=Path, required=True)
    both.add_argument("--eurusd-causal-root", type=Path, required=True)
    both.add_argument("--eth-manifest", type=Path, required=True)
    both.add_argument("--eth-dataset", type=Path, required=True)
    both.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-bundle")
    verify.add_argument("--manifest", type=Path, required=True)
    fixture = commands.add_parser("integration-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-eurusd":
            result = build_eurusd_bundle(args.evidence_root, args.retained_root, args.causal_root, args.output)
        elif args.command == "prepare-eth":
            result = build_eth_bundle(args.manifest, args.dataset, args.output)
        elif args.command == "prepare-all":
            result = build_combined_bundle(
                args.eurusd_evidence_root, args.eurusd_retained_root,
                args.eurusd_causal_root, args.eth_manifest, args.eth_dataset,
                args.output,
            )
        elif args.command == "verify-bundle":
            result = verify_bundle(args.manifest)
        else:
            result = run_disposable_fixture(args.output)
    except (DeploymentError, OSError, ValueError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    printable = json.loads(json.dumps(result))
    for value in (printable, printable.get("eurusd", {})):
        adoption = value.get("adoption") if isinstance(value, dict) else None
        if isinstance(adoption, dict) and isinstance(adoption.get("envelopes"), list):
            adoption["envelope_count"] = len(adoption.pop("envelopes"))
    print(json.dumps(printable, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
