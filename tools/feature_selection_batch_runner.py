#!/usr/bin/env python3
"""Unattended batch driver for the TRAIN-only feature-selection campaign (lane FS-PRED).

One runner executes every sealed selector of :mod:`tools.feature_selection_campaign`
over every admissible feature, every target, every inner TRAIN fold and every
sealed K, resumably and with a ``progress.json`` written after each method.
It is a driver around the campaign, not a second orchestrator: all method
dispatch, identity digests, complete-ranking validation and failure retention
live in the campaign.

Population contract (refused otherwise):

* inputs are the lane A ``ps1`` batches (``features_train.parquet`` x3,
  ``targets_train.parquet``, ``folds.json``) whose sha256 must match the relay
  manifest;
* every decision row lies before ``TRAIN_END_UTC``; a file holding a later row,
  or a path naming validation/test, is refused -- there is no parameter through
  which outer rows could be accepted;
* each cell's population is the fold's TRAIN fit rows with a finite target;
  feature NaN are imputed causally inside those rows (forward fill, then the
  fit-row mean for leading gaps) and the imputed fraction is recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.feature_selection_campaign import (
    CampaignDisposition,
    CampaignMethodOutcome,
    FeatureSelectionCampaignPlan,
    compute_campaign_group_digest,
    compute_campaign_plan_digest,
    compute_campaign_row_digest,
    run_feature_selection_campaign,
)
from tools.feature_selector_comparators import RandomTape
from tools.feature_selector_elastic_net import ChronologicalInnerFold
from tools.feature_selector_predictive import _RidgeProbe

SCHEMA_RECORD = "fs_pred_method_record.v1"
SCHEMA_PROGRESS = "fs_pred_progress.v1"
SCHEMA_SELECTOR_SETS = "fs_pred_selector_sets_draft.v1"
TRAIN_END_UTC = datetime(2024, 1, 1, tzinfo=timezone.utc)
FORBIDDEN_PATH = re.compile(r"(^|[^a-z0-9])(validation|valid|test|holdout|2024|2025)([^a-z0-9]|$)", re.IGNORECASE)
DEFAULT_TARGETS = (
    "Y_s_1h", "Y_s_2h", "Y_s_3h", "Y_s_4h", "Y_s_5h", "Y_s_6h",
    "Y_l_24h", "Y_l_48h", "Y_l_72h", "Y_l_96h", "Y_l_120h", "Y_l_144h",
    "Y_b_s6", "Y_b_l144",
)
BATCHES = ("batch_001", "batch_002", "batch_003")
EXPECTED_DENOMINATOR = {"batch_001": 46, "batch_002": 298, "batch_003": 22}
LABEL_PURGE_ROWS = 144
LAMBDA_PATH = tuple(float(v) for v in np.logspace(-1, -3.5, 6))


class OuterDataRefused(RuntimeError):
    """Raised when anything outside TRAIN would be read."""


class DenominatorMismatch(RuntimeError):
    """Raised when the admissible candidate count differs from the declared denominator."""


class InputIdentityMismatch(RuntimeError):
    """Raised when an input file's sha256 differs from the relay manifest."""


@dataclass(frozen=True)
class OuterFold:
    name: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int


@dataclass(frozen=True)
class CellPopulation:
    target: str
    fold: str
    matrix: np.ndarray
    target_values: np.ndarray
    row_ids: np.ndarray
    n_fit_rows: int
    n_dropped_target_nan: int
    imputed_fraction: tuple[tuple[str, float], ...]
    nested_splits: tuple[ChronologicalInnerFold, ...]


# --------------------------------------------------------------------------- inputs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_outer_path(path: Path, input_root: Path | None = None) -> None:
    """Refuse a file whose name, or whose path below the declared input root, names non-TRAIN data."""

    parts = path.relative_to(input_root).parts if input_root is not None else (path.name,)
    for part in parts:
        if FORBIDDEN_PATH.search(part):
            raise OuterDataRefused(f"path component {part!r} names non-TRAIN data: {path}")


def verify_input_manifest(input_root: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    observed: dict[str, str] = {}
    for relative, expected in sorted(manifest["sha256"].items()):
        actual = sha256_file(input_root / relative)
        observed[relative] = actual
        if actual != expected:
            raise InputIdentityMismatch(f"{relative}: expected {expected}, observed {actual}")
    return observed


def load_admissible_feature_order(
    input_root: Path, expected: dict[str, int] | None = None
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return the 366 model-input ids in batch/csv order and the reconciliation report."""

    expected = expected if expected is not None else EXPECTED_DENOMINATOR
    order: list[str] = []
    report: dict[str, Any] = {"per_batch": {}, "selector_episode_source": 0, "excluded_role": 0}
    for batch in BATCHES:
        path = input_root / batch / "admissible_features.csv"
        refuse_outer_path(path, input_root)
        count = 0
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["role"] == "feature" and row["admissibility"] == "ADMISSIBLE":
                    order.append(row["feature_id"])
                    count += 1
                elif row["role"] == "feature":
                    report["selector_episode_source"] += 1
                else:
                    report["excluded_role"] += 1
        report["per_batch"][batch] = count
        if batch in expected and expected[batch] != count:
            raise DenominatorMismatch(f"{batch}: declared {expected[batch]} admissible, observed {count}")
    if len(set(order)) != len(order):
        raise DenominatorMismatch("duplicate feature ids across batches")
    report["total"] = len(order)
    report["declared_total"] = sum(expected.values())
    if report["total"] != report["declared_total"]:
        raise DenominatorMismatch(f"declared {report['declared_total']} admissible, observed {report['total']}")
    return tuple(order), report


def load_train_population(input_root: Path, feature_order: tuple[str, ...]):
    """Load the joined TRAIN feature matrix (NaN allowed), targets and decision times."""

    import pandas as pd

    frames = []
    row_ids = None
    for batch in BATCHES:
        path = input_root / batch / "features_train.parquet"
        refuse_outer_path(path, input_root)
        frame = pd.read_parquet(path)
        _refuse_outer_rows(frame["t_decision_utc"], path)
        if row_ids is None:
            row_ids = frame["row_id"].to_numpy()
        elif not np.array_equal(row_ids, frame["row_id"].to_numpy()):
            raise RuntimeError(f"{batch}: row_id does not align with batch_001")
        frames.append(frame.drop(columns=["t_decision_utc", "row_id"]))
    joined = pd.concat(frames, axis=1)
    missing = [name for name in feature_order if name not in joined.columns]
    if missing:
        raise DenominatorMismatch(f"admissible ids absent from parquet: {missing[:5]}")
    matrix = joined.loc[:, list(feature_order)].to_numpy(dtype=np.float64)
    targets_path = input_root / "batch_001" / "targets_train.parquet"
    refuse_outer_path(targets_path, input_root)
    targets = pd.read_parquet(targets_path)
    _refuse_outer_rows(targets["t_decision_utc"], targets_path)
    if not np.array_equal(targets["row_id"].to_numpy(), row_ids):
        raise RuntimeError("targets row_id does not align with features")
    return matrix, targets, row_ids


def _refuse_outer_rows(times, path: Path) -> None:
    import pandas as pd

    stamps = pd.to_datetime(times, utc=True)
    if len(stamps) == 0:
        raise OuterDataRefused(f"{path} holds no rows")
    if stamps.max() >= TRAIN_END_UTC:
        raise OuterDataRefused(f"{path} holds rows at or after {TRAIN_END_UTC.isoformat()}")


def load_folds(input_root: Path, n_rows: int) -> tuple[OuterFold, ...]:
    path = input_root / "batch_001" / "folds.json"
    payload = json.loads(path.read_text())
    folds = []
    for item in payload["folds"]:
        val_end_time = datetime.fromisoformat(item["val_time"][1])
        if val_end_time >= TRAIN_END_UTC:
            raise OuterDataRefused(f"fold {item['name']} validates at or after TRAIN end")
        fold = OuterFold(
            item["name"], int(item["train_rows"][0]), int(item["train_rows"][1]),
            int(item["val_rows"][0]), int(item["val_rows"][1]),
        )
        if fold.train_end > fold.val_start or fold.val_end > n_rows:
            raise RuntimeError(f"fold {fold.name} boundaries are inconsistent with TRAIN rows")
        folds.append(fold)
    return tuple(folds)


def load_fold_clusters(path: Path, feature_order: tuple[str, ...]) -> dict[str, tuple[tuple[str, str], ...]]:
    payload = json.loads(path.read_text())
    result = {}
    for fold, mapping in payload["folds"].items():
        missing = [name for name in feature_order if name not in mapping]
        if missing:
            raise RuntimeError(f"fold clusters {fold} miss {len(missing)} features")
        result[fold] = tuple((name, str(mapping[name])) for name in feature_order)
    return result


# --------------------------------------------------------------------------- population


def impute_causal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill each column using only earlier rows, then fill leading NaN with the column mean.

    The column mean is computed on the rows handed in (the fold's fit rows), so
    no value outside those rows influences the result.  Returns the imputed
    matrix and the imputed fraction per column.
    """

    filled = np.array(matrix, dtype=np.float64, copy=True)
    missing = ~np.isfinite(filled)
    rows, cols = filled.shape
    index = np.where(missing, 0, np.arange(rows)[:, None])
    np.maximum.accumulate(index, axis=0, out=index)
    filled = filled[index, np.arange(cols)[None, :]]
    still = ~np.isfinite(filled)
    if still.any():
        finite = np.isfinite(matrix)
        counts = finite.sum(axis=0)
        sums = np.where(finite, matrix, 0.0).sum(axis=0)
        # A column with no finite value in these rows becomes the constant 0.0;
        # the caller records it as unavailable in the fit rows.
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        filled = np.where(still, means[None, :], filled)
    return filled, missing.mean(axis=0)


def nested_splits(n_rows: int, gap: int | None = None) -> tuple[ChronologicalInnerFold, ...]:
    """Two expanding chronological splits inside the fit rows; the last one is the predictive nested split.

    The purge gap between nested train and nested validation is the 144-row
    label support at real population sizes and shrinks only for populations
    too small to hold it (never below zero rows).
    """

    if gap is None:
        gap = min(LABEL_PURGE_ROWS, int(0.02 * n_rows))
    a_val_start, b_val_start = int(0.6 * n_rows), int(0.8 * n_rows)
    return (
        ChronologicalInnerFold("nested_a", 0, a_val_start - gap, a_val_start, b_val_start),
        ChronologicalInnerFold("nested_b", 0, b_val_start - gap, b_val_start, n_rows),
    )


def cell_population(
    matrix: np.ndarray, targets, row_ids: np.ndarray, feature_order: tuple[str, ...], target: str, fold: OuterFold
) -> CellPopulation:
    fit_x = matrix[fold.train_start : fold.train_end]
    fit_y = targets[target].to_numpy(dtype=np.float64)[fold.train_start : fold.train_end]
    fit_ids = row_ids[fold.train_start : fold.train_end]
    imputed, fraction = impute_causal(fit_x)
    keep = np.isfinite(fit_y)
    return CellPopulation(
        target=target,
        fold=fold.name,
        matrix=np.ascontiguousarray(imputed[keep]),
        target_values=np.ascontiguousarray(fit_y[keep]),
        row_ids=fit_ids[keep],
        n_fit_rows=int(fit_x.shape[0]),
        n_dropped_target_nan=int((~keep).sum()),
        imputed_fraction=tuple((name, float(value)) for name, value in zip(feature_order, fraction)),
        nested_splits=nested_splits(int(keep.sum())),
    )


def univariate_inner_gains(population: CellPopulation, feature_order: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    probe = _RidgeProbe(population.matrix, population.target_values, population.nested_splits[-1], 1e-3)
    return tuple((name, probe.naive_mae - probe.validation_mae((index,))) for index, name in enumerate(feature_order))


def build_plan(
    feature_order: tuple[str, ...],
    membership: tuple[tuple[str, str], ...],
    gains: tuple[tuple[str, float], ...],
    *,
    k: int,
    sensitivity_ks: tuple[int, ...],
    seed: int,
    methods: tuple[str, ...] | None,
    chronoepilogi_implementation: str,
    extratrees: dict[str, Any] | None = None,
) -> FeatureSelectionCampaignPlan:
    return FeatureSelectionCampaignPlan(
        subset_k=k,
        random_tapes=(RandomTape(f"seed{seed}", seed),),
        elastic_lambda_path=LAMBDA_PATH,
        elastic_max_iter=5_000,
        elastic_tolerance=1e-6,
        redundancy_inner_fold_gains=gains,
        causal_annotations=tuple((name, "NOT_IDENTIFIED") for name in feature_order),
        sensitivity_ks=sensitivity_ks,
        seed=seed,
        group_membership=membership,
        extratrees=dict(extratrees or {"n_estimators": 100, "max_features": 0.3, "min_samples_leaf": 20, "block_length": 168, "n_jobs": 4}),
        sequential_budget=max((k,) + tuple(sensitivity_ks)),
        group_gate_null_shifts=(720, 1440, 2160, 2880, 3600),
        chronoepilogi={"implementation": chronoepilogi_implementation},
        enabled_methods=methods,
    )


# --------------------------------------------------------------------------- records


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value") and hasattr(value, "name") and not isinstance(value, (int, float)):
        return value.value
    return value


def write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=1, default=str))
    os.replace(tmp, path)


def record_path(output_root: Path, target: str, fold: str, method: str) -> Path:
    return output_root / "cells" / target / fold / f"{method.replace(':', '__')}.json"


def population_record(population: CellPopulation, identity: dict[str, str], feature_order: tuple[str, ...]) -> dict[str, Any]:
    """One per-cell record of the exact population every method of the cell shares."""

    imputed = [(name, fraction) for name, fraction in population.imputed_fraction if fraction > 0.0]
    unavailable = [name for name, fraction in population.imputed_fraction if fraction >= 1.0]
    return {
        "schema": "fs_pred_cell_population.v1",
        "target": population.target,
        "fold": population.fold,
        "identity": identity,
        "rows": int(population.target_values.shape[0]),
        "fit_rows": population.n_fit_rows,
        "dropped_target_nan": population.n_dropped_target_nan,
        "first_row_id": int(population.row_ids[0]),
        "last_row_id": int(population.row_ids[-1]),
        "n_features": len(feature_order),
        "nested_splits": [_jsonable(split) for split in population.nested_splits],
        "imputation": "causal forward fill within fit rows, then fit-row mean for leading gaps; a feature with no finite fit-row value becomes a constant (recorded as unavailable)",
        "n_features_imputed": len(imputed),
        "imputed_fraction": imputed,
        "unavailable_in_fit_rows": unavailable,
        "target_summary": {
            "mean": float(np.mean(population.target_values)),
            "std": float(np.std(population.target_values)),
            "min": float(np.min(population.target_values)),
            "max": float(np.max(population.target_values)),
        },
    }


def method_record(
    outcome: CampaignMethodOutcome,
    *,
    identity: dict[str, str],
    population: CellPopulation,
    sealed_ks: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SCHEMA_RECORD,
        "method": outcome.outcome_id,
        "target": population.target,
        "fold": population.fold,
        "seed": seed,
        "sealed_ks": list(sealed_ks),
        "disposition": outcome.disposition.value,
        "failure_type": outcome.failure_type,
        "failure_message": outcome.failure_message,
        "wall_seconds": outcome.wall_seconds,
        "identity": identity,
        "population": {
            "rows": int(population.target_values.shape[0]),
            "fit_rows": population.n_fit_rows,
            "dropped_target_nan": population.n_dropped_target_nan,
            "first_row_id": int(population.row_ids[0]),
            "last_row_id": int(population.row_ids[-1]),
            "nested_splits": [_jsonable(split) for split in population.nested_splits],
        },
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    result = outcome.result
    if result is not None:
        payload = _jsonable(result)
        record["implementation"] = payload.get("implementation", payload.get("penalty", "deterministic"))
        record["full_ranking"] = payload["full_ranking"]
        # Elastic-net results carry coefficient norms instead of a scores tuple.
        record["scores"] = payload.get("scores") or payload.get("coefficient_norms") or [
            [group, 0.0] for group in payload["full_ranking"]
        ]
        record["selected_by_k"] = payload.get("selected_by_k", [])
        record["k_failures"] = payload.get("k_failures", [])
        record["result"] = payload
    record["record_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in record.items() if k != "written_utc"}, sort_keys=True, default=str).encode()
    ).hexdigest()
    return record


def existing_record_matches(path: Path, identity: dict[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return record.get("schema") == SCHEMA_RECORD and record.get("identity") == identity and (
        record.get("disposition") == CampaignDisposition.COMPLETE.value
    )


# --------------------------------------------------------------------------- progress


class Progress:
    """done/total per method x target x fold x K with retained failures and an observed-rate ETA."""

    def __init__(self, path: Path, methods: tuple[str, ...], targets: tuple[str, ...], folds: tuple[str, ...], ks: tuple[int, ...]):
        self.path = path
        self.methods, self.targets, self.folds, self.ks = methods, targets, folds, ks
        self.done: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []
        self.walls: list[float] = []
        self.started = time.time()

    def preload(self, output_root: Path) -> int:
        """Count every record already on disk for the whole plan, so chunked runs report one progress."""

        loaded = 0
        for path in sorted(output_root.glob("cells/*/*/*.json")):
            if path.name == "population.json":
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get("schema") != SCHEMA_RECORD or record.get("method") not in self.methods:
                continue
            if record.get("target") not in self.targets or record.get("fold") not in self.folds:
                continue
            self._mark(record, skipped=True, write=False)
            loaded += 1
        self.write()
        return loaded

    @property
    def total_method_cells(self) -> int:
        return len(self.methods) * len(self.targets) * len(self.folds)

    @property
    def total(self) -> int:
        return self.total_method_cells * len(self.ks)

    def mark(self, record: dict[str, Any], *, skipped: bool = False) -> None:
        self._mark(record, skipped=skipped, write=True)

    def _mark(self, record: dict[str, Any], *, skipped: bool, write: bool) -> None:
        key = f"{record['method']}|{record['target']}|{record['fold']}"
        complete = record["disposition"] == CampaignDisposition.COMPLETE.value
        k_done = [int(k) for k, _ in record.get("selected_by_k", [])] if complete else []
        if key in self.done:
            # a re-run of a cell replaces its earlier failure entries
            self.failures = [f for f in self.failures if (f["method"], f["target"], f["fold"]) != (record["method"], record["target"], record["fold"])]
        self.done[key] = {"complete": complete, "ks_done": k_done, "skipped": skipped}
        if complete and "wall_seconds" in record:
            # observed walls from every record (skipped ones were real runs too)
            self.walls.append(float(record["wall_seconds"]))
        if not complete:
            self.failures.append({
                "method": record["method"], "target": record["target"], "fold": record["fold"],
                "failure_type": record["failure_type"], "failure_message": record["failure_message"],
            })
        for k, reason in record.get("k_failures", []):
            self.failures.append({
                "method": record["method"], "target": record["target"], "fold": record["fold"],
                "failure_type": "K_SENSITIVITY", "failure_message": f"K={k}: {reason}",
            })
        if write:
            self.write()

    def snapshot(self) -> dict[str, Any]:
        done_cells = sum(len(item["ks_done"]) for item in self.done.values())
        done_method_cells = sum(1 for item in self.done.values() if item["complete"])
        remaining_method_cells = self.total_method_cells - len(self.done)
        median = statistics.median(self.walls) if self.walls else None
        eta_seconds = remaining_method_cells * median if median is not None else None
        per_target = {}
        for target in self.targets:
            cells = [item for key, item in self.done.items() if key.split("|")[1] == target]
            per_target[target] = {
                "method_cells_done": sum(1 for item in cells if item["complete"]),
                "method_cells_total": len(self.methods) * len(self.folds),
            }
        per_method = {}
        for method in self.methods:
            cells = [item for key, item in self.done.items() if key.split("|")[0] == method]
            per_method[method] = {
                "done": sum(1 for item in cells if item["complete"]),
                "total": len(self.targets) * len(self.folds),
            }
        return {
            "schema": SCHEMA_PROGRESS,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "grain": "method x target x fold x K",
            "done": done_cells,
            "total": self.total,
            "method_cells_done": done_method_cells,
            "method_cells_total": self.total_method_cells,
            "methods": list(self.methods),
            "targets": list(self.targets),
            "folds": list(self.folds),
            "sealed_ks": list(self.ks),
            "per_target": per_target,
            "per_method": per_method,
            "failures": self.failures,
            "n_failures": len(self.failures),
            "observed_median_method_wall_seconds": median,
            "observed_p90_method_wall_seconds": (sorted(self.walls)[int(0.9 * (len(self.walls) - 1))] if self.walls else None),
            "observed_method_cells_timed": len(self.walls),
            "workers": 1,
            "eta_seconds": eta_seconds,
            "eta_utc": (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat() if eta_seconds is not None else None,
            "runner_started_utc": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
        }

    def write(self) -> None:
        write_atomic(self.path, self.snapshot())


# --------------------------------------------------------------------------- aggregation


def aggregate_target(output_root: Path, target: str, folds: tuple[str, ...], methods: tuple[str, ...]) -> None:
    """Write rankings/selections parquet and the selector_sets draft entry for one target."""

    import pandas as pd

    ranking_rows, selection_rows, sets = [], [], {}
    for method in methods:
        per_fold = {}
        for fold in folds:
            path = record_path(output_root, target, fold, method)
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["disposition"] != CampaignDisposition.COMPLETE.value:
                sets.setdefault(method, {})[fold] = {"disposition": record["disposition"], "failure": record["failure_message"]}
                continue
            for rank, (feature, score) in enumerate(record["scores"], start=1):
                ranking_rows.append({"method": method, "target": target, "fold": fold, "rank": rank, "feature": feature, "score": score})
            by_k = {}
            for k, selection in record["selected_by_k"]:
                by_k[int(k)] = list(selection)
                for position, feature in enumerate(selection, start=1):
                    selection_rows.append({"method": method, "target": target, "fold": fold, "K": int(k), "position": position, "feature": feature})
            per_fold[fold] = by_k
            sets.setdefault(method, {})[fold] = {
                "disposition": "COMPLETE",
                "selected_by_k": {str(k): v for k, v in by_k.items()},
                "k_failures": record["k_failures"],
                "identity": record["identity"],
                "record_sha256": record["record_sha256"],
                "implementation": record.get("implementation"),
                "wall_seconds": record["wall_seconds"],
                "population_rows": record["population"]["rows"],
            }
        if per_fold:
            stability = {}
            ks = sorted({k for by_k in per_fold.values() for k in by_k})
            for k in ks:
                selections = [set(by_k[k]) for by_k in per_fold.values() if k in by_k]
                pairs = [(a, b) for i, a in enumerate(selections) for b in selections[i + 1 :]]
                jaccard = [len(a & b) / len(a | b) for a, b in pairs if a | b]
                stability[str(k)] = {"mean_pairwise_jaccard": (sum(jaccard) / len(jaccard)) if jaccard else None, "n_folds": len(selections)}
            sets[method]["cross_fold_stability"] = stability
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if ranking_rows:
        pd.DataFrame(ranking_rows).to_parquet(tables / f"rankings_{target}.parquet", index=False)
        pd.DataFrame(selection_rows).to_parquet(tables / f"selections_{target}.parquet", index=False)
    draft_path = output_root / "selector_sets.json"
    draft = json.loads(draft_path.read_text()) if draft_path.exists() else {"schema": SCHEMA_SELECTOR_SETS, "targets": {}}
    draft["targets"][target] = sets
    draft["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_atomic(draft_path, draft)


# --------------------------------------------------------------------------- main loop


def run_batch(args: argparse.Namespace) -> int:
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    observed = verify_input_manifest(input_root, Path(args.input_manifest))
    feature_order, denominator = load_admissible_feature_order(input_root)
    matrix, targets, row_ids = load_train_population(input_root, feature_order)
    outer_folds = load_folds(input_root, matrix.shape[0])
    clusters = load_fold_clusters(Path(args.clusters), feature_order)
    target_names = tuple(args.targets.split(",")) if args.targets else DEFAULT_TARGETS
    fold_names = tuple(args.folds.split(",")) if args.folds else tuple(fold.name for fold in outer_folds)
    # progress denominators always cover the WHOLE plan (every target, every fold), not this chunk
    plan_targets = tuple(args.plan_targets.split(",")) if args.plan_targets else DEFAULT_TARGETS
    plan_folds = tuple(fold.name for fold in outer_folds)
    sensitivity_ks = tuple(int(k) for k in args.sensitivity_ks.split(",")) if args.sensitivity_ks else ()
    methods = tuple(args.methods.split(",")) if args.methods else None
    probe_plan = build_plan(
        feature_order, clusters[fold_names[0]], tuple((name, 0.0) for name in feature_order),
        k=args.k, sensitivity_ks=sensitivity_ks, seed=args.seed, methods=methods,
        chronoepilogi_implementation=args.chronoepilogi,
    )
    method_ids = probe_plan.method_ids
    progress = Progress(output_root / "progress.json", method_ids, plan_targets, plan_folds, probe_plan.sealed_ks)
    progress.preload(output_root)
    write_atomic(output_root / "run_contract.json", {
        "plan_method_cells": len(method_ids) * len(plan_targets) * len(plan_folds),
        "plan_k_cells": len(method_ids) * len(plan_targets) * len(plan_folds) * len(probe_plan.sealed_ks),
        "schema": "fs_pred_run_contract.v1",
        "input_sha256": observed,
        "denominator": denominator,
        "feature_order_sha256": hashlib.sha256(json.dumps(feature_order).encode()).hexdigest(),
        "n_features": len(feature_order),
        "train_rows": int(matrix.shape[0]),
        "train_end_utc_exclusive": TRAIN_END_UTC.isoformat(),
        "targets": list(plan_targets),
        "chunk_targets": list(target_names),
        "folds": [asdict(fold) for fold in outer_folds],
        "k_primary": args.k,
        "sealed_ks": list(probe_plan.sealed_ks),
        "seed": args.seed,
        "methods": list(method_ids),
        "chronoepilogi_implementation": args.chronoepilogi,
        "imputation": "causal forward fill within fit rows, then fit-row mean for leading gaps; fraction recorded per cell",
        "barrier_targets": "codes -1/0/+1 (SL/timeout/TP) used as numeric; NaN (ambiguous/censored) rows dropped",
    })
    progress.write()
    for target in target_names:
        if target not in targets.columns:
            raise RuntimeError(f"target {target} not in targets_train.parquet")
        for fold in outer_folds:
            if fold.name not in fold_names:
                continue
            population = cell_population(matrix, targets, row_ids, feature_order, target, fold)
            gains = univariate_inner_gains(population, feature_order)
            plan = build_plan(
                feature_order, clusters[fold.name], gains,
                k=args.k, sensitivity_ks=sensitivity_ks, seed=args.seed, methods=methods,
                chronoepilogi_implementation=args.chronoepilogi,
            )
            identity = {
                "row_digest": compute_campaign_row_digest(population.matrix, population.target_values),
                "group_digest": compute_campaign_group_digest(feature_order),
                "plan_digest": compute_campaign_plan_digest(plan, population.nested_splits),
            }
            write_atomic(output_root / "cells" / target / fold.name / "population.json",
                         population_record(population, identity, feature_order))
            pending = []
            for method in plan.method_ids:
                path = record_path(output_root, target, fold.name, method)
                if existing_record_matches(path, identity):
                    progress.mark(json.loads(path.read_text()), skipped=True)
                else:
                    pending.append(method)
            if not pending:
                continue
            chunk_plan = build_plan(
                feature_order, clusters[fold.name], gains,
                k=args.k, sensitivity_ks=sensitivity_ks, seed=args.seed, methods=tuple(pending),
                chronoepilogi_implementation=args.chronoepilogi,
            )

            def persist(outcome: CampaignMethodOutcome) -> None:
                record = method_record(outcome, identity=identity, population=population, sealed_ks=plan.sealed_ks, seed=args.seed)
                write_atomic(record_path(output_root, target, fold.name, outcome.outcome_id), record)
                progress.mark(record)
                print(f"{datetime.now(timezone.utc).isoformat()} {target} {fold.name} {outcome.outcome_id} {outcome.disposition.value} {outcome.wall_seconds:.1f}s", flush=True)

            run_feature_selection_campaign(
                population.matrix, population.target_values, feature_order, population.nested_splits, chunk_plan,
                expected_row_digest=identity["row_digest"],
                expected_group_digest=identity["group_digest"],
                expected_plan_digest=identity["plan_digest"],
                on_outcome=persist,
            )
        aggregate_target(output_root, target, fold_names, method_ids)
    progress.write()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, help="directory holding ps1 batch_001..003 (TRAIN only)")
    parser.add_argument("--input-manifest", required=True, help="JSON {sha256: {relative_path: digest}} from the relay")
    parser.add_argument("--clusters", required=True, help="fold_clusters.json built from lane B evidence")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", default="", help="comma list for THIS chunk; default all 14")
    parser.add_argument("--plan-targets", default="", help="comma list defining progress denominators; default all 14")
    parser.add_argument("--folds", default="", help="comma list of inner fold names; default all 5")
    parser.add_argument("--methods", default="", help="comma list of outcome ids; default every method")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--sensitivity-ks", default="8,16,24,32,48")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chronoepilogi", default="auto", choices=("auto", "official", "adapter"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
