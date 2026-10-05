from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tools.feature_selection_batch_runner import (
    DenominatorMismatch,
    InputIdentityMismatch,
    OuterDataRefused,
    OuterFold,
    Progress,
    cell_population,
    existing_record_matches,
    impute_causal,
    load_admissible_feature_order,
    load_folds,
    load_train_population,
    nested_splits,
    record_path,
    refuse_outer_path,
    verify_input_manifest,
    sha256_file,
)


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


FEATURES = {"batch_001": ("px.a", "px.b"), "batch_002": ("yh.c",), "batch_003": ("rg.d",)}


def _write_inputs(root: Path, *, rows: int = 60, last_time: str = "2023-12-29 22:00:00+00:00", selector_cols: int = 1) -> None:
    times = pd.date_range(end=pd.Timestamp(last_time), periods=rows, freq="h")
    row_id = np.arange(rows)
    rng = np.random.default_rng(0)
    for batch, names in FEATURES.items():
        directory = root / batch
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"t_decision_utc": times, "row_id": row_id})
        for name in names:
            frame[name] = rng.normal(size=rows)
        extra = []
        if batch == "batch_001":
            for index in range(selector_cols):
                frame[f"ev.src{index}"] = rng.normal(size=rows)
                extra.append(f"ev.src{index}")
            frame["q.n5"] = 1.0
        frame.to_parquet(directory / "features_train.parquet", index=False)
        with (directory / "admissible_features.csv").open("w") as stream:
            stream.write("feature_id,role,admissibility\n")
            for name in names:
                stream.write(f"{name},feature,ADMISSIBLE\n")
            for name in extra:
                stream.write(f"{name},feature,ADMISSIBLE_WITH_DECLARED_ASSUMPTION\n")
            if batch == "batch_001":
                stream.write("q.n5,quality_excluded,EXCLUDED_ROLE:quality_excluded\n")
    targets = pd.DataFrame({"t_decision_utc": times, "row_id": row_id})
    targets["Y_s_1h"] = rng.normal(size=rows)
    targets["Y_b_s6"] = np.where(rng.uniform(size=rows) < 0.1, np.nan, rng.integers(-1, 2, size=rows).astype(float))
    targets.to_parquet(root / "batch_001" / "targets_train.parquet", index=False)
    folds = {"folds": [
        {"name": "inner_a", "train_rows": [0, 30], "val_rows": [32, 45], "val_time": [str(times[32]), str(times[44])]},
        {"name": "inner_b", "train_rows": [0, 45], "val_rows": [47, 60], "val_time": [str(times[47]), str(times[59])]},
    ]}
    (root / "batch_001" / "folds.json").write_text(json.dumps(folds))


EXPECTED = {"batch_001": 2, "batch_002": 1, "batch_003": 1}


def test_denominator_is_reconciled_and_selector_sources_are_excluded(tmp_path: Path) -> None:
    _write_inputs(tmp_path, selector_cols=2)
    order, report = load_admissible_feature_order(tmp_path, EXPECTED)
    assert order == ("px.a", "px.b", "yh.c", "rg.d")
    assert report["selector_episode_source"] == 2
    assert report["excluded_role"] == 1
    assert report["total"] == 4 == report["declared_total"]
    with pytest.raises(DenominatorMismatch):
        load_admissible_feature_order(tmp_path, {"batch_001": 3, "batch_002": 1, "batch_003": 1})


def test_rows_at_or_after_train_end_are_refused(tmp_path: Path) -> None:
    _write_inputs(tmp_path, last_time="2024-01-03 00:00:00+00:00")
    order, _ = load_admissible_feature_order(tmp_path, EXPECTED)
    with pytest.raises(OuterDataRefused):
        load_train_population(tmp_path, order)


def test_paths_naming_validation_or_test_are_refused(tmp_path: Path) -> None:
    for name in ("validation_rows", "test_2025", "holdout", "batch_2024"):
        with pytest.raises(OuterDataRefused):
            refuse_outer_path(tmp_path / name / "features_train.parquet", tmp_path)
    with pytest.raises(OuterDataRefused):
        refuse_outer_path(tmp_path / "batch_001" / "features_test.parquet", tmp_path)
    # the operator-declared root above the inputs is not judged (pytest's own tmp dir says "test")
    refuse_outer_path(tmp_path / "batch_001" / "features_train.parquet", tmp_path)
    refuse_outer_path(Path("/x/pytest-of-user/test_case0/batch_001/features_train.parquet"), Path("/x/pytest-of-user/test_case0"))


def test_folds_validating_outside_train_are_refused(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    folds = load_folds(tmp_path, 60)
    assert [fold.name for fold in folds] == ["inner_a", "inner_b"]
    payload = json.loads((tmp_path / "batch_001" / "folds.json").read_text())
    payload["folds"][1]["val_time"][1] = "2024-06-01 00:00:00+00:00"
    (tmp_path / "batch_001" / "folds.json").write_text(json.dumps(payload))
    with pytest.raises(OuterDataRefused):
        load_folds(tmp_path, 60)


def test_input_manifest_identity_is_enforced(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": {"batch_001/folds.json": sha256_file(tmp_path / "batch_001" / "folds.json")}}))
    assert verify_input_manifest(tmp_path, manifest)
    manifest.write_text(json.dumps({"sha256": {"batch_001/folds.json": "0" * 64}}))
    with pytest.raises(InputIdentityMismatch):
        verify_input_manifest(tmp_path, manifest)


def test_causal_imputation_never_reads_the_future() -> None:
    matrix = np.array([[np.nan, 1.0], [np.nan, np.nan], [3.0, 2.0], [np.nan, 5.0]])
    filled, fraction = impute_causal(matrix)
    assert filled[2:, 0].tolist() == [3.0, 3.0]
    assert filled[1, 1] == 1.0 and filled[3, 1] == 5.0
    assert filled[0, 0] == filled[1, 0] == 3.0  # leading gap -> mean of the finite rows handed in
    assert fraction.tolist() == [0.75, 0.25]
    future_changed = matrix.copy()
    future_changed[3, 1] = 99.0
    again, _ = impute_causal(future_changed)
    assert np.array_equal(again[:3, 1], filled[:3, 1])


def test_cell_population_is_fit_rows_with_finite_target_and_identical_across_methods(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    order, _ = load_admissible_feature_order(tmp_path, EXPECTED)
    matrix, targets, row_ids = load_train_population(tmp_path, order)
    fold = OuterFold("inner_b", 0, 45, 47, 60)
    cell = cell_population(matrix, targets, row_ids, order, "Y_b_s6", fold)
    assert cell.n_fit_rows == 45
    assert cell.matrix.shape[0] == cell.target_values.shape[0] == 45 - cell.n_dropped_target_nan
    assert np.all(np.isfinite(cell.target_values)) and np.all(np.isfinite(cell.matrix))
    assert cell.row_ids.max() < 45
    assert set(np.unique(cell.target_values)) <= {-1.0, 0.0, 1.0}
    again = cell_population(matrix, targets, row_ids, order, "Y_b_s6", fold)
    assert np.array_equal(again.matrix, cell.matrix) and np.array_equal(again.target_values, cell.target_values)
    assert cell.nested_splits[-1].validation_end == cell.matrix.shape[0]


def test_nested_splits_stay_chronological_with_a_purge_gap() -> None:
    a, b = nested_splits(1000, gap=10)
    assert a.train_end == 590 and a.validation_start == 600 and a.validation_end == 800
    assert b.train_end == 790 and b.validation_start == 800 and b.validation_end == 1000


def test_progress_counts_failures_and_eta_from_observed_rate(tmp_path: Path) -> None:
    progress = Progress(tmp_path / "progress.json", ("M1", "M2"), ("T",), ("f1", "f2"), (8, 24))
    assert progress.total == 2 * 1 * 2 * 2
    progress.mark({"method": "M1", "target": "T", "fold": "f1", "disposition": "COMPLETE", "wall_seconds": 10.0,
                   "selected_by_k": [[8, ["a"]], [24, ["a"]]], "k_failures": [], "failure_type": None, "failure_message": None})
    progress.mark({"method": "M2", "target": "T", "fold": "f1", "disposition": "FAILED", "wall_seconds": 1.0,
                   "selected_by_k": [], "k_failures": [], "failure_type": "ValueError", "failure_message": "boom"})
    snap = json.loads((tmp_path / "progress.json").read_text())
    assert snap["done"] == 2 and snap["total"] == 8
    assert snap["n_failures"] == 1 and snap["failures"][0]["failure_type"] == "ValueError"
    assert snap["observed_median_method_wall_seconds"] == 10.0
    assert snap["eta_seconds"] == 2 * 10.0  # two method cells remaining at the observed median
    assert snap["per_method"]["M1"] == {"done": 1, "total": 2}


def test_resume_skips_only_records_with_the_same_identity(tmp_path: Path) -> None:
    identity = {"row_digest": "r", "group_digest": "g", "plan_digest": "p"}
    path = record_path(tmp_path, "Y", "f", "RANDOM_K:seed0")
    assert path.name == "RANDOM_K__seed0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "fs_pred_method_record.v1", "identity": identity, "disposition": "COMPLETE"}))
    assert existing_record_matches(path, identity)
    assert not existing_record_matches(path, dict(identity, row_digest="other"))
    path.write_text(json.dumps({"schema": "fs_pred_method_record.v1", "identity": identity, "disposition": "FAILED"}))
    assert not existing_record_matches(path, identity)
