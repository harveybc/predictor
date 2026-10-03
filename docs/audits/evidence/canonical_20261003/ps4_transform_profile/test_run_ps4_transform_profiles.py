"""PS4 transform-profile contract. These tests are the specification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_ps4_transform_profiles as ps4  # noqa: E402


def _parquet(path: Path, columns: dict) -> None:
    pq.write_table(pa.table(columns), path)


def _folds(path: Path, folds) -> None:
    path.write_text(json.dumps({"schema": "laneA_folds.v1", "folds": folds}))


def _join(path: Path, ids) -> None:
    lines = ["variant_id,emitted_feature_id,metric"]
    for i, feature in enumerate(ids):
        lines.append(f"v{i},{feature},acf")
    path.write_text("\n".join(lines) + "\n")


def test_wrong_parquet_digest(tmp_path: Path) -> None:
    path = tmp_path / "features_train.parquet"
    _parquet(path, {"tv.wav_d1": [1.0, 2.0]})
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.assert_parquet_digest(path, ps4.REQUIRED_PARQUET_SHA256)
    assert caught.value.code == "DIGEST_MISMATCH"


def test_absent_or_duplicated_emitted_feature(tmp_path: Path) -> None:
    absent = tmp_path / "absent.csv"
    _join(absent, [f"f{i}" for i in range(9)])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.emitted_feature_ids(absent)
    assert caught.value.code == "EMITTED_FEATURE_COUNT"
    duplicated = tmp_path / "dup.csv"
    _join(duplicated, [f"f{i}" for i in range(9)] + ["f0"])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.emitted_feature_ids(duplicated)
    assert caught.value.code == "DUPLICATE_EMITTED_FEATURE"


def test_feature_absent_from_parquet(tmp_path: Path) -> None:
    path = tmp_path / "features_train.parquet"
    _parquet(path, {"other": [1.0, 2.0, 3.0]})
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.assert_columns_present(path, ["tv.wav_d1"])
    assert caught.value.code == "FEATURE_ABSENT_FROM_PARQUET"


def test_fold_boundaries(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    _folds(missing, [{"name": "a"}])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.load_train_folds(missing, 100)
    assert caught.value.code == "FOLD_BOUNDARY_ABSENT"

    overlap = tmp_path / "overlap.json"
    _folds(overlap, [
        {"name": "a", "train_rows": [0, 40]},
        {"name": "b", "train_rows": [20, 60]},
        {"name": "c", "train_rows": [0, 70]},
        {"name": "d", "train_rows": [0, 80]},
        {"name": "e", "train_rows": [0, 90]},
    ])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.load_train_folds(overlap, 100)
    assert caught.value.code == "FOLD_OVERLAP"

    backward = tmp_path / "back.json"
    _folds(backward, [
        {"name": "a", "train_rows": [0, 80]},
        {"name": "b", "train_rows": [0, 40]},
        {"name": "c", "train_rows": [0, 50]},
        {"name": "d", "train_rows": [0, 60]},
        {"name": "e", "train_rows": [0, 70]},
    ])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.load_train_folds(backward, 100)
    assert caught.value.code == "FOLD_NONCHRONOLOGICAL"

    outside = tmp_path / "out.json"
    _folds(outside, [
        {"name": "a", "train_rows": [0, 40]},
        {"name": "b", "train_rows": [0, 50]},
        {"name": "c", "train_rows": [0, 60]},
        {"name": "d", "train_rows": [0, 70]},
        {"name": "e", "train_rows": [0, 120]},
    ])
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.load_train_folds(outside, 100)
    assert caught.value.code == "FOLD_OUT_OF_RANGE"


def test_partition_must_be_train() -> None:
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.train_population([("validation", 0, 10)], 10)
    assert caught.value.code == "PARTITION_NOT_TRAIN"
    assert ps4.train_population([("train", 0, 10)], 10) == [("train", 0, 10)]


def test_rows_at_or_after_train_end_do_not_change_the_fold() -> None:
    column = np.linspace(-1.0, 1.0, 80)
    first = ps4.profile_fold("ds", "tv.wav_d1", column, "inner", 40)
    mutated = column.copy()
    mutated[40:] = 999.0
    second = ps4.profile_fold("ds", "tv.wav_d1", mutated, "inner", 40)
    assert ps4.scientific_bytes(first) == ps4.scientific_bytes(second)
    mutated[10] = -50.0
    third = ps4.profile_fold("ds", "tv.wav_d1", mutated, "inner", 40)
    assert ps4.scientific_bytes(third) != ps4.scientific_bytes(first)


def test_duplicate_output_identity() -> None:
    row = {"feature_id": "tv.wav_d1", "fold": "inner", "metric": "discrete_entropy_bits",
           "estimator": {"name": "discrete_entropy_plugin"}, "status": "COMPLETED", "value": 1.0}
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.assert_unique([row, dict(row)])
    assert caught.value.code == "DUPLICATE_OUTPUT_IDENTITY"


def test_missing_unit() -> None:
    features = [f"f{i}" for i in range(10)]
    folds = [f"k{i}" for i in range(5)]
    rows = [{"feature_id": features[0], "fold": folds[0]}]
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.assert_complete_units(rows, features, folds)
    assert caught.value.code == "MISSING_UNIT"
    assert "f0/k1" in caught.value.detail or "f1/k0" in caught.value.detail


def test_non_finite_completed_value() -> None:
    with pytest.raises(ps4.ProfileRefusal) as caught:
        ps4.assert_finite_completed([
            {"feature_id": "tv.wav_d1", "fold": "inner", "metric": "discrete_entropy_bits",
             "status": "COMPLETED", "value": float("nan")}
        ])
    assert caught.value.code == "NON_FINITE_COMPLETED"


def test_replay_is_byte_identical(tmp_path: Path) -> None:
    features = [f"f{i:02d}" for i in range(10)]
    columns = {name: np.linspace(-1.0, 1.0, 64) + i for i, name in enumerate(features)}
    parquet = tmp_path / "features_train.parquet"
    _parquet(parquet, columns)
    folds = tmp_path / "folds.json"
    _folds(folds, [
        {"name": f"k{i}", "train_rows": [0, 20 + 5 * i]} for i in range(5)
    ])
    join = tmp_path / "join.csv"
    _join(join, features)
    out = tmp_path / "out"
    digest = ps4.sha256_file(parquet)
    ps4.execute(parquet, folds, join, out, expected_digest=digest, dataset_id="fixture")
    snapshots = {
        name: (out / name).read_bytes()
        for name in ("profile_rows.jsonl", "REPORT.json", "input_digests.json")
    }
    ps4.execute(parquet, folds, join, out, expected_digest=digest, dataset_id="fixture")
    for name, body in snapshots.items():
        assert (out / name).read_bytes() == body
    receipts = list((out / "attempts").glob("*.json"))
    assert receipts
    assert b"cpu_seconds" in snapshots["profile_rows.jsonl"]
