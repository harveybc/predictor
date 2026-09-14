"""The contract, exercised through the loader predictor actually runs.

R1 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`: the
previous round's rules drove `app/column_roles.py` directly, so they proved what the module
does, not what a run does. Everything here calls
`preprocessor_plugins.helpers.load_normalized_csv` with files on disk, which is the function
the pipeline uses. `read_csv(..., index_col=0)` makes the first column the index, so the time
column arrives as the index name rather than as a column — the contract has to hold there too.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from app.column_roles import ColumnRoleError
from preprocessor_plugins.helpers import load_normalized_csv

CONTRACT = {"time": "DATE_TIME", "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "targets": ["CLOSE"], "metadata": ["available_time"],
            "allow_target_as_feature": True}


def csv_at(tmp_path, name, columns, rows=4):
    stamps = [f"2024-01-0{i + 1} 00:00:00" for i in range(rows)]
    body = {column: (stamps if column in ("DATE_TIME", "available_time")
                     else [float(i + 1) for i in range(rows)]) for column in columns}
    path = tmp_path / name
    pd.DataFrame(body).to_csv(path, index=False)
    return str(path)


def config_for(path, **extra):
    """The smallest configuration that reaches the loader with one training file."""
    return dict({"x_train_file": path, "max_steps_train": 0}, **extra)


def test_the_declared_features_reach_the_run_in_the_declared_order(tmp_path):
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "CLOSE", "OPEN", "available_time", "HIGH",
                                      "LOW"])
    config = config_for(path, column_roles=CONTRACT)
    data, _ = load_normalized_csv(config)
    assert list(data["x_train_df"].columns) == ["OPEN", "HIGH", "LOW", "CLOSE"]
    record = config["column_roles_applied"]["x_train_df"]
    assert record["features"] == ["OPEN", "HIGH", "LOW", "CLOSE"]
    assert record["target_is_feature"] is True, "the declared overlap must reach the receipt"
    assert len(record["contract_sha256"]) == 64


def test_an_undeclared_column_stops_the_run_instead_of_being_skipped(tmp_path):
    """A contract refusal must not be swallowed by the loader's per-file `except`."""
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE",
                                      "available_time", "SURPRISE"])
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_normalized_csv(config_for(path, column_roles=CONTRACT))


def test_the_target_overlap_needs_the_declaration_at_the_real_loader(tmp_path):
    """Musashi's first counterexample, through the loader rather than through the module."""
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE",
                                      "available_time"])
    without = {key: value for key, value in CONTRACT.items()
               if key != "allow_target_as_feature"}
    with pytest.raises(ColumnRoleError, match="CLOSE"):
        load_normalized_csv(config_for(path, column_roles=without))


def test_a_contradictory_metadata_feature_stops_the_run(tmp_path):
    """Musashi's second counterexample: numeric metadata must not reach the model."""
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"],
                "metadata": ["available_time"]}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_normalized_csv(config_for(path, column_roles=contract))


def test_a_timestamp_declared_as_a_feature_is_refused_before_the_tensor(tmp_path):
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"], "metadata": []}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_normalized_csv(config_for(path, column_roles=contract))


def test_a_run_without_a_contract_is_refused_at_the_real_loader(tmp_path):
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN"])
    with pytest.raises(ColumnRoleError, match="column_roles"):
        load_normalized_csv(config_for(path))


def test_the_declared_legacy_migration_still_loads_everything(tmp_path):
    path = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "CLOSE"])
    config = config_for(path, column_roles_migration="LEGACY_ALL_COLUMNS_ARE_FEATURES")
    data, _ = load_normalized_csv(config)
    assert list(data["x_train_df"].columns) == ["OPEN", "CLOSE"]
    assert config["column_roles_applied"]["x_train_df"]["migration"] == (
        "LEGACY_ALL_COLUMNS_ARE_FEATURES")


def test_every_file_of_the_run_is_checked_not_only_the_first(tmp_path):
    """A leak in the validation file is a leak: the contract is per file, not per run."""
    clean = csv_at(tmp_path, "x.csv", ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE",
                                       "available_time"])
    dirty = csv_at(tmp_path, "v.csv", ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE",
                                       "available_time", "LEAK"])
    config = config_for(clean, column_roles=CONTRACT, x_validation_file=dirty,
                        max_steps_val=0)
    with pytest.raises(ColumnRoleError, match="LEAK"):
        load_normalized_csv(config)
