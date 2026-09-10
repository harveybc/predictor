import copy
import json

import pytest

from olap.crispdm_inventory import (
    build_inventory,
    canonical_json_bytes,
    resolve_dataset_path,
    sha256_bytes,
)
from olap.information_schema import validate_inventory_document


def _registry(relative_path="data.csv"):
    return {
        "schema": "predictor.crispdm_dataset_registry.v1",
        "scope": {"crisp_dm_phase": "DATA_UNDERSTANDING"},
        "datasets": [{
            "dataset_id": "public.fixture.v1",
            "source_class": "public",
            "domain": "fixture",
            "provider": "test",
            "version": "v1",
            "license_id": "CC0-1.0",
            "availability_policy": "STATIC_PUBLICATION",
            "exposure_status": "DEVELOPMENT_FIXTURE",
            "root_id": "fixture",
            "relative_path": relative_path,
            "format": "csv",
            "timestamp_column": "DATE_TIME",
            "timestamp_semantics": "OBSERVATION_TIME_UTC",
            "default_variable_role": "input_candidate",
            "default_unit": "dimensionless",
            "variable_roles": {"DATE_TIME": "timestamp", "target": "target_candidate"},
            "units": {"DATE_TIME": "utc_timestamp"},
        }],
    }


def test_inventory_is_deterministic_and_profiles_variables(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text(
        "DATE_TIME,x,target\n"
        "2026-01-01T00:00:00Z,1,0\n"
        "2026-01-01T01:00:00Z,2,1\n"
        "2026-01-01T02:00:00Z,,1\n",
        encoding="utf-8",
    )
    roots = {"fixture": tmp_path.resolve()}

    first = build_inventory(_registry(), roots, "2026-09-10T00:00:00Z")
    second = build_inventory(_registry(), roots, "2026-09-10T00:00:00Z")

    assert first == second
    assert first["summary"] == {
        "dataset_count": 1,
        "variable_count": 3,
        "row_count": 3,
        "status_counts": {"PROFILED_COMPLETE": 1},
    }
    dataset = first["datasets"][0]
    assert dataset["time_profile"]["median_sampling_seconds"] == 3600.0
    x_profile = next(item for item in dataset["variables"] if item["name"] == "x")
    assert x_profile["missing_count"] == 1
    assert x_profile["finite_count"] == 2
    validate_inventory_document(first)


def test_inventory_digest_changes_when_physical_data_changes(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("DATE_TIME,x\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    roots = {"fixture": tmp_path.resolve()}
    first = build_inventory(_registry(), roots, "2026-09-10T00:00:00Z")

    data.write_text("DATE_TIME,x\n2026-01-01T00:00:00Z,2\n", encoding="utf-8")
    second = build_inventory(_registry(), roots, "2026-09-10T00:00:00Z")

    assert first["inventory_sha256"] != second["inventory_sha256"]
    assert first["datasets"][0]["physical_sha256"] != second["datasets"][0]["physical_sha256"]


def test_undeclared_metadata_is_visible_instead_of_inferred(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("DATE_TIME,x\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    registry = _registry()
    registry["datasets"][0]["license_id"] = "UNDECLARED"
    registry["datasets"][0]["availability_policy"] = "UNDECLARED"
    registry["datasets"][0]["default_unit"] = "UNDECLARED"

    inventory = build_inventory(registry, {"fixture": tmp_path.resolve()}, "2026-09-10T00:00:00Z")

    dataset = inventory["datasets"][0]
    assert dataset["profile_status"] == "PROFILED_WITH_METADATA_GAPS"
    assert "LICENSE_UNDECLARED" in dataset["metadata_issues"]
    assert "AVAILABILITY_UNDECLARED" in dataset["metadata_issues"]
    assert any(issue.startswith("UNITS_UNDECLARED:") for issue in dataset["metadata_issues"])


def test_duplicate_columns_refuse(tmp_path):
    (tmp_path / "data.csv").write_text(
        "DATE_TIME,x,x\n2026-01-01T00:00:00Z,1,2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate column"):
        build_inventory(_registry(), {"fixture": tmp_path.resolve()}, "2026-09-10T00:00:00Z")


def test_dataset_path_must_stay_below_declared_root(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalized and relative"):
        resolve_dataset_path(inside.resolve(), "../outside.csv")


def test_inventory_validator_rejects_producer_counts(tmp_path):
    (tmp_path / "data.csv").write_text(
        "DATE_TIME,x\n2026-01-01T00:00:00Z,1\n", encoding="utf-8"
    )
    inventory = build_inventory(_registry(), {"fixture": tmp_path.resolve()}, "2026-09-10T00:00:00Z")
    forged = copy.deepcopy(inventory)
    forged["datasets"][0]["variable_count"] = 99
    dataset_digest_input = copy.deepcopy(forged["datasets"][0])
    dataset_digest_input.pop("profile_sha256")
    forged["datasets"][0]["profile_sha256"] = sha256_bytes(canonical_json_bytes(dataset_digest_input))
    inventory_digest_input = copy.deepcopy(forged)
    inventory_digest_input.pop("inventory_sha256")
    forged["inventory_sha256"] = sha256_bytes(canonical_json_bytes(inventory_digest_input))

    with pytest.raises(ValueError, match="does not match variables"):
        validate_inventory_document(forged)
