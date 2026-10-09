"""Tests for authenticated I6-D shard reconciliation."""

import json

import pytest

from tools import i6d_weekly_shard_merge as merger
from tools import i6d_weekly_walk_forward as weekly


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _design():
    return weekly._seal({
        "schema": weekly.DESIGN_SCHEMA,
        "control_kind": "BRANCH_ONLY",
        "target_transform": "ROBUST_Z_FIT",
        "test_paths": None,
        "test_read": False,
        "evaluation_mode": weekly.EVALUATION_MODE,
        "update_mode": weekly.UPDATE_MODE,
        "rolling_calendar_years": 4,
        "config": {"seed": 0},
        "seeds": [0],
        "weeks": [{"ordinal": 0}],
        "arms": ["DENSE", "CONV"],
        "expected_cells": 2,
        "branch_only_contract": {},
    }, "design_sha256")


def _cell(design, arm, value=1.0):
    return weekly._seal({
        "schema": weekly.CELL_SCHEMA,
        "design_sha256": design["design_sha256"],
        "seed": 0,
        "week_ordinal": 0,
        "arm": arm,
        "metric": value,
    }, "cell_sha256")


def _campaign(root, design, cells=()):
    _write_json(root / "WEEKLY_DESIGN.json", design)
    for cell in cells:
        _write_json(
            weekly._cell_path(root, cell["seed"], cell["week_ordinal"], cell["arm"]),
            cell,
        )


def test_merge_copies_verified_cells_and_is_idempotent(tmp_path):
    design = _design()
    destination = tmp_path / "destination"
    source = tmp_path / "source"
    _campaign(destination, design)
    _campaign(source, design, [_cell(design, "DENSE"), _cell(design, "CONV")])

    first = merger.merge_shards(destination, [source])
    second = merger.merge_shards(destination, [source])

    assert first["copied"] == 2
    assert first["existing"] == 0
    assert second["copied"] == 0
    assert second["existing"] == 2
    assert first["complete"] is True


def test_merge_rejects_a_different_design_before_copying(tmp_path):
    design = _design()
    other = dict(design)
    other["expected_cells"] = 3
    other = weekly._seal(other, "design_sha256")
    destination = tmp_path / "destination"
    source = tmp_path / "source"
    _campaign(destination, design)
    _campaign(source, other, [_cell(other, "DENSE")])

    with pytest.raises(ValueError, match="design identity mismatch"):
        merger.merge_shards(destination, [source])
    assert not weekly._cell_path(destination, 0, 0, "DENSE").exists()


def test_merge_rejects_tampered_and_conflicting_cells(tmp_path):
    design = _design()
    destination = tmp_path / "destination"
    source = tmp_path / "source"
    _campaign(destination, design, [_cell(design, "DENSE", 1.0)])
    _campaign(source, design, [_cell(design, "DENSE", 2.0)])

    with pytest.raises(ValueError, match="conflicting cell"):
        merger.merge_shards(destination, [source])

    bad = _cell(design, "CONV")
    bad["metric"] = 999.0
    _campaign(source, design, [bad])
    with pytest.raises(ValueError, match="cell_sha256 mismatch"):
        merger.merge_shards(destination, [source])


def test_cross_source_conflict_is_rejected_before_any_copy(tmp_path):
    design = _design()
    destination = tmp_path / "destination"
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _campaign(destination, design)
    _campaign(source_a, design, [_cell(design, "DENSE", 1.0)])
    _campaign(source_b, design, [_cell(design, "DENSE", 2.0)])

    with pytest.raises(ValueError, match="conflicting cell across sources"):
        merger.merge_shards(destination, [source_a, source_b])
    assert not weekly._cell_path(destination, 0, 0, "DENSE").exists()


def test_close_does_not_report_complete_when_full_population_fails_scientific_gates(tmp_path):
    design = _design()
    destination = tmp_path / "destination"
    source = tmp_path / "source"
    _campaign(destination, design)
    _campaign(source, design, [_cell(design, "DENSE"), _cell(design, "CONV")])

    result = merger.merge_shards(destination, [source], close=True)

    assert result["complete_cells"] == result["expected_cells"] == 2
    assert result["closure_state"] == "INCOMPLETE_EVIDENCE"
    assert result["complete"] is False
