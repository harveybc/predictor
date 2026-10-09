import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools import i6b_weekly_campaign as C


def _design():
    return {
        "design_sha256": "d" * 64,
        "members": ["a", "b"],
        "config": {"seed": 7},
        "weeks": [{
            "ordinal": 0, "fit_start": "2020-01-01T00:00:00Z",
            "cutoff": "2024-01-01T00:00:00Z", "start": "2024-01-01T00:00:00Z",
            "end": "2024-01-08T00:00:00Z",
        }],
    }


def _store():
    start = 1577836800
    rows = 4 * 366 * 24
    ts = start + np.arange(rows, dtype="int64") * 3600
    values = np.column_stack((np.arange(rows), np.arange(rows) * 2.0)).astype("float64")
    values[5, 0] = np.nan
    return SimpleNamespace(
        names=["a", "b"], col={"a": 0, "b": 1}, X=values, ts=ts,
        digests={"train:a": "a" * 64}, population="eurusd",
    )


def test_prepare_week_uses_four_year_boundary_and_never_the_cutoff_or_future():
    windows, corpus, mapping = C.prepare_week_windows(_store(), _design(), _design()["weeks"][0])

    assert mapping == {"branch_000": "a", "branch_001": "b"}
    assert windows["branch_000"].shape == (35041, 24, 2)
    assert windows["branch_001"].shape == (35041, 24, 2)
    assert corpus["support"]["fit_start"] == "2020-01-01T00:00:00Z"
    assert corpus["support"]["cutoff_exclusive"] == "2024-01-01T00:00:00Z"
    assert corpus["support"]["last_origin_utc"] == "2023-12-31T23:00:00Z"
    assert corpus["support"]["outer_validation_scoring_rows_read"] is False
    assert np.isfinite(windows["branch_000"]).all()


def test_prepare_week_rejects_missing_member_and_wrong_calendar():
    design = _design()
    design["members"] = ["a", "missing"]
    with pytest.raises(ValueError, match="missing selected members"):
        C.prepare_week_windows(_store(), design, design["weeks"][0])

    design = _design()
    design["weeks"][0]["fit_start"] = "2020-01-01T01:00:00Z"
    with pytest.raises(ValueError, match="four calendar years"):
        C.prepare_week_windows(_store(), design, design["weeks"][0])


def test_status_requires_every_week_and_verifies_the_index(tmp_path):
    design = _design()
    root = tmp_path / "campaign"
    week = root / "week_000"
    week.mkdir(parents=True)
    report = {
        "schema": "predictor.i6b.branch_pretraining.v1", "status": "COMPLETE",
        "branch_count": 2,
        "branches": [
            {"branch": "branch_000", "model_sha256": "0" * 64},
            {"branch": "branch_001", "model_sha256": "1" * 64},
        ],
    }
    (week / "REPORT.json").write_text(json.dumps(report))
    C.write_week_receipt(root, design, 0, {"branch_000": "a", "branch_001": "b"}, report)

    status = C.campaign_status(root, design)
    assert status["status"] == "COMPLETE"
    assert status["completed_weeks"] == 1
    assert status["completed_donors"] == 2

    receipt = json.loads((week / "WEEK_RECEIPT.json").read_text())
    receipt["donors"][0]["feature"] = "wrong"
    (week / "WEEK_RECEIPT.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        C.campaign_status(root, design)


def test_status_is_incomplete_when_any_week_is_missing(tmp_path):
    design = _design()
    design["weeks"].append({**design["weeks"][0], "ordinal": 1})
    status = C.campaign_status(tmp_path, design)
    assert status["status"] == "IN_PROGRESS"
    assert status["pending_weeks"] == [0, 1]


def test_cli_has_no_test_split_or_test_paths():
    parser = C.build_parser()
    help_text = parser.format_help().lower()
    assert "test" not in help_text
    run_help = parser._subparsers._group_actions[0].choices["run-week"].format_help().lower()
    assert "target" not in run_help


def test_feature_store_reads_selected_columns_and_only_pre_cutoff_rows(tmp_path):
    timestamps = pa.array(
        [C._parse(value) for value in
         ("2023-12-31T22:00:00Z", "2023-12-31T23:00:00Z", "2024-01-01T00:00:00Z")],
        type=pa.timestamp("s", tz="UTC"),
    )
    path = tmp_path / "features.parquet"
    pq.write_table(pa.table({
        "t_decision_utc": timestamps, "row_id": [1, 2, 3],
        "keep": [1.0, 2.0, 999.0], "ignore": [4.0, 5.0, 999.0],
    }), path)

    store = C._feature_store(
        (("base", [path]), ("history", [])), "eurusd", ["keep"],
        C._parse("2023-12-31T22:00:00Z"), C._parse("2024-01-01T00:00:00Z"),
    )
    assert store.names == ["keep"]
    assert store.X.tolist() == [[1.0], [2.0]]
    assert store.row_ids.tolist() == [1, 2]
