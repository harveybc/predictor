"""A replayed metrics report must not duplicate DuckDB child rows."""

import sys
from pathlib import Path

import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "olap/store/src"), str(ROOT / "olap/duckdb_store/src")]

from predictor_duckdb_store.provider import PredictorDuckdbStore  # noqa: E402
from predictor_olap_store.query import report_sha256  # noqa: E402


def test_replay_keeps_one_report_and_its_two_metrics(tmp_path):
    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
                     memory_limit="1GB", threads=1, min_free_bytes=1)
    report = {
        "experiment_key": "idempotence:test", "actor": "predictor",
        "lake": "LOCAL_RETAINED", "lineage": "UNVERIFIED",
        "metrics": [
            {"metric": "MAE", "value": 0.1, "split": "validation", "horizon": 1},
            {"metric": "Naive_MAE", "value": 0.2, "split": "validation", "horizon": 1},
        ],
    }
    report["report_sha256"] = report_sha256(report)
    try:
        assert store.write_metrics(report)["stored"] is True
        assert store.write_metrics(report)["already_stored"] is True
        with store.engine().connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM main.gov_report")).scalar() == 1
            assert connection.execute(text("SELECT count(*) FROM main.gov_metric")).scalar() == 2
        report["metrics"][0]["value"] = 0.9
        with pytest.raises(ValueError, match="report_sha256 mismatch"):
            store.write_metrics(report)
    finally:
        if store._engine is not None:
            store._engine.dispose()
