"""Additive DOIN projection: gov_metric rows plus versioned DOIN metadata, with reorg history.

Runs on throwaway SQLite and (when duckdb is importable) a throwaway DuckDB file. Neither
certifies the deployed cube.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "olap/store/src"), str(ROOT / "olap/duckdb_store/src")]

from predictor_olap_store import doin_projection_store as dps  # noqa: E402
from predictor_olap_store.query import Plugin, report_sha256  # noqa: E402

IDENTITY = {
    "experiment": "i6n-swarm-test", "candidate_identity_sha256": "a" * 64, "fold": "2023Q1",
    "horizon_hours": 12, "head_offset_hours": 12, "role": "management_origin", "population": "S",
    "split": "TRAIN", "metric": "ABS_AND_SQUARED_ERROR_SUMS", "unit": "PRICE",
    "schema_version": "i6n.candidate_receipt.v1",
}


def _document(block="b" * 64, value=0.001, finality="FINALIZED", identity=None, mode="external_payload"):
    identity = dict(identity or IDENTITY)
    row_key = dps.row_key_sha256(identity)
    report = {
        "experiment_key": "i6n-doin:" + row_key, "actor": "predictor", "lake": "doin_archive_cas",
        "lineage": "UNVERIFIED", "project": "predictor", "phase": "I6N_SWARM_DOIN",
        "config_sha256": "c" * 64, "code_commit": "0" * 40,
        "tags": {"row_key_sha256": row_key, "doin_metadata_schema": dps.DOIN_METADATA_SCHEMA},
        "metrics": [
            {"metric": "mae", "value": value, "unit": "PRICE", "split": "TRAIN", "horizon": 12},
            {"metric": "n", "value": 10.0, "unit": "count", "split": "TRAIN", "horizon": 12},
        ],
    }
    report["report_sha256"] = report_sha256(report)
    projection = dict(identity, search_candidate_sha256="d" * 64, head_role="ENDPOINT",
                      comparison=True, status="SCORED", row_key_sha256=row_key,
                      report_sha256=report["report_sha256"], block_hash=block, block_height=3,
                      tx_id="e" * 64, body_digest="f" * 64, manifest_digest="1" * 64,
                      locator=("doin-cas://doin_archive_cas/sha256/" + "f" * 64) if mode == "external_payload" else None,
                      archive_mode=mode, finality_state=finality, confirmation_depth=6, depth=6)
    return {"schema": dps.SCHEMA_WRITE, "report": report, "projection": projection}


def _sqlite(tmp_path):
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    return plugin


def _duckdb(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
                     memory_limit="512MB", threads=1, min_free_bytes=1)
    return store


@pytest.fixture(params=["sqlite", "duckdb"])
def store(request, tmp_path):
    backend = _sqlite(tmp_path) if request.param == "sqlite" else _duckdb(tmp_path)
    yield backend
    if backend._engine is not None:
        backend._engine.dispose()


def _count(store, relation, where=""):
    with store.engine().connect() as connection:
        return connection.execute(text(f"SELECT count(*) FROM {store._qualified(relation)} {where}")).scalar()


def _canonical(store):
    with store.engine().connect() as connection:
        rows = connection.execute(text(
            f"SELECT derived_metric, value, block_hash FROM {store._qualified('gov_doin_metric_canonical')}"
            " ORDER BY derived_metric")).fetchall()
    return [tuple(r) for r in rows]


def test_projection_writes_metrics_and_reads_back(store):
    receipt = store.write_doin_projection(_document())
    assert receipt["state"] == "PROJECTED" and receipt["metrics_inserted"] == 2
    read = store.read_doin_projection({"report_sha256": [receipt["report_sha256"]]})
    assert [p["state"] for p in read["projections"]] == ["PROJECTED"]
    assert sorted((m["metric"], m["value"]) for m in read["metrics"]) == [("mae", 0.001), ("n", 10.0)]
    assert _canonical(store) == [("mae", 0.001, "b" * 64), ("n", 10.0, "b" * 64)]


def test_replay_and_duplicate_block_do_not_add_metrics(store):
    store.write_doin_projection(_document())
    again = store.write_doin_projection(_document())
    assert again["state"] == "ALREADY_PROJECTED" and again["metrics_inserted"] == 0
    other_block = store.write_doin_projection(_document(block="9" * 64))
    assert other_block["state"] == "DUPLICATE_NOT_PROJECTED"
    assert _count(store, "gov_metric") == 2
    assert len(_canonical(store)) == 2


def test_conflicting_values_for_an_active_row_are_refused(store):
    store.write_doin_projection(_document())
    with pytest.raises(ValueError, match="CONFLICT"):
        store.write_doin_projection(_document(block="9" * 64, value=0.5))
    assert _count(store, "gov_metric") == 2


def test_reorg_revokes_canonical_projection_and_keeps_history(store):
    store.write_doin_projection(_document())
    revoked = store.revoke_doin_projection({"schema": dps.SCHEMA_REVOKE, "block_hash": "b" * 64,
                                            "reason": "REORG", "observed_tip_hash": "7" * 64,
                                            "observed_tip_height": 9})
    assert revoked["revoked"] == 1
    assert _canonical(store) == []
    assert _count(store, "gov_metric") == 2           # history kept
    assert _count(store, "gov_doin_projection_event") == 2
    # The same rows re-committed in a block of the new canonical chain project again.
    moved = store.write_doin_projection(_document(block="8" * 64))
    assert moved["state"] == "PROJECTED" and moved["metrics_inserted"] == 0
    assert _canonical(store) == [("mae", 0.001, "8" * 64), ("n", 10.0, "8" * 64)]
    read = store.read_doin_projection({"row_key_sha256": dps.row_key_sha256(IDENTITY)})
    assert sorted(p["state"] for p in read["projections"]) == ["PROJECTED", "REVOKED"]
    assert [e["event"] for e in read["events"]] == ["PROJECTED", "REVOKED", "PROJECTED"]


def test_reinstating_a_block_after_a_reorg_back(store):
    store.write_doin_projection(_document())
    store.revoke_doin_projection({"schema": dps.SCHEMA_REVOKE, "block_hash": "b" * 64, "reason": "REORG"})
    back = store.write_doin_projection(_document())
    assert back["state"] == "REINSTATED"
    assert len(_canonical(store)) == 2


def test_refusals_before_any_write(store):
    bad = _document(finality="CONFIRMING")
    with pytest.raises(ValueError, match="FINALIZED"):
        store.write_doin_projection(bad)
    test_split = _document(identity=dict(IDENTITY, split="TEST"))
    with pytest.raises(ValueError, match="split"):
        store.write_doin_projection(test_split)
    tampered = _document()
    tampered["report"]["metrics"][0]["value"] = 9.0
    with pytest.raises(ValueError, match="report_sha256"):
        store.write_doin_projection(tampered)
    wrong_key = _document()
    wrong_key["projection"]["row_key_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row_key"):
        store.write_doin_projection(wrong_key)
    no_locator = _document()
    no_locator["projection"]["locator"] = None
    with pytest.raises(ValueError, match="locator"):
        store.write_doin_projection(no_locator)
    assert _count(store, "gov_metric") == 0
    assert _count(store, "gov_doin_projection") == 0


def test_migration_is_additive_and_idempotent(store):
    store.engine()
    store._ensure_schema(store.engine())  # second application creates nothing new, drops nothing
    for relation in dps.ALL_TABLES + dps.VIEWS:
        assert _count(store, relation) == 0
    assert "CREATE TABLE IF NOT EXISTS" in dps.render_sql()
    assert "DROP" not in dps.render_sql().upper()
