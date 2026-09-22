"""A terminal's metrics, datasets and artifacts must actually land, on every engine.

Found by the G1 reconciler on production evidence: every terminal written natively by the
DuckDB provider holds **zero** metrics, while the equivalent PostgreSQL-era terminal holds two.
Its datasets landed. `write_terminal` reported `stored: True` throughout — it dropped the
metrics silently, which is the worst possible way to lose data.

This is not the incident. These writes happened after the cutover, through the ordinary
governed path, and nobody would have noticed without comparing against what was accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))

PG_DATABASE = os.environ.get("U2_PG_DATABASE")
DUCKDB = os.environ.get("U2_DUCKDB_PATH")
ENGINES = ["sqlite"] + (["postgres"] if PG_DATABASE else []) + (["duckdb"] if DUCKDB else [])


@pytest.fixture(params=ENGINES)
def store(request, tmp_path):
    from predictor_olap_store.query import Plugin
    from sqlalchemy import text

    if request.param == "duckdb":
        from predictor_duckdb_store.provider import PredictorDuckdbStore

        plugin = PredictorDuckdbStore()
        plugin.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
                          memory_limit="1GB", threads=2, min_free_bytes=1)
        plugin.engine()
        return plugin
    plugin = Plugin()
    if request.param == "postgres":
        os.environ["PGDATABASE"] = PG_DATABASE
        plugin.set_params(sqlite_path=None, schema="public")
        plugin.engine()
        with plugin.write_engine().begin() as conn:
            for name in ("gov_terminal_metric", "gov_terminal_dataset",
                         "gov_terminal_artifact", "gov_terminal"):
                conn.execute(text(f"DELETE FROM {name}"))
        return plugin
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    plugin.engine()
    return plugin


def terminal_with(metrics=2, datasets=1, artifacts=1, artifact_bytes=10) -> dict:
    """A terminal built the way governance builds one, including its own digest.

    The digest is the sha256 of the canonical body, so it cannot be invented: a fixture with a
    made-up digest tests the validator, not the persistence.
    """
    from predictor_olap_store.query import canonical_text

    body = {
        "schema": "governed_terminal.v1",
        "campaign_sha256": "c" * 64, "campaign_key": "children-probe", "unit_id": "u1",
        "generation": 1, "actor": "a", "project": "p", "classification": "NON_GOVERNING",
        "status": "COMPLETED", "reason": None, "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z", "terminal_lake": "olap_cube",
        "config_sha256": "e" * 64, "code_identity": {"kind": "git_commit", "value": "d" * 40},
        "costs": {"wall_seconds": 1.0}, "tags": {}, "synthetic_spec_sha256": None,
        "deliveries": [f"{index:032x}" for index in range(datasets)],
        "metrics": [{"metric": f"m{index}", "split": "test", "horizon": 0, "unit": "u",
                     "value": float(index), "std_dev": None, "min_value": None,
                     "max_value": None} for index in range(metrics)],
        "verified_datasets": [{"delivery_id": f"{index:032x}", "lake_id": "l",
                               "resource_id": "r", "role": "input", "sha256": "b" * 64,
                               "bytes": 1, "source_sha256": None, "range_from": None,
                               "range_to": None, "delivery_kind": "AS_IS", "time_column": "",
                               "availability_contract_sha256": "f" * 64,
                               "state": "VERIFIED_TRANSFER"} for index in range(datasets)],
        "artifacts": [{"role": f"a{index}", "sha256": "a" * 64, "bytes": artifact_bytes}
                      for index in range(artifacts)],
    }
    body["terminal_sha256"] = hashlib.sha256(
        canonical_text({k: v for k, v in body.items()}).encode("ascii")).hexdigest()
    return body


def counts(store, digest):
    from sqlalchemy import text

    out = {}
    with store.engine().connect() as conn:
        for child in ("gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact"):
            out[child] = conn.execute(text(
                f'SELECT count(*) FROM {store._qualified(child)} '
                "WHERE terminal_sha256 = :d"), {"d": digest}).scalar()
    return out


def test_every_child_row_of_a_terminal_is_persisted(store):
    """Two metrics, one dataset and one artifact go in; all four must come back out."""
    terminal = terminal_with(metrics=2, datasets=1, artifacts=1)
    outcome = store.write_terminal(terminal)

    assert outcome["stored"] is True
    assert counts(store, terminal["terminal_sha256"]) == {
        "gov_terminal_metric": 2, "gov_terminal_dataset": 1, "gov_terminal_artifact": 1}


def test_many_metrics_all_land(store):
    """A single row could pass by accident; a batch is what the real path writes."""
    terminal = terminal_with(metrics=25, datasets=3, artifacts=2)
    store.write_terminal(terminal)
    assert counts(store, terminal["terminal_sha256"]) == {
        "gov_terminal_metric": 25, "gov_terminal_dataset": 3, "gov_terminal_artifact": 2}


def test_a_terminal_with_no_children_is_still_stored(store):
    terminal = terminal_with(metrics=0, datasets=0, artifacts=0)
    assert store.write_terminal(terminal)["stored"] is True
    assert counts(store, terminal["terminal_sha256"]) == {
        "gov_terminal_metric": 0, "gov_terminal_dataset": 0, "gov_terminal_artifact": 0}


def test_an_artifact_larger_than_two_gigabytes_lands(store):
    """The T=720 prediction array of the SOTA reproduction is 4,198,064,038 bytes; a 32-bit
    `bytes` column refused its terminal (2026-09-22). Every engine must store it."""
    from sqlalchemy import text

    terminal = terminal_with(artifacts=1, artifact_bytes=4_198_064_038)
    assert store.write_terminal(terminal)["stored"]
    with store.engine().connect() as conn:
        stored = conn.execute(text(
            f"SELECT bytes FROM {store._qualified('gov_terminal_artifact')}"
            " WHERE terminal_sha256 = :d"), {"d": terminal["terminal_sha256"]}).scalar()
    assert int(stored) == 4_198_064_038


@pytest.mark.skipif(not DUCKDB, reason="U2_DUCKDB_PATH not set")
def test_a_duckdb_store_created_with_a_32_bit_bytes_column_is_widened_in_place(tmp_path):
    """The deployed cube was created with `bytes INTEGER`; the schema hook widens it to BIGINT
    without touching the rows already stored."""
    import duckdb
    from sqlalchemy import text
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE gov_terminal_artifact (terminal_sha256 TEXT NOT NULL, role TEXT NOT NULL,"
                " sha256 TEXT NOT NULL, bytes INTEGER NOT NULL)")
    con.execute("INSERT INTO gov_terminal_artifact VALUES ('t', 'r', 's', 7)")
    con.close()
    plugin = PredictorDuckdbStore()
    plugin.set_params(duckdb_path=str(path), schema="main", memory_limit="1GB", threads=2, min_free_bytes=1)
    plugin.engine()
    with plugin.engine().connect() as conn:
        kind = conn.execute(text("SELECT data_type FROM information_schema.columns WHERE table_name = 'gov_terminal_artifact'"
                                 " AND column_name = 'bytes'")).scalar()
        kept = conn.execute(text("SELECT bytes FROM gov_terminal_artifact WHERE terminal_sha256 = 't'")).scalar()
    assert kind == "BIGINT" and kept == 7
    terminal = terminal_with(artifacts=1, artifact_bytes=4_198_064_038)
    assert plugin.write_terminal(terminal)["stored"]
