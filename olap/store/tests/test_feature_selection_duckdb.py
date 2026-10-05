"""Production DuckDB acceptance tests for feature-selection envelopes.

These tests deliberately instantiate the real DuckDB provider against a temporary file.  The
SQLite provider in data-warehouse proves the HTTP contract; this suite proves that the provider
which owns the production cube preserves that contract, its transaction and its snapshot.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import text


REPO = Path(__file__).resolve().parents[3]
STORE_SRC = REPO / "olap" / "store" / "src"
DUCKDB_SRC = REPO / "olap" / "duckdb_store" / "src"
sys.path[:0] = [str(STORE_SRC), str(DUCKDB_SRC)]

from predictor_duckdb_store.provider import PredictorDuckdbStore  # noqa: E402
from predictor_olap_store.feature_selection import digest  # noqa: E402
from predictor_olap_store.feature_selection_reconciliation import (  # noqa: E402
    validate_reconciliation_request,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64

TABLES = (
    "df_dim_feature_selection_run",
    "df_fact_sampling_quality",
    "df_fact_variable_profile",
    "df_fact_information_metric",
    "df_fact_pair_relation",
    "df_fact_feature_causal_evidence",
    "df_fact_feature_selection_decision",
    "df_fact_feature_selection_load_receipt",
)
VIEWS = (
    "df_feature_profile_current",
    "df_feature_causal_ladder_current",
    "df_feature_selection_current",
    "df_feature_selection_coverage",
    "df_feature_selection_failures",
    "df_feature_selection_dashboard",
)
CONTRACT_SHA256 = "91fcb4fde495239a4e0a21d3a39f0b66d50bd0a5df4865db7bd720b454f5f75a"
RECONCILIATION_CONTRACT_SHA256 = "b8718cc2879d89ce133474f10f9ab028b5b9b2762c31df617b21dc2ddacd213b"


def sealed(row: dict) -> dict:
    result = copy.deepcopy(row)
    result["row_sha256"] = digest(result)
    return result


def envelope() -> dict:
    document = {
        "schema_version": "feature_selection_envelope.v1",
        "run": {
            "run_id": "fs-phase1-duckdb-test",
            "campaign_sha256": HEX_A,
            "code_sha256": HEX_B,
            "input_sha256": HEX_C,
            "inventory_sha256": HEX_D,
            "created_at": "2026-10-05T12:00:00Z",
        },
        "rows": {
            "sampling_quality": [sealed({
                "feature_id": "market.eth.close", "split": "train",
                "metric_name": "missing_timestamp_count", "metric_value": 2,
                "state": "MEASURED", "unit": "count",
            })],
            "variable_profiles": [sealed({
                "feature_id": "market.eth.close", "split": "train",
                "metric_name": "median", "metric_value": 2048.5,
                "state": "MEASURED", "unit": "USD",
            })],
            "information_metrics": [sealed({
                "feature_id": "market.eth.close", "target_id": "eth.return",
                "horizon": 24, "split": "train", "metric_name": "mutual_information",
                "metric_value": 0.21, "state": "MEASURED",
            })],
            "pair_relations": [sealed({
                "feature_id": "market.eth.close", "target_id": "eth.return",
                "horizon": 24, "split": "train", "lag": 1,
                "metric_name": "spearman", "metric_value": 0.17, "state": "MEASURED",
            })],
            "causal_evidence": [sealed({
                "feature_id": "market.eth.close", "target_id": "eth.return",
                "horizon": 24, "split": "train", "rung": 2, "estimand": "ATE",
                "estimator": "aipw", "state": "IDENTIFIED", "effect": 0.012,
                "lower": 0.004, "upper": 0.020, "support_n": 1200,
                "assumptions": ["exchangeability", "positivity"],
                "adjustment_set": ["hour", "day_of_week"], "evidence_sha256": HEX_A,
            })],
            "selection_decisions": [sealed({
                "feature_id": "market.eth.close", "target_id": "eth.return",
                "horizon": 24, "method": "causal_ladder_fdr", "score": 0.83,
                "rank": 1, "decision": "SELECTED",
                "rule": "q_value <= 0.05 and stable_sign", "evidence_sha256": HEX_A,
            })],
        },
    }
    document["envelope_sha256"] = digest(document)
    return document


@pytest.fixture()
def store(tmp_path):
    plugin = PredictorDuckdbStore()
    plugin.set_params(
        duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
        memory_limit="1GB", threads=2, min_free_bytes=1,
    )
    plugin.engine()
    yield plugin
    if plugin._engine is not None:
        plugin._engine.dispose()


def scalar(store, relation):
    with store.engine().connect() as connection:
        return connection.execute(text(
            f'SELECT count(*) FROM "main"."{relation}"'
        )).scalar()


def reconciliation_identity(document, *, feature_id="market.eth.close"):
    return {
        "feature_id": feature_id,
        "feature_key": feature_id.replace(".", "_"),
        "terminal_state": "COMPLETED",
        "terminal_sha256": HEX_C,
        "envelope_sha256": document["envelope_sha256"],
        "warehouse_receipt_sha256": HEX_D,
    }


def reconciliation_request(identities):
    request = {
        "schema": "phase1.warehouse_reconciliation_request.v1",
        "population_id": "ETH",
        "plan_sha256": HEX_A,
        "expected_count": len(identities),
        "identities_sha256": digest(identities),
        "identities": identities,
        "authentication_profile": "data-gov-service-token",
    }
    request["request_sha256"] = digest(request)
    return request


def test_validation_contract_is_the_exact_file_from_data_warehouse_50bddf3():
    contract = STORE_SRC / "predictor_olap_store" / "feature_selection.py"
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == CONTRACT_SHA256


def test_reconciliation_contract_is_the_exact_file_from_data_warehouse_2d4550d():
    contract = STORE_SRC / "predictor_olap_store" / "feature_selection_reconciliation.py"
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == RECONCILIATION_CONTRACT_SHA256


def test_production_provider_exposes_write_and_reconciliation_to_the_host(store):
    assert "write_feature_selection_envelope" in store.capabilities()
    assert "reconcile_feature_selection" in store.capabilities()
    assert callable(store.reconcile_feature_selection)


def test_reconciliation_returns_exact_request_bound_identity_and_canonical_digest(store):
    document = envelope()
    store.write_feature_selection_envelope(document)
    request = reconciliation_request([reconciliation_identity(document)])

    result = store.reconcile_feature_selection(request)

    assert result["schema"] == "phase1.warehouse_reconciliation.v1"
    assert result["state"] == "RECONCILED"
    for field in (
        "population_id", "plan_sha256", "expected_count", "identities_sha256",
        "authentication_profile", "request_sha256",
    ):
        assert result[field] == request[field]
    assert result["observed_count"] == 1
    assert result["observed_identities"] == request["identities"]
    assert result["observed_identities_sha256"] == digest(request["identities"])
    assert result["complete"] is True
    assert result["contradictions"] == []
    assert result["reconciliation_sha256"] == digest(
        {key: value for key, value in result.items() if key != "reconciliation_sha256"}
    )


def test_reconciliation_rejects_missing_envelope_and_wrong_feature_membership(store):
    document = envelope()
    missing = reconciliation_request([reconciliation_identity(document)])
    with pytest.raises(ValueError, match="incomplete warehouse population"):
        store.reconcile_feature_selection(missing)

    store.write_feature_selection_envelope(document)
    wrong = reconciliation_request([
        reconciliation_identity(document, feature_id="market.eur.close")
    ])
    with pytest.raises(ValueError, match="warehouse population contradiction"):
        store.reconcile_feature_selection(wrong)


def test_reconciliation_queries_retained_run_receipt_and_rows(store):
    document = envelope()
    store.write_feature_selection_envelope(document)
    request = reconciliation_request([reconciliation_identity(document)])

    with store.engine().begin() as connection:
        connection.execute(text(
            'DELETE FROM "main"."df_fact_sampling_quality" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})
        connection.execute(text(
            'DELETE FROM "main"."df_fact_variable_profile" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})
        connection.execute(text(
            'DELETE FROM "main"."df_fact_information_metric" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})
        connection.execute(text(
            'DELETE FROM "main"."df_fact_pair_relation" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})
        connection.execute(text(
            'DELETE FROM "main"."df_fact_feature_causal_evidence" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})
        connection.execute(text(
            'DELETE FROM "main"."df_fact_feature_selection_decision" WHERE feature_id = :feature_id'
        ), {"feature_id": "market.eth.close"})

    with pytest.raises(ValueError, match="retained rows"):
        store.reconcile_feature_selection(request)


def test_reconciliation_rejects_a_retained_run_whose_identity_changed(store):
    document = envelope()
    store.write_feature_selection_envelope(document)
    request = reconciliation_request([reconciliation_identity(document)])
    with store.engine().begin() as connection:
        connection.execute(text(
            'UPDATE "main"."df_dim_feature_selection_run" '
            'SET campaign_sha256 = :changed WHERE run_id = :run_id'
        ), {"changed": HEX_D, "run_id": document["run"]["run_id"]})

    with pytest.raises(ValueError, match="changed identity"):
        store.reconcile_feature_selection(request)


def test_reconciliation_does_not_infer_missing_envelope_membership_from_run_rows(store):
    document = envelope()
    store.write_feature_selection_envelope(document)
    request = reconciliation_request([reconciliation_identity(document)])
    with store.engine().begin() as connection:
        connection.execute(text(
            'UPDATE "main"."df_fact_feature_selection_load_receipt" '
            'SET feature_ids_json = NULL WHERE envelope_sha256 = :envelope_sha256'
        ), {"envelope_sha256": document["envelope_sha256"]})

    with pytest.raises(ValueError, match="receipt feature identity is absent"):
        store.reconcile_feature_selection(request)


def test_unavailable_identity_needs_no_warehouse_payload(store):
    identity = {
        "feature_id": "fred.unavailable",
        "feature_key": "fred_unavailable",
        "terminal_state": "UNAVAILABLE",
        "terminal_sha256": HEX_C,
        "envelope_sha256": None,
        "warehouse_receipt_sha256": None,
    }
    request = reconciliation_request([identity])

    assert validate_reconciliation_request(request) == request
    assert store.reconcile_feature_selection(request)["observed_identities"] == [identity]


def test_real_duckdb_provider_atomically_loads_every_family_and_receipt(store):
    result = store.write_feature_selection_envelope(envelope())

    assert result == {
        "stored": True,
        "already_stored": False,
        "envelope_sha256": envelope()["envelope_sha256"],
        "row_count": 6,
    }
    assert callable(store.write_feature_selection_envelope)
    for relation in TABLES:
        assert scalar(store, relation) == 1


def test_identical_replay_is_noop_and_contradiction_rolls_back_whole_envelope(store):
    original = envelope()
    assert store.write_feature_selection_envelope(original)["stored"] is True
    assert store.write_feature_selection_envelope(original)["already_stored"] is True

    changed = copy.deepcopy(original)
    changed["rows"]["sampling_quality"].append(sealed({
        "feature_id": "market.eth.close", "split": "train",
        "metric_name": "duplicate_timestamp_count", "metric_value": 0,
        "state": "MEASURED", "unit": "count",
    }))
    row = changed["rows"]["variable_profiles"][0]
    row["metric_value"] = 999.0
    row["row_sha256"] = digest({key: value for key, value in row.items()
                                if key != "row_sha256"})
    changed["envelope_sha256"] = digest({key: value for key, value in changed.items()
                                          if key != "envelope_sha256"})

    with pytest.raises(ValueError, match="contradicts"):
        store.write_feature_selection_envelope(changed)

    assert scalar(store, "df_fact_sampling_quality") == 1
    assert scalar(store, "df_fact_variable_profile") == 1
    assert scalar(store, "df_fact_feature_selection_load_receipt") == 1


def test_concurrent_identical_replays_commit_exactly_once(store):
    document = envelope()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: store.write_feature_selection_envelope(document), range(8)
        ))

    assert sum(result["stored"] for result in results) == 1
    assert sum(result["already_stored"] for result in results) == 7
    assert scalar(store, "df_fact_feature_selection_load_receipt") == 1
    assert scalar(store, "df_fact_feature_causal_evidence") == 1


def test_invalid_row_digest_is_rejected_before_any_write(store):
    document = envelope()
    document["rows"]["pair_relations"][0]["metric_value"] = -0.99
    document["envelope_sha256"] = digest(
        {key: value for key, value in document.items() if key != "envelope_sha256"}
    )

    with pytest.raises(ValueError, match="row_sha256"):
        store.write_feature_selection_envelope(document)

    for relation in TABLES:
        assert scalar(store, relation) == 0


def test_rows_and_read_only_views_survive_reopening_the_real_duckdb_file(store):
    path = store._database_path()
    store.write_feature_selection_envelope(envelope())
    store._engine.dispose()
    store._engine = None

    reopened = PredictorDuckdbStore()
    reopened.set_params(duckdb_path=str(path), schema="main", memory_limit="1GB",
                        threads=2, min_free_bytes=1)
    reopened.engine()
    try:
        assert scalar(reopened, "df_fact_feature_causal_evidence") == 1
        assert scalar(reopened, "df_fact_feature_selection_decision") == 1
        assert scalar(reopened, "df_feature_selection_dashboard") == 1
        with reopened.engine().connect() as connection:
            dashboard = connection.execute(text(
                'SELECT total_features, selected_features, failed_rows '
                'FROM "main"."df_feature_selection_dashboard"'
            )).one()
        assert tuple(dashboard) == (1, 1, 0)
    finally:
        reopened._engine.dispose()


def _migration_module():
    path = REPO / "tools" / "olap_duckdb_migrate.py"
    spec = importlib.util.spec_from_file_location("olap_duckdb_migrate_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verified_snapshot_covers_feature_selection_tables_and_views(store, tmp_path):
    source = store._database_path()
    store.write_feature_selection_envelope(envelope())
    store._engine.dispose()
    store._engine = None
    target = tmp_path / "snapshot.duckdb"

    report = _migration_module().snapshot_database(
        str(source), str(target), expect_terminals=0, owner_stopped=True,
    )

    assert report["schema"] == "olap_duckdb_snapshot.v5"
    assert report["kind"] == "VERIFIED_SNAPSHOT"
    expected_counts = {name: 1 for name in TABLES + VIEWS}
    expected_counts["df_feature_profile_current"] = 4
    expected_counts["df_feature_selection_failures"] = 0
    for relation in TABLES + VIEWS:
        assert relation in report["counts_in_snapshot"]
        assert relation in report["content_digests"]
        assert report["counts_in_snapshot"][relation] == expected_counts[relation]
        assert report["content_digests"][relation]
        assert report["content_digests"][relation] == report["source_digests"][relation]


def test_snapshot_with_a_missing_phase1_view_is_named_unverified(store, tmp_path):
    source = store._database_path()
    store.write_feature_selection_envelope(envelope())
    with store.engine().begin() as connection:
        connection.execute(text(
            'DROP VIEW "main"."df_feature_selection_dashboard"'
        ))
    store._engine.dispose()
    store._engine = None

    report = _migration_module().snapshot_database(
        str(source), str(tmp_path / "incomplete.duckdb"),
        expect_terminals=0, owner_stopped=True,
    )

    assert report["kind"] == "UNVERIFIED_COPY"
    assert "df_feature_selection_dashboard" in json.dumps(report["unverified_because"])
