"""Acceptance tests for tools/fs_phase23_warehouse.py on THROWAWAY DuckDB files.

Subplan §6 points covered here (the engineering agent covers the numerical ones):
  * a receipt / row with a foreign run or population identity is rejected;
  * a restart (identical replay) produces the same digests and duplicates no row;
  * column order does not change identity or digests;
  * the closure reconciliation fails when one expected disposition is missing;
  * the 12 EURUSD and 3 ETH phase-1 features are labelled CAUSAL_SUPPORTED, not selected;
  * the migration is additive and idempotent; the code is the single DDL source and the
    checked-in .sql file must equal it;
  * local and SQL digests agree, and they equal the driver adapter's rule;
  * the driver's interface (open_warehouse / submit_rows / read_run / reconcile) holds;
  * the packaged backend (SQLAlchemy over DuckDB) stores and reads the same rows;
  * the snapshot procedure refuses a live store and verifies its own asset.

No test opens the configured warehouse, contacts the service or restarts anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "olap" / "store" / "src"))
sys.path.insert(0, str(ROOT / "olap" / "duckdb_store" / "src"))

import fs_phase23_warehouse as wh  # noqa: E402
from predictor_olap_store import fs_phase23_store as core  # noqa: E402

CONTRACT = ROOT / "docs/audits/evidence/canonical_20261003/fs_phase23/data/CONTRACT.json"
HEX = "a" * 64
HEX_B = "b" * 64
RUN = "phase2-eurusd:test"


def row_key(run_id, *parts):  # the driver's rule, tools/feature_pairwise_worker.py
    return hashlib.sha256("|".join([run_id, *[str(p) for p in parts]]).encode()).hexdigest()


def adapter_rows_digest(rows):  # the driver adapter's rule, verbatim
    parts = sorted(hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                             allow_nan=False).encode()).hexdigest() for r in rows)
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def metric_row(left, right, *, run_id=RUN, population="EURUSD", fold="inner_2019", metric="spearman",
               lag=0, value=0.5, state="MEASURED", support=1000, unit="unit-0", **extra):
    row = {"run_id": run_id, "population_id": population, "unit_id": unit, "method": "pairwise_v1",
           "params_sha256": HEX, "code_sha256": HEX, "left": left, "right": right, "fold_id": fold,
           "metric": metric, "lag_hours": lag, "value": value, "support_n": support, "state": state,
           "reason": None, "estimator": "pearson_r", "params": {"bins": 16},
           "row_key": row_key(run_id, "metric", left, right, fold, metric, lag)}
    row.update(extra)
    return row


def stability_row(left, right, *, run_id=RUN, metric="spearman", lag=0, unit="unit-0"):
    return {"run_id": run_id, "population_id": "EURUSD", "unit_id": unit, "method": "pairwise_v1",
            "params_sha256": HEX, "code_sha256": HEX, "left": left, "right": right, "metric": metric,
            "lag_hours": lag, "fold_count": 5, "mean": 0.4, "sd": 0.1, "sign_consistent": True,
            "min": 0.3, "max": 0.5, "valid_folds": 5, "state": "MEASURED",
            "fold_states": {"inner_2019": "MEASURED"},
            "row_key": row_key(run_id, "stability", left, right, metric, lag)}


def gate_row(left, right, *, run_id=RUN, unit="unit-0", gate_state="DISTINCT"):
    return {"run_id": run_id, "population_id": "EURUSD", "unit_id": unit, "method": "pairwise_v1",
            "params_sha256": HEX, "code_sha256": HEX, "left": left, "right": right, "fold_id": "TRAIN",
            "gate_state": gate_state, "shared_support": 7000, "byte_identical": False,
            "row_key": row_key(run_id, "gate", left, right)}


@pytest.fixture
def wh_file(tmp_path):
    store = wh.open_warehouse(tmp_path / "throwaway.duckdb")
    yield store
    store.close()


# ----------------------------------------------------------------------------- denominators
def test_pair_denominators_are_the_declared_ones():
    assert wh.pair_denominator(366) == 66_795
    assert wh.pair_denominator(83) == 3_403
    with pytest.raises(wh.Refusal):
        wh.pair_denominator(1)


def test_order_pair_is_lexicographic_and_refuses_self_pairs():
    assert wh.order_pair("px.b", "px.a") == ("px.a", "px.b")
    with pytest.raises(wh.Refusal):
        wh.order_pair("x", "x")


# ----------------------------------------------------------------------------- contract
def test_contract_is_frozen_with_phase1_identities_and_arithmetic():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    summary = wh.verify_contract(contract)
    assert summary["EURUSD"] == {"features": 366, "pairs": 66_795, "targets": 14, "folds": 5,
                                 "causal_supported_features": 12}
    assert summary["ETH"] == {"features": 83, "pairs": 3_403, "targets": 6, "folds": 3,
                              "causal_supported_features": 3}
    e = contract["populations"]["EURUSD"]
    assert e["phase1_run_id"] == "phase1-eurusd-final:94d20c038d55e152"
    assert e["phase1_identity"]["input_sha256"].startswith("94d20c038d55e152")
    assert [f["name"] for f in e["folds"]] == ["inner_2019", "inner_2020", "inner_2021",
                                               "inner_2022", "inner_2023"]
    assert e["train_period"] == ["2012-05-01T00:00:00Z", "2024-01-01T00:00:00Z"]
    assert e["causal_supported"]["pair_count"] == 34
    t = contract["populations"]["ETH"]
    assert t["phase1_run_id"] == "phase1-final:ETH:29d2f745f5d9e87c"
    assert t["phase1_identity"]["plan_sha256"].startswith("29d2f745f5d9e87c")
    assert sorted(t["causal_supported"]["distinct_features"]) == [
        "log_return_1", "return_1", "statistical__log_return_1"]
    assert contract["warehouse"]["tables"] == list(core.ALL_TABLES)
    assert contract["warehouse"]["migration_sql_sha256"] == wh.file_sha256(wh.MIGRATION_SQL)


def test_contract_labels_causal_features_as_candidates_not_selection():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for pop in contract["populations"].values():
        assert pop["causal_supported"]["label"] == "CAUSAL_SUPPORTED"
        assert "NOT a final selection" in pop["causal_supported"]["meaning"]
    assert "never count as independent evidence" in contract["rules"]["alias_evidence"]
    text = CONTRACT.read_text(encoding="utf-8")
    for forbidden in ("/home/", "harveybc", "192.168.", ".local.json"):
        assert forbidden not in text, f"private reference {forbidden!r} in the contract"


def test_contract_digest_is_load_bearing():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["populations"]["EURUSD"]["pair_denominator"] = 66_794
    with pytest.raises(wh.Refusal):
        wh.verify_contract(contract)


# ----------------------------------------------------------------------------- migration
def test_migration_is_additive_idempotent_and_the_sql_file_equals_the_code(tmp_path):
    assert wh.migration_file_matches_code(), "regenerate with `fs_phase23_warehouse.py render-migration`"
    connection = duckdb.connect(str(tmp_path / "m.duckdb"))
    first = wh.apply_migration(connection)
    assert set(core.ALL_RELATIONS) <= set(first["created"])
    second = wh.apply_migration(connection)
    assert second["created"] == []
    assert sorted(second["already_present"]) == sorted(core.ALL_RELATIONS)
    plan = wh.migration_plan(connection)
    assert plan["would_create"] == [] and plan["destructive_statements"] == 0
    for statement in core.ddl():
        assert statement.upper().startswith(("CREATE TABLE IF NOT EXISTS", "CREATE OR REPLACE VIEW"))
        assert "DROP" not in statement.upper() and "ALTER" not in statement.upper()
    connection.close()


def test_a_stale_sql_file_is_refused(tmp_path):
    stale = tmp_path / "stale.sql"
    stale.write_text(core.render_sql().replace("UNIQUE (run_id, row_key)", "UNIQUE (row_key)"))
    with pytest.raises(wh.Refusal, match="does not match"):
        wh.apply_migration(duckdb.connect(str(tmp_path / "m.duckdb")), stale)


def test_every_fact_table_has_both_unique_keys():
    for table in core.FACT_TABLES:
        statement = next(s for s in core.ddl() if f"EXISTS {table} (" in s)
        assert "UNIQUE (run_id, row_key)" in statement
        assert f"UNIQUE ({', '.join(core.IDENTITY[table])})" in statement
        assert "row_identity_sha256 TEXT PRIMARY KEY" in statement


# ----------------------------------------------------------------------------- the driver interface
def test_open_warehouse_submit_read_reconcile_hold_the_driver_contract(wh_file):
    rows = [metric_row("a", "b"), metric_row("a", "c", value=0.1), metric_row("a", "b", metric="pearson")]
    receipt = wh_file.submit_rows(RUN, "feature_pair_metrics", rows)
    for key in ("run_id", "table", "row_count", "rows_sha256", "receipt_sha256", "inserted",
                "duplicates_ignored", "backend", "unit_id", "population_id"):
        assert key in receipt, key
    assert receipt["run_id"] == RUN and receipt["table"] == "feature_pair_metrics"
    assert receipt["row_count"] == 3 and receipt["inserted"] == 3 and receipt["duplicates_ignored"] == 0
    assert receipt["unit_id"] == "unit-0" and receipt["population_id"] == "EURUSD"
    assert receipt["rows_sha256"] == adapter_rows_digest(rows), "digest rule must equal the driver's"
    back = wh_file.read_run(RUN, "feature_pair_metrics", "unit-0")
    assert len(back) == 3 and adapter_rows_digest(back) == receipt["rows_sha256"]
    assert sorted(json.dumps(r, sort_keys=True) for r in back) == sorted(json.dumps(r, sort_keys=True) for r in rows)
    assert [r["row_key"] for r in back] == sorted(r["row_key"] for r in rows)
    assert wh_file.read_run(RUN, "feature_pair_metrics", "other-unit") == []
    assert len(wh_file.read_run(RUN, "feature_pair_metrics")) == 3
    recon = wh_file.reconcile(RUN)
    assert recon["run_id"] == RUN
    assert recon["tables"]["feature_pair_metrics"] == dict(
        recon["tables"]["feature_pair_metrics"], count=3, rows_sha256=receipt["rows_sha256"])
    for table in core.FACT_TABLES:
        assert set(recon["tables"][table]) >= {"count", "rows_sha256"}
    assert recon["tables"]["feature_pair_gate"]["rows_sha256"] == core.EMPTY_DIGEST


def test_all_seven_driver_tables_accept_their_row_shapes(wh_file):
    submissions = {
        "feature_pair_metrics": [metric_row("a", "b")],
        "feature_pair_stability": [stability_row("a", "b")],
        "feature_pair_gate": [gate_row("a", "b", gate_state="BYTE_IDENTICAL")],
        "feature_alias_groups": [{"run_id": RUN, "population_id": "EURUSD", "alias_group_id": HEX_B,
                                  "members": ["a", "b"], "representative": "a",
                                  "representative_rule": "clock OBSERVED, coverage desc",
                                  "evidence": [{"left": "a", "right": "b", "gate_state": "BYTE_IDENTICAL"}],
                                  "disposition": "ALIAS_GROUP", "dropped_columns": [],
                                  "row_key": row_key(RUN, "alias", HEX_B)}],
        "feature_redundancy_clusters": [{"run_id": RUN, "population_id": "EURUSD", "cluster_id": HEX_B,
                                         "members": ["a", "c"], "size": 2, "representative": "a",
                                         "rule": "average-linkage on 1-|spearman_TRAIN|",
                                         "row_key": row_key(RUN, "cluster", HEX_B)}],
        "feature_filter_rankings": [{"run_id": RUN, "population_id": "EURUSD", "target_id": "Y_s_1h",
                                     "horizon_hours": 1, "method": "mrmr_mi", "rank": 1, "feature_id": "a",
                                     "score": 0.9, "base_score": 0.8, "terms": {"relevance": 0.9},
                                     "causal_label": "NOT_IDENTIFIED", "params_sha256": HEX,
                                     "unit_id": "p3-unit", "row_key": row_key(RUN, "rank", "Y_s_1h", "mrmr_mi", 1)}],
        "feature_filter_subsets": [{"run_id": RUN, "population_id": "EURUSD", "target_id": "Y_s_1h",
                                    "horizon_hours": 1, "method": "mrmr_mi", "k": 4, "members": ["a", "c"],
                                    "subset_sha256": HEX_B, "label": "FILTER_CANDIDATE",
                                    "is_final_selection": False, "params_sha256": HEX, "unit_id": "p3-unit",
                                    "row_key": row_key(RUN, "subset", "Y_s_1h", "mrmr_mi", 4)}],
    }
    for table, rows in submissions.items():
        receipt = wh_file.submit_rows(RUN, table, rows)
        assert receipt["inserted"] == 1 and receipt["rows_sha256"] == adapter_rows_digest(rows), table
        assert wh_file.read_run(RUN, table) == rows, table
    recon = wh_file.reconcile(RUN)
    assert all(recon["tables"][t]["count"] == 1 for t in core.FACT_TABLES)
    stored = wh_file.query("SELECT representative, members_json FROM feature_alias_groups")
    assert stored == [{"representative": "a", "members_json": '["a","b"]'}]
    assert wh_file.query("SELECT k, label FROM feature_filter_subsets") == [{"k": 4, "label": "FILTER_CANDIDATE"}]


def test_restart_replay_is_a_no_op_with_the_same_digest(wh_file):
    rows = [metric_row("a", "b"), metric_row("a", "c")]
    first = wh_file.submit_rows(RUN, "feature_pair_metrics", rows)
    second = wh_file.submit_rows(RUN, "feature_pair_metrics", rows)
    assert second["inserted"] == 0 and second["duplicates_ignored"] == 2
    assert second["rows_sha256"] == first["rows_sha256"] and second["row_count"] == 2
    assert wh_file.query("SELECT count(*) AS n FROM feature_pair_metrics") == [{"n": 2}]


def test_column_order_does_not_change_identity_or_digest():
    row = metric_row("a", "b")
    shuffled = dict(reversed(list(row.items())))
    a, b = core.prepare_row("feature_pair_metrics", row), core.prepare_row("feature_pair_metrics", shuffled)
    assert a["row_identity_sha256"] == b["row_identity_sha256"] and a["row_sha256"] == b["row_sha256"]


def test_same_identity_different_content_rejects_the_whole_batch(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b", value=0.5)])
    with pytest.raises(wh.Refusal, match="different content"):
        wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "c"), metric_row("a", "b", value=0.6)])
    assert wh_file.query("SELECT count(*) AS n FROM feature_pair_metrics") == [{"n": 1}]
    assert wh_file.query("SELECT count(*) AS n FROM fs_phase23_load_receipt") == [{"n": 1}]
    with pytest.raises(wh.Refusal, match="one identity with two different contents"):
        wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("x", "y", value=0.1), metric_row("x", "y", value=0.2)])
    # the same row_key under a different typed identity is refused as well
    forged = metric_row("a", "d")
    forged["row_key"] = metric_row("a", "b")["row_key"]
    with pytest.raises(wh.Refusal, match="row_key"):
        wh_file.submit_rows(RUN, "feature_pair_metrics", [forged])


def test_foreign_run_and_population_identities_are_rejected(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    with pytest.raises(wh.Refusal, match="foreign run identity"):
        wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "c", run_id="phase2-eurusd:other")])
    with pytest.raises(wh.Refusal, match="foreign population identity"):
        wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "c", population="ETH")])
    with pytest.raises(wh.Refusal, match="one population"):
        wh_file.submit_rows("phase2-mixed:test", "feature_pair_metrics",
                            [metric_row("a", "b", run_id="phase2-mixed:test"),
                             metric_row("a", "c", run_id="phase2-mixed:test", population="ETH")])
    assert wh_file.query("SELECT count(*) AS n FROM feature_pair_metrics") == [{"n": 1}]
    with pytest.raises(wh.Refusal, match="not registered"):
        wh_file.reconcile("phase2-never:seen")


def test_a_receipt_carrying_a_foreign_identity_is_rejected(wh_file):
    receipt = wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    assert wh_file.verify_receipt(receipt)["accepted"]

    def resign(doc):
        doc = dict(doc)
        doc["receipt_sha256"] = core.digest({k: v for k, v in doc.items() if k != "receipt_sha256"})
        return doc
    with pytest.raises(wh.Refusal, match="foreign population"):
        wh_file.verify_receipt(resign(dict(receipt, population_id="ETH")))
    with pytest.raises(wh.Refusal, match="digest does not match"):
        wh_file.verify_receipt(dict(receipt, rows_sha256=HEX_B))
    with pytest.raises(wh.Refusal, match="unknown run"):
        wh_file.verify_receipt(resign(dict(receipt, run_id="phase2-eth:test")))
    with pytest.raises(wh.Refusal, match="no such receipt"):
        wh_file.verify_receipt(resign(dict(receipt, submitted_at="2020-01-01T00:00:00Z")))


def test_contract_binding_upgrades_a_first_submission_run_but_never_its_population(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    run = core.get_run(wh_file.conn, RUN)
    assert run["registration"] == "FIRST_SUBMISSION" and run["population_id"] == "EURUSD"
    bound = {"run_id": RUN, "population_id": "EURUSD", "phase": "PHASE_2", "contract_sha256": HEX,
             "campaign_sha256": HEX, "code_sha256": HEX, "input_sha256": HEX,
             "expected_json": {"feature_pair_metrics": 2}}
    assert wh_file.register_run(bound)["upgraded"]
    assert wh_file.register_run(bound)["already_registered"]
    with pytest.raises(wh.Refusal, match="different identity"):
        wh_file.register_run(dict(bound, population_id="ETH"))
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "c")])
    assert wh_file.reconcile(RUN)["complete"]
    assert wh_file.read_run(RUN, "feature_pair_metrics")[0]["run_id"] == RUN


# ----------------------------------------------------------------------------- row validation
def test_unordered_pair_closed_splits_and_host_names_are_refused():
    with pytest.raises(wh.Refusal, match="ordered"):
        core.prepare_row("feature_pair_metrics", metric_row("b", "a"))
    with pytest.raises(wh.Refusal, match="closed"):
        core.prepare_row("feature_pair_metrics", metric_row("a", "b", fold="validation"))
    with pytest.raises(wh.Refusal, match="role"):
        core.prepare_row("feature_pair_metrics", metric_row("a", "b", host_role="some-host-name"))
    with pytest.raises(wh.Refusal, match="row_key"):
        core.prepare_row("feature_pair_metrics", dict(metric_row("a", "b"), row_key=None))
    with pytest.raises(wh.Refusal, match="state"):
        core.prepare_row("feature_pair_metrics", metric_row("a", "b", state="GUESSED"))
    with pytest.raises(wh.Refusal, match="required"):
        core.prepare_row("feature_pair_metrics", dict(metric_row("a", "b"), params_sha256=None))
    abstained = core.prepare_row("feature_pair_metrics",
                                 metric_row("a", "b", value=None, state="INSUFFICIENT_SUPPORT", support=3))
    assert abstained["state"] == "INSUFFICIENT_SUPPORT" and abstained["metric_value"] is None
    assert abstained["shared_support"] == 3


# ----------------------------------------------------------------------------- reconciliation
def test_closure_reconciliation_fails_on_one_missing_disposition(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b"), metric_row("a", "c")])
    report = wh_file.reconcile(RUN, expected={"feature_pair_metrics": 3})
    assert not report["complete"]
    assert report["tables"]["feature_pair_metrics"]["count_matches_expected"] is False
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("b", "c")])
    assert wh_file.reconcile(RUN, expected={"feature_pair_metrics": 3})["complete"]


def test_rows_without_a_receipt_are_detected(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    smuggled = core.prepare_row("feature_pair_metrics", metric_row("a", "c"))
    cols = core.columns("feature_pair_metrics")
    wh_file.conn.execute(f"INSERT INTO feature_pair_metrics ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [smuggled[c] for c in cols])
    report = wh_file.reconcile(RUN)
    assert report["tables"]["feature_pair_metrics"]["receipts_cover_store"] is False and not report["complete"]


def test_sql_digest_equals_python_digest(wh_file):
    rows = [metric_row("a", "b"), metric_row("a", "c"), metric_row("b", "c")]
    receipt = wh_file.submit_rows(RUN, "feature_pair_metrics", rows)
    sha = wh_file.conn.execute("SELECT sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) "
                               "FROM feature_pair_metrics").fetchone()[0]
    assert sha == receipt["rows_sha256"] == core.rows_sha256(rows) == adapter_rows_digest(rows)


def test_database_unique_keys_block_a_duplicate_by_any_path(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    cols = core.columns("feature_pair_metrics")
    rec = core.prepare_row("feature_pair_metrics", metric_row("a", "b", value=0.9))
    rec["row_identity_sha256"] = HEX_B  # forged PK: the typed UNIQUE key still holds
    with pytest.raises(duckdb.ConstraintException):
        wh_file.conn.execute(f"INSERT INTO feature_pair_metrics ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             [rec[c] for c in cols])
    rec = core.prepare_row("feature_pair_metrics", metric_row("a", "d"))
    rec["row_key"] = metric_row("a", "b")["row_key"]  # same driver key: UNIQUE (run_id, row_key)
    with pytest.raises(duckdb.ConstraintException):
        wh_file.conn.execute(f"INSERT INTO feature_pair_metrics ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             [rec[c] for c in cols])


# ----------------------------------------------------------------------------- readback report
def test_readback_report_groups_by_asset_method_and_host_role(wh_file):
    wh_file.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b"), metric_row("a", "c", metric="pearson")],
                        host_role="worker_a")
    wh_file.submit_rows(RUN, "feature_pair_metrics",
                        [metric_row("b", "c", state="INSUFFICIENT_SUPPORT", value=None, support=2)], host_role="worker_b")
    wh_file.submit_rows("phase2-eth:test", "feature_pair_metrics",
                        [metric_row("x", "y", run_id="phase2-eth:test", population="ETH")], host_role="coordinator")
    report = wh.readback_report(wh_file.query)
    pm = report["tables"]["feature_pair_metrics"]
    assert pm["total"] == 4
    assert pm["by_asset"] == {"ETH": 1, "EURUSD": 3}
    assert pm["by_method"] == {"pairwise_v1": 4}
    assert pm["by_host_role"] == {"coordinator": 1, "worker_a": 2, "worker_b": 1}
    group = next(g for g in pm["groups"] if g["host_role"] == "worker_b")
    assert group["insufficient_support"] == 1 and group["measured"] == 0
    only_eth = wh.readback_report(wh_file.query, "phase2-eth:test")
    assert only_eth["tables"]["feature_pair_metrics"]["by_asset"] == {"ETH": 1}
    assert wh.compare_readback(report, report)["agree"]
    other = json.loads(json.dumps(report))
    other["tables"]["feature_pair_metrics"]["groups"][0]["n"] += 1
    diff = wh.compare_readback(report, other)
    assert not diff["agree"] and diff["differences"][0]["table"] == "feature_pair_metrics"


def test_service_transport_needs_a_token_from_the_environment(monkeypatch):
    monkeypatch.delenv("WAREHOUSE_TOKEN", raising=False)
    with pytest.raises(wh.Refusal, match="never from an argument"):
        wh.open_warehouse("http://127.0.0.1:1")


# ----------------------------------------------------------------------------- the packaged backend
def test_the_packaged_duckdb_backend_serves_the_same_semantics(tmp_path):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("duckdb_engine")
    from predictor_duckdb_store.provider import PredictorDuckdbStore
    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(tmp_path / "backend.duckdb"), schema="main", min_free_bytes=0)
    for capability in ("write_fs_phase23_rows", "read_fs_phase23_rows", "reconcile_fs_phase23"):
        assert capability in store.capabilities()
    rows = [metric_row("a", "b"), metric_row("a", "c")]
    receipt = store.write_fs_phase23_rows({"run_id": RUN, "table": "feature_pair_metrics", "rows": rows,
                                           "host_role": "coordinator"})
    assert receipt["inserted"] == 2 and receipt["rows_sha256"] == adapter_rows_digest(rows)
    again = store.write_fs_phase23_rows({"run_id": RUN, "table": "feature_pair_metrics", "rows": rows})
    assert again["inserted"] == 0 and again["duplicates_ignored"] == 2
    with pytest.raises(ValueError, match="foreign population"):
        store.write_fs_phase23_rows({"run_id": RUN, "table": "feature_pair_metrics",
                                     "rows": [metric_row("a", "d", population="ETH")]})
    page = store.read_fs_phase23_rows({"run_id": RUN, "table": "feature_pair_metrics", "limit": 1})
    assert page["count"] == 1 and page["next_after"] == page["rows"][0]["row_key"]
    rest = store.read_fs_phase23_rows({"run_id": RUN, "table": "feature_pair_metrics", "after": page["next_after"]})
    assert rest["next_after"] is None and adapter_rows_digest(page["rows"] + rest["rows"]) == receipt["rows_sha256"]
    recon = store.reconcile_fs_phase23({"run_id": RUN, "receipts": [receipt]})
    assert recon["tables"]["feature_pair_metrics"]["count"] == 2
    assert recon["tables"]["feature_pair_metrics"]["rows_sha256"] == receipt["rows_sha256"]
    assert recon["receipts_verified"] == 1
    store.engine().dispose()
    # the file written by the backend is readable by the local tool with identical digests
    local = wh.Warehouse(tmp_path / "backend.duckdb", read_only=True)
    assert local.reconcile(RUN)["tables"]["feature_pair_metrics"]["rows_sha256"] == receipt["rows_sha256"]
    local.close()


# ----------------------------------------------------------------------------- snapshot
def _have_zstd():
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return shutil.which("zstd") is not None


@pytest.mark.skipif(not _have_zstd(), reason="no zstd available")
def test_snapshot_writes_manifest_and_verifies_its_own_asset(tmp_path):
    source = tmp_path / "copy.duckdb"
    store = wh.open_warehouse(source)
    store.submit_rows(RUN, "feature_pair_metrics", [metric_row("a", "b")])
    store.conn.execute("CHECKPOINT")
    store.close()
    manifest = wh.snapshot(source, tmp_path / "out", tag="t-test", repo="owner/repo", phase="TEST",
                           zstd_bin=None, level=3)
    assert manifest["relations"]["feature_pair_metrics"]["rows"] == 1
    assert manifest["missing_relations"] == []
    assert manifest["asset_url"] == "https://github.com/owner/repo/releases/download/t-test/copy.duckdb.zst"
    asset = tmp_path / "out" / "copy.duckdb.zst"
    assert wh.file_sha256(asset) == manifest["compressed_sha256"]
    result = wh.verify_snapshot_asset(asset, manifest, zstd_bin=None, work_dir=tmp_path / "restore")
    assert result["verified"] and result["relations"]["feature_pair_metrics"] == 1
    asset.write_bytes(asset.read_bytes() + b"x")
    with pytest.raises(wh.Refusal, match="does not match"):
        wh.verify_snapshot_asset(asset, manifest, zstd_bin=None, work_dir=tmp_path / "restore2")


def test_snapshot_refuses_a_live_store(tmp_path):
    source = tmp_path / "cube.duckdb"
    source.write_bytes(b"not opened")
    (tmp_path / "cube.duckdb.wal").write_bytes(b"pending transactions")
    with pytest.raises(wh.Refusal, match="live store"):
        wh.snapshot(source, tmp_path / "out", tag="t", repo="o/r", phase="TEST", zstd_bin=None)


# ----------------------------------------------------------------------------- CLI
def test_cli_migrate_dry_run_creates_nothing(tmp_path, capsys):
    target = tmp_path / "x.duckdb"
    assert wh.main(["migrate", "--duckdb", str(target), "--dry-run"]) == 0
    assert not target.exists()
    assert "would_create" in capsys.readouterr().out
    assert wh.main(["migrate", "--duckdb", str(target)]) == 0
    assert target.exists()
    assert wh.main(["verify-contract", str(CONTRACT)]) == 0
