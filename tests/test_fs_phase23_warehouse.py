"""Acceptance tests for tools/fs_phase23_warehouse.py on THROWAWAY DuckDB files.

Subplan §6 points covered here (the engineering agent covers the numerical ones):
  * a receipt / row with a foreign run or population identity is rejected;
  * a restart (identical replay) produces the same digests and duplicates no row;
  * column order does not change identity or digests;
  * the closure reconciliation fails when one expected disposition is missing;
  * the 12 EURUSD and 3 ETH phase-1 features are labelled CAUSAL_SUPPORTED, not selected;
  * the migration is additive and idempotent; the SQL file is the only DDL source;
  * local and SQL digests agree, so a terminal and the warehouse share one arithmetic;
  * the snapshot procedure refuses a live store and verifies its own asset.

No test opens the configured warehouse, contacts the service or restarts anything.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fs_phase23_warehouse as wh  # noqa: E402

CONTRACT = ROOT / "docs/audits/evidence/canonical_20261003/fs_phase23/data/CONTRACT.json"
HEX = "a" * 64
HEX_B = "b" * 64


@pytest.fixture
def conn(tmp_path):
    connection = duckdb.connect(str(tmp_path / "throwaway.duckdb"),
                                config={"memory_limit": "128MB", "threads": 1})
    wh.apply_migration(connection)
    yield connection
    connection.close()


def run_doc(run_id="phase2-eurusd:test", population="EURUSD", **expected):
    return {"run_id": run_id, "population_id": population, "phase": "PHASE_2",
            "contract_sha256": HEX, "campaign_sha256": HEX, "code_sha256": HEX,
            "input_sha256": HEX, "expected_json": expected}


def pair_row(left, right, *, run_id="phase2-eurusd:test", population="EURUSD", fold="inner_2019",
             lag=0, method="spearman", value=0.5, state="MEASURED", support=1000, **extra):
    row = {"run_id": run_id, "population_id": population, "feature_left": left,
           "feature_right": right, "fold": fold, "lag": lag, "method": method,
           "params_sha256": HEX, "code_sha256": HEX, "input_sha256": HEX,
           "shared_support": support, "metric_value": value, "state": state}
    row.update(extra)
    return row


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
def test_migration_is_additive_idempotent_and_sourced_from_the_sql_file(tmp_path):
    connection = duckdb.connect(str(tmp_path / "m.duckdb"))
    first = wh.apply_migration(connection)
    assert set(wh.ALL_TABLES) <= set(first["created"])
    second = wh.apply_migration(connection)
    assert second["created"] == []
    assert sorted(second["already_present"]) == sorted(wh.ALL_TABLES)
    plan = wh.migration_plan(connection)
    assert plan["would_create"] == [] and plan["destructive_statements"] == 0
    for statement in wh.load_ddl():
        assert statement.upper().startswith(("CREATE TABLE IF NOT EXISTS", "CREATE OR REPLACE VIEW"))
        assert "DROP" not in statement.upper() and "ALTER" not in statement.upper()
    connection.close()


def test_migration_refuses_a_non_additive_statement(tmp_path):
    bad = tmp_path / "bad.sql"
    bad.write_text(wh.MIGRATION_SQL.read_text() + "\nDROP TABLE feature_pair_metrics;\n")
    connection = duckdb.connect(str(tmp_path / "m.duckdb"))
    with pytest.raises(wh.Refusal, match="additive only"):
        wh.apply_migration(connection, bad)
    connection.close()


def test_unique_key_in_the_schema_covers_every_identity_column():
    sql = wh.MIGRATION_SQL.read_text()
    for table, identity in wh.IDENTITY.items():
        block = sql.split(f"CREATE TABLE IF NOT EXISTS {table} (")[1].split(");")[0]
        unique = block.split("UNIQUE (")[1].split(")")[0]
        assert {c.strip() for c in unique.split(",")} == set(identity), table


# ----------------------------------------------------------------------------- submit / receipts
def test_submit_returns_receipt_and_readback_matches(conn):
    wh.register_run(conn, run_doc(feature_pair_metrics=2))
    rows = [pair_row("a", "b"), pair_row("a", "c", value=0.1)]
    receipt = wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", rows,
                             host_role="worker_b", shard_id="shard-000")
    assert receipt["stored_new"] == 2 and receipt["already_stored"] == 0
    assert receipt["population_id"] == "EURUSD" and receipt["host_role"] == "worker_b"
    back = wh.read_run(conn, "phase2-eurusd:test", "feature_pair_metrics")
    assert [(r["feature_left"], r["feature_right"]) for r in back] == sorted(
        [(r["feature_left"], r["feature_right"]) for r in back])
    assert len(back) == 2 and all(r["host_role"] == "worker_b" for r in back)
    assert wh.verify_receipt(conn, receipt)["accepted"]
    report = wh.reconcile(conn, "phase2-eurusd:test", [receipt])
    assert report["complete"] and report["tables"]["feature_pair_metrics"]["stored"] == 2
    assert report["tables"]["feature_pair_metrics"]["stored_rows_sha256"] == receipt["rows_sha256"]


def test_restart_replay_is_a_no_op_with_the_same_digest(conn):
    wh.register_run(conn, run_doc())
    rows = [pair_row("a", "b"), pair_row("a", "c")]
    first = wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", rows)
    second = wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", rows)
    assert second["stored_new"] == 0 and second["already_stored"] == 2
    assert second["rows_sha256"] == first["rows_sha256"]
    assert conn.execute("SELECT count(*) FROM feature_pair_metrics").fetchone()[0] == 2


def test_column_order_does_not_change_identity_or_digest():
    row = pair_row("a", "b")
    shuffled = dict(reversed(list(row.items())))
    a, b = wh.normalise_row("feature_pair_metrics", row), wh.normalise_row("feature_pair_metrics", shuffled)
    assert a["row_identity_sha256"] == b["row_identity_sha256"]
    assert a["row_sha256"] == b["row_sha256"]


def test_same_identity_different_content_rejects_the_whole_batch(conn):
    wh.register_run(conn, run_doc())
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b", value=0.5)])
    with pytest.raises(wh.Refusal, match="different content"):
        wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                       [pair_row("a", "c"), pair_row("a", "b", value=0.6)])
    assert conn.execute("SELECT count(*) FROM feature_pair_metrics").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM fs_phase23_load_receipt").fetchone()[0] == 1
    with pytest.raises(wh.Refusal, match="one identity with two different contents"):
        wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                       [pair_row("x", "y", value=0.1), pair_row("x", "y", value=0.2)])


def test_foreign_run_and_population_identities_are_rejected(conn):
    wh.register_run(conn, run_doc())
    with pytest.raises(wh.Refusal, match="foreign run identity"):
        wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                       [pair_row("a", "b", run_id="phase2-eurusd:other")])
    with pytest.raises(wh.Refusal, match="foreign population identity"):
        wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                       [pair_row("a", "b", population="ETH")])
    with pytest.raises(wh.Refusal, match="not registered"):
        wh.submit_rows(conn, "phase2-eth:unregistered", "feature_pair_metrics",
                       [pair_row("a", "b", run_id="phase2-eth:unregistered", population="ETH")])
    assert conn.execute("SELECT count(*) FROM feature_pair_metrics").fetchone()[0] == 0


def test_a_receipt_carrying_a_foreign_identity_is_rejected(conn):
    wh.register_run(conn, run_doc())
    receipt = wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b")])
    foreign = dict(receipt, population_id="ETH")
    foreign["receipt_sha256"] = wh.digest({k: v for k, v in foreign.items()
                                           if k not in ("submitted_at", "receipt_sha256")})
    with pytest.raises(wh.Refusal, match="foreign population"):
        wh.verify_receipt(conn, foreign)
    tampered = dict(receipt, rows_sha256=HEX_B)
    with pytest.raises(wh.Refusal, match="digest does not match"):
        wh.verify_receipt(conn, tampered)
    wh.register_run(conn, run_doc(run_id="phase2-eth:test", population="ETH"))
    other_run = dict(receipt, run_id="phase2-eth:test")
    other_run["receipt_sha256"] = wh.digest({k: v for k, v in other_run.items()
                                             if k not in ("submitted_at", "receipt_sha256")})
    with pytest.raises(wh.Refusal):
        wh.verify_receipt(conn, other_run)


def test_run_registration_is_immutable(conn):
    wh.register_run(conn, run_doc())
    assert wh.register_run(conn, run_doc())["already_registered"]
    with pytest.raises(wh.Refusal, match="different identity"):
        wh.register_run(conn, run_doc(feature_pair_metrics=5))


# ----------------------------------------------------------------------------- row validation
def test_unordered_pair_wrong_split_and_host_names_are_refused():
    with pytest.raises(wh.Refusal, match="ordered"):
        wh.normalise_row("feature_pair_metrics", pair_row("b", "a"))
    with pytest.raises(wh.Refusal, match="split"):
        wh.normalise_row("feature_pair_metrics", pair_row("a", "b", split="validation"))
    with pytest.raises(wh.Refusal, match="role"):
        wh.normalise_row("feature_pair_metrics", pair_row("a", "b", host_role="some-host-name"))
    with pytest.raises(wh.Refusal, match="MEASURED"):
        wh.normalise_row("feature_pair_metrics", pair_row("a", "b", value=None))
    with pytest.raises(wh.Refusal, match="unknown columns"):
        wh.normalise_row("feature_pair_metrics", pair_row("a", "b", hostname="x"))
    abstained = wh.normalise_row("feature_pair_metrics",
                                 pair_row("a", "b", value=None, state="INSUFFICIENT_SUPPORT", support=3))
    assert abstained["state"] == "INSUFFICIENT_SUPPORT" and abstained["metric_value"] is None


def test_other_families_validate_their_vocabularies(conn):
    wh.register_run(conn, run_doc())
    base = {"run_id": "phase2-eurusd:test", "population_id": "EURUSD", "fold": "ALL_TRAIN",
            "method": "identity_gate", "params_sha256": HEX, "code_sha256": HEX, "input_sha256": HEX}
    alias = dict(base, group_id="g1", feature_id="a", disposition="ALIAS_BYTE_EXACT", representative=True)
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_alias_groups", [alias])
    with pytest.raises(wh.Refusal, match="disposition"):
        wh.normalise_row("feature_alias_groups", dict(alias, disposition="DELETED"))
    cluster = dict(base, method="hierarchical_spearman_abs", cluster_id="c1", feature_id="a",
                   representative=False, linkage_distance=0.2, threshold=0.3)
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_redundancy_clusters", [cluster])
    rank = dict(base, method="mrmr_mi", target_id="Y_s_1h", horizon=1, feature_id="a", rank=1,
                score=0.9, state="MEASURED", k_membership_json=[4, 8])
    stored = wh.normalise_row("feature_filter_rankings", rank)
    assert stored["k_membership_json"] == "[4,8]"
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_filter_rankings", [rank])
    with pytest.raises(wh.Refusal, match="method"):
        wh.normalise_row("feature_filter_rankings", dict(rank, method="winner"))
    with pytest.raises(wh.Refusal, match="1-based"):
        wh.normalise_row("feature_filter_rankings", dict(rank, rank=0))


# ----------------------------------------------------------------------------- reconciliation
def test_closure_reconciliation_fails_on_one_missing_disposition(conn):
    wh.register_run(conn, run_doc(feature_pair_metrics=3))
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b"), pair_row("a", "c")])
    report = wh.reconcile(conn, "phase2-eurusd:test")
    assert not report["complete"]
    assert report["tables"]["feature_pair_metrics"]["expected"] == 3
    assert report["tables"]["feature_pair_metrics"]["count_matches_expected"] is False
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("b", "c")])
    assert wh.reconcile(conn, "phase2-eurusd:test")["complete"]


def test_rows_without_a_receipt_are_detected(conn):
    wh.register_run(conn, run_doc(feature_pair_metrics=2))
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b")])
    smuggled = wh.normalise_row("feature_pair_metrics", pair_row("a", "c"))
    cols = wh.COLUMNS["feature_pair_metrics"]
    conn.execute(f"INSERT INTO feature_pair_metrics ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                 [smuggled[c] for c in cols])
    report = wh.reconcile(conn, "phase2-eurusd:test")
    assert report["tables"]["feature_pair_metrics"]["count_matches_expected"] is True
    assert report["tables"]["feature_pair_metrics"]["receipts_cover_store"] is False
    assert not report["complete"]


def test_sql_digest_equals_python_digest(conn):
    wh.register_run(conn, run_doc())
    receipt = wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                             [pair_row("a", "b"), pair_row("a", "c"), pair_row("b", "c")])
    sha = conn.execute("SELECT sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) "
                       "FROM feature_pair_metrics").fetchone()[0]
    assert sha == receipt["rows_sha256"]
    rows = conn.execute("SELECT row_sha256 FROM feature_pair_metrics").fetchall()
    assert wh.rows_digest(r[0] for r in rows) == sha


def test_database_unique_key_blocks_a_duplicate_identity_by_any_path(conn):
    wh.register_run(conn, run_doc())
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b")])
    row = wh.normalise_row("feature_pair_metrics", pair_row("a", "b", value=0.9))
    row["row_identity_sha256"] = HEX_B  # even with a forged primary key the UNIQUE key holds
    cols = wh.COLUMNS["feature_pair_metrics"]
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(f"INSERT INTO feature_pair_metrics ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [row[c] for c in cols])


# ----------------------------------------------------------------------------- readback report
def test_readback_report_groups_by_asset_method_and_host_role(conn):
    wh.register_run(conn, run_doc())
    wh.register_run(conn, run_doc(run_id="phase2-eth:test", population="ETH"))
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                   [pair_row("a", "b"), pair_row("a", "c", method="pearson")], host_role="worker_a")
    wh.submit_rows(conn, "phase2-eurusd:test", "feature_pair_metrics",
                   [pair_row("b", "c", state="INSUFFICIENT_SUPPORT", value=None, support=2)], host_role="worker_b")
    wh.submit_rows(conn, "phase2-eth:test", "feature_pair_metrics",
                   [pair_row("x", "y", run_id="phase2-eth:test", population="ETH")], host_role="coordinator")
    report = wh.readback_report(wh.local_query(conn))
    pm = report["tables"]["feature_pair_metrics"]
    assert pm["total"] == 4
    assert pm["by_asset"] == {"ETH": 1, "EURUSD": 3}
    assert pm["by_method"] == {"pearson": 1, "spearman": 3}
    assert pm["by_host_role"] == {"coordinator": 1, "worker_a": 2, "worker_b": 1}
    group = next(g for g in pm["groups"] if g["host_role"] == "worker_b")
    assert group["insufficient_support"] == 1 and group["measured"] == 0
    only_eth = wh.readback_report(wh.local_query(conn), "phase2-eth:test")
    assert only_eth["tables"]["feature_pair_metrics"]["by_asset"] == {"ETH": 1}
    assert wh.compare_readback(report, report)["agree"]
    other = json.loads(json.dumps(report))
    other["tables"]["feature_pair_metrics"]["groups"][0]["n"] += 1
    diff = wh.compare_readback(report, other)
    assert not diff["agree"] and diff["differences"][0]["table"] == "feature_pair_metrics"


def test_service_transport_needs_a_token_from_the_environment(monkeypatch):
    monkeypatch.delenv("WAREHOUSE_TOKEN", raising=False)
    with pytest.raises(wh.Refusal, match="never from an argument"):
        wh.Service("http://127.0.0.1:1")


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
    connection = duckdb.connect(str(source))
    wh.apply_migration(connection)
    wh.register_run(connection, run_doc())
    wh.submit_rows(connection, "phase2-eurusd:test", "feature_pair_metrics", [pair_row("a", "b")])
    connection.execute("CHECKPOINT")
    connection.close()
    manifest = wh.snapshot(source, tmp_path / "out", tag="t-test", repo="owner/repo", phase="TEST",
                           zstd_bin=None, level=3)
    assert manifest["relations"]["feature_pair_metrics"]["rows"] == 1
    assert manifest["missing_relations"] == []
    assert manifest["asset_url"] == "https://github.com/owner/repo/releases/download/t-test/copy.duckdb.zst"
    assert (tmp_path / "out" / "SNAPSHOT_MANIFEST.json").exists()
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
