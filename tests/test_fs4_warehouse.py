"""Acceptance tests for the phase-4 warehouse (olap/store fs4_store + tools/fs4_warehouse.py) on
THROWAWAY DuckDB files.

Plan §6 points covered here:
  * FS4-06: a missing numeric, an absent population/identity field, NaN/inf, a wrong seed or arm
    and an unplanned task_id are refused before any write (Python rule and DB CHECK agree);
  * FS4-07: an identical replay is a no-op with the same digests; the same task with another
    terminal is refused (a restart adopts, never rewrites);
  * FS4-09: the three arms of one feature x fold must share rows/mask/input/population_n/naive_mae,
    in a batch and against what is stored; reconcile reports a stored disagreement;
  * FS4-13: every metric written is read back unchanged, readback and reconcile digests equal
    the Python rule, receipts verify, the migration is additive/idempotent, the .sql file equals
    the code, the snapshot boundary names the relations, and the packaged DuckDB backend serves
    the same semantics the service will.

No test opens the configured warehouse, contacts the service or restarts anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "olap" / "store" / "src", ROOT / "olap" / "duckdb_store" / "src", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for name in [n for n in sys.modules if n == "predictor_olap_store" or n.startswith("predictor_olap_store.")]:
    del sys.modules[name]  # an installed older package must not shadow this checkout

import fs4_warehouse as wh  # noqa: E402
from predictor_olap_store import fs4_store as core  # noqa: E402

PLAN = "f" * 64
HEX = {k: c * 64 for k, c in zip(core.RESULT_DIGESTS, "abcde")}


def task(arm, feature="px.rv5", fold="inner_2023", population="EURUSD", identity="train-v1"):
    return {"schema": core.SCHEMA_TASK, "population_id": population, "identity": identity,
            "feature_id": feature, "fold_id": fold, "arm": arm, "seed": 0}


def terminal(arm, *, feature="px.rv5", fold="inner_2023", population="EURUSD", mae=0.1, naive=0.2, **over):
    t = task(arm, feature, fold, population)
    tid = core.task_id_of(t)
    result = {"task_id": tid, "status": "COMPLETE", "seed": 0, **HEX, "population_n": 20,
              "metrics": {"mae": mae, "naive_mae": naive, "mse": 0.01},
              "cost": {"cpu_s": 1.5, "wall_s": 2.0, "peak_ram_bytes": 1000, "peak_vram_bytes": 0},
              "training": {"chosen_epoch": 3, "updates": 40}}
    result.update(over)
    return {"plan_sha256": PLAN, "task": {**t, "task_id": tid}, "result": result,
            "owner": "worker_a-gpu1", "attempt": 1, "started_at": 90.0, "finished_at": 100.0}


def python_terminals_digest(terminals):
    return core.terminals_digest(hashlib.sha256(core.canonical_bytes(t["result"])).hexdigest() for t in terminals)


@pytest.fixture
def store(tmp_path):
    w = wh.open_warehouse(tmp_path / "throwaway.duckdb")
    yield w
    w.close()


# ----------------------------------------------------------------------------- migration (FS4-13)
def test_migration_is_additive_idempotent_and_the_sql_file_equals_the_code(tmp_path):
    first = wh.Warehouse(tmp_path / "m.duckdb")
    assert set(core.ALL_RELATIONS) <= set(first.migration["created"]) | set(first.migration["already_present"])
    first.close()
    second = wh.Warehouse(tmp_path / "m.duckdb")
    assert second.migration["created"] == []
    assert sorted(second.migration["already_present"]) == sorted(core.ALL_RELATIONS)
    second.close()
    assert wh.MIGRATION_SQL.is_file() and wh.migration_file_matches_code()
    assert all(s.upper().startswith(("CREATE TABLE IF NOT EXISTS", "CREATE OR REPLACE VIEW")) for s in core.ddl())
    assert not any(word in s.upper() for s in core.ddl() for word in ("DROP", "ALTER", "DELETE", "TRUNCATE"))


def test_a_stale_sql_file_is_refused(tmp_path, monkeypatch):
    stale = tmp_path / "stale.sql"
    stale.write_text("CREATE TABLE IF NOT EXISTS feature_extractibility_v1 (x INTEGER);\n")
    conn = duckdb.connect(str(tmp_path / "x.duckdb"))
    with pytest.raises(core.Refusal, match="does not match"):
        wh.apply_migration(conn, stale)
    conn.close()


def test_sqlite_dialect_renders_without_isfinite():
    statements = core.ddl(dialect="sqlite")
    assert "isfinite" not in " ".join(statements)
    assert "CREATE VIEW IF NOT EXISTS" in statements[-1]


def test_snapshot_boundary_names_the_fs4_relations():
    import olap_duckdb_migrate as mig
    for relation in core.ALL_RELATIONS:
        assert relation in mig.SNAPSHOT_RELATIONS


# ----------------------------------------------------------------------------- FS4-06
@pytest.mark.parametrize("mutate, match", [
    (lambda r: r["metrics"].__setitem__("mae", float("nan")), "finite"),
    (lambda r: r["metrics"].__setitem__("naive_mae", float("inf")), "finite"),
    (lambda r: r["metrics"].__setitem__("mae", -0.1), "finite and non-negative"),
    (lambda r: r["metrics"].pop("naive_mae"), "naive_mae is required"),
    (lambda r: r.pop("metrics"), "metrics"),
    (lambda r: r.__setitem__("population_n", 0), "population_n"),
    (lambda r: r.__setitem__("population_n", 20.0), "population_n"),
    (lambda r: r.__setitem__("rows_sha256", "zz"), "rows_sha256"),
    (lambda r: r.__setitem__("seed", 1), "seed"),
    (lambda r: r.__setitem__("status", "FAILED"), "COMPLETE"),
    (lambda r: r["metrics"].__setitem__("mse", float("nan")), "finite"),
    (lambda r: r["cost"].__setitem__("cpu_s", float("nan")), "finite"),
])
def test_fs4_06_missing_numeric_or_non_finite_results_are_refused(store, mutate, match):
    doc = terminal("RAW")
    mutate(doc["result"])
    with pytest.raises(core.Refusal, match=match):
        store.submit_terminals(PLAN, [doc])
    assert store.readback_summary(PLAN)["count"] == 0


@pytest.mark.parametrize("field", ["population_id", "identity", "feature_id", "fold_id", "arm"])
def test_fs4_06_an_absent_task_field_is_refused(store, field):
    doc = terminal("RAW")
    doc["task"].pop(field)
    with pytest.raises(core.Refusal, match=f"task.{field} is required"):
        store.submit_terminals(PLAN, [doc])


def test_fs4_06_an_unplanned_task_id_wrong_arm_or_closed_fold_is_refused(store):
    doc = terminal("RAW")
    doc["task"]["task_id"] = "0" * 64
    with pytest.raises(core.Refusal, match="not a planned task"):
        store.submit_terminals(PLAN, [doc])
    doc = terminal("RAW")
    doc["task"]["arm"] = "FANCY"
    doc["task"]["task_id"] = core.task_id_of(doc["task"])
    with pytest.raises(core.Refusal, match="task.arm"):
        store.submit_terminals(PLAN, [doc])
    doc = terminal("RAW", fold="test")
    with pytest.raises(core.Refusal, match="stay closed"):
        store.submit_terminals(PLAN, [doc])
    with pytest.raises(core.Refusal, match="no host names"):
        store.submit_terminals(PLAN, [terminal("RAW")], host_role="some-machine")
    with pytest.raises(core.Refusal, match="no terminals"):
        store.submit_terminals(PLAN, [])


def test_fs4_06_the_database_check_constraints_refuse_what_the_python_rule_refuses(store):
    rec = core.prepare_terminal(terminal("RAW"))
    cols = ", ".join(core.COLUMNS)
    marks = ", ".join("?" * len(core.COLUMNS))
    for column, value in (("mae", float("nan")), ("naive_mae", float("inf")), ("mae", -1.0),
                          ("seed", 1), ("arm", "FANCY"), ("population_n", 0), ("host_role", "some-machine")):
        bad = dict(rec, **{column: value, "task_id": hashlib.sha256(column.encode()).hexdigest()})
        with pytest.raises(duckdb.Error):
            store.conn.execute(f"INSERT INTO {core.TABLE} ({cols}) VALUES ({marks})", [bad[c] for c in core.COLUMNS])
    assert store.readback_summary(PLAN)["count"] == 0


# ----------------------------------------------------------------------------- FS4-07
def test_fs4_07_replay_is_a_no_op_and_a_rewritten_terminal_is_refused(store):
    docs = [terminal("RAW"), terminal("RANDOM_ENCODER")]
    first = store.submit_terminals(PLAN, docs, host_role="coordinator")
    assert first["inserted"] == 2 and first["duplicates_ignored"] == 0
    assert first["terminals_sha256"] == python_terminals_digest(docs)
    again = store.submit_terminals(PLAN, docs)
    assert again["inserted"] == 0 and again["duplicates_ignored"] == 2
    assert again["terminals_sha256"] == first["terminals_sha256"]
    assert store.readback_summary(PLAN) == {"count": 2, "terminals_sha256": first["terminals_sha256"]}
    changed = terminal("RAW", mae=0.05)
    with pytest.raises(core.Refusal, match="different terminal"):
        store.submit_terminals(PLAN, [changed])
    assert store.read_terminals(PLAN, arm="RAW")[0]["result"]["metrics"]["mae"] == 0.1


def test_fs4_07_the_batch_cannot_carry_one_task_twice_with_different_content(store):
    with pytest.raises(core.Refusal, match="two different terminals"):
        store.submit_terminals(PLAN, [terminal("RAW"), terminal("RAW", mae=0.3)])
    assert store.readback_summary(PLAN)["count"] == 0


def test_fs4_07_a_foreign_plan_is_refused_before_any_write(store):
    doc = terminal("RAW")
    doc["plan_sha256"] = "e" * 64
    with pytest.raises(core.Refusal, match="foreign plan"):
        store.submit_terminals(PLAN, [doc])


# ----------------------------------------------------------------------------- FS4-09
def test_fs4_09_arms_must_share_rows_mask_input_population_n_and_naive_mae(store):
    store.submit_terminals(PLAN, [terminal("RAW")])
    for key, value in (("rows_sha256", "0" * 64), ("mask_sha256", "0" * 64), ("input_sha256", "0" * 64),
                       ("population_n", 21)):
        with pytest.raises(core.Refusal, match="PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH"):
            store.submit_terminals(PLAN, [terminal("TRAINED_ENCODER", **{key: value})])
    with pytest.raises(core.Refusal, match="PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH"):
        store.submit_terminals(PLAN, [terminal("TRAINED_ENCODER", naive=0.25)])
    with pytest.raises(core.Refusal, match="in the batch"):
        store.submit_terminals(PLAN, [terminal("RANDOM_ENCODER"), terminal("TRAINED_ENCODER", rows_sha256="0" * 64)])
    assert store.readback_summary(PLAN)["count"] == 1
    # a different feature or fold is another triple and is free to differ
    store.submit_terminals(PLAN, [terminal("RAW", feature="px.rv10", rows_sha256="0" * 64)])
    store.submit_terminals(PLAN, [terminal("RAW", fold="inner_2022", mask_sha256="0" * 64)])
    assert store.readback_summary(PLAN)["count"] == 3


def test_fs4_09_reconcile_reports_a_stored_disagreement_and_partial_triples(store):
    store.submit_terminals(PLAN, [terminal("RAW"), terminal("RANDOM_ENCODER"), terminal("TRAINED_ENCODER")])
    report = store.reconcile(PLAN, {"total": 3, "by_population": {"EURUSD": 3}})
    assert report["complete"] and report["triples"] == {"complete": 1, "partial": 0, "inconsistent": 0,
                                                        "inconsistent_examples": []}
    # a disagreement that bypassed the submission path (direct SQL) is still seen by reconcile
    rec = core.prepare_terminal(terminal("RAW", feature="px.rv10"))
    peer = dict(core.prepare_terminal(terminal("RANDOM_ENCODER", feature="px.rv10", rows_sha256="0" * 64)))
    cols = ", ".join(core.COLUMNS)
    marks = ", ".join("?" * len(core.COLUMNS))
    for r in (rec, peer):
        store.conn.execute(f"INSERT INTO {core.TABLE} ({cols}) VALUES ({marks})", [r[c] for c in core.COLUMNS])
    report = store.reconcile(PLAN, {"total": 5, "by_population": {"EURUSD": 5}})
    assert not report["complete"]
    assert report["triples"]["inconsistent"] == 1 and report["triples"]["partial"] == 1
    assert report["triples"]["inconsistent_examples"][0]["feature_id"] == "px.rv10"
    assert report["receipts"]["tasks_without_receipt"] == 2


# ----------------------------------------------------------------------------- FS4-13
def test_fs4_13_every_metric_is_read_back_unchanged_and_digests_agree(store):
    docs = [terminal("RAW"), terminal("RANDOM_ENCODER", mae=0.15), terminal("TRAINED_ENCODER", mae=0.05)]
    receipt = store.submit_terminals(PLAN, docs, host_role="coordinator")
    assert receipt["schema"] == core.SCHEMA_RECEIPT and receipt["task_ids"] == sorted(d["task"]["task_id"] for d in docs)
    for doc in docs:
        back = store.read_terminals(PLAN, task_id=doc["task"]["task_id"])
        assert len(back) == 1
        assert back[0]["result"] == doc["result"]
        assert back[0]["task"] == doc["task"]
        assert back[0]["terminal_sha256"] == hashlib.sha256(core.canonical_bytes(doc["result"])).hexdigest()
        assert back[0]["host_role"] == "coordinator" and back[0]["owner"] == "worker_a-gpu1"
    rows = store.query(f"SELECT arm, mae, naive_mae, mse, population_n, chosen_epoch, updates, cpu_s, wall_s,"
                       f" peak_ram_bytes, peak_vram_bytes, finished_at FROM {core.TABLE} ORDER BY arm")
    assert [r["mae"] for r in rows] == [0.15, 0.1, 0.05]
    assert all(r["naive_mae"] == 0.2 and r["mse"] == 0.01 and r["population_n"] == 20 for r in rows)
    assert all(r["chosen_epoch"] == 3 and r["updates"] == 40 and r["cpu_s"] == 1.5 and r["wall_s"] == 2.0 for r in rows)
    assert all(r["peak_ram_bytes"] == 1000 and r["peak_vram_bytes"] == 0 and r["finished_at"] == 100.0 for r in rows)
    summary = store.readback_summary(PLAN)
    assert summary["count"] == 3 and summary["terminals_sha256"] == python_terminals_digest(docs) == receipt["terminals_sha256"]
    report = store.reconcile(PLAN, {"total": 3, "by_population": {"EURUSD": 3}}, receipts=[receipt])
    assert report["complete"] and report["receipts_verified"] == 1
    assert report["stored"]["terminals_sha256"] == receipt["terminals_sha256"]
    assert report["stored"]["by_population_arm"] == {"EURUSD": {"RAW": 1, "RANDOM_ENCODER": 1, "TRAINED_ENCODER": 1}}
    assert store.verify_receipt(receipt)["accepted"]
    forged = dict(receipt, terminals_sha256="0" * 64)
    with pytest.raises(core.Refusal):
        store.verify_receipt(forged)
    readback = wh.readback_report(store.query, PLAN)
    assert readback["total"] == 3 and readback["by_arm"] == {"RAW": 1, "RANDOM_ENCODER": 1, "TRAINED_ENCODER": 1}
    assert readback["by_host_role"] == {"coordinator": 3}
    assert readback["triples"] == {"total": 1, "complete": 1, "inconsistent": 0}
    assert wh.compare_readback(readback, readback)["agree"]


def test_fs4_13_counts_against_the_controllers_expectation(store):
    store.submit_terminals(PLAN, [terminal("RAW"), terminal("RAW", population="ETH", feature="eth.x")])
    report = store.reconcile(PLAN, {"total": 6495, "by_population": {"EURUSD": 5325, "ETH": 1170}})
    assert report["count_matches_expected"] is False and not report["complete"]
    assert report["stored"]["by_population"] == {"ETH": 1, "EURUSD": 1}
    assert store.reconcile(PLAN)["count_matches_expected"] is None


def test_fs4_13_column_order_does_not_change_identity_or_digest():
    a = terminal("RAW")
    b = {"result": dict(reversed(list(a["result"].items()))), "task": dict(reversed(list(a["task"].items()))),
         "plan_sha256": PLAN, "owner": a["owner"], "attempt": 1, "started_at": 90.0, "finished_at": 100.0}
    ra, rb = core.prepare_terminal(a), core.prepare_terminal(b)
    assert ra["task_id"] == rb["task_id"] and ra["terminal_sha256"] == rb["terminal_sha256"]


def test_fs4_13_read_document_pages_by_task_id(store):
    docs = [terminal(arm) for arm in core.ARMS] + [terminal("RAW", feature="px.rv10")]
    store.submit_terminals(PLAN, docs)
    page = core.read_document(store.conn, {"plan_sha256": PLAN, "limit": 3})
    assert page["count"] == 3 and page["next_after"] == page["terminals"][-1]["task"]["task_id"]
    rest = core.read_document(store.conn, {"plan_sha256": PLAN, "after": page["next_after"]})
    assert rest["count"] == 1 and rest["next_after"] is None
    ids = [t["task"]["task_id"] for t in page["terminals"] + rest["terminals"]]
    assert ids == sorted(ids) and len(set(ids)) == 4
    with pytest.raises(core.Refusal, match="limit"):
        core.read_document(store.conn, {"plan_sha256": PLAN, "limit": 0})
    with pytest.raises(core.Refusal, match="arm"):
        core.read_document(store.conn, {"plan_sha256": PLAN, "arm": "FANCY"})


def test_service_transport_needs_a_token_from_the_environment(monkeypatch):
    monkeypatch.delenv("WAREHOUSE_TOKEN", raising=False)
    with pytest.raises(core.Refusal, match="WAREHOUSE_TOKEN"):
        wh.open_warehouse("http://127.0.0.1:1")


def test_the_packaged_duckdb_backend_serves_the_same_semantics(tmp_path):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("duckdb_engine")
    from predictor_duckdb_store.provider import PredictorDuckdbStore
    from predictor_olap_store.provider import PredictorOlapStore
    for capability in ("write_fs4_terminals", "read_fs4_terminals", "reconcile_fs4"):
        assert capability in PredictorDuckdbStore().capabilities()
        assert capability in PredictorOlapStore().capabilities()
    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(tmp_path / "backend.duckdb"), schema="main", min_free_bytes=0)
    docs = [terminal("RAW"), terminal("RANDOM_ENCODER")]
    receipt = store.write_fs4_terminals({"plan_sha256": PLAN, "terminals": docs, "host_role": "coordinator"})
    assert receipt["inserted"] == 2 and receipt["terminals_sha256"] == python_terminals_digest(docs)
    again = store.write_fs4_terminals({"plan_sha256": PLAN, "terminals": docs})
    assert again["inserted"] == 0 and again["duplicates_ignored"] == 2
    with pytest.raises(ValueError, match="PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH"):
        store.write_fs4_terminals({"plan_sha256": PLAN, "terminals": [terminal("TRAINED_ENCODER", rows_sha256="0" * 64)]})
    with pytest.raises(ValueError, match="finite"):
        store.write_fs4_terminals({"plan_sha256": PLAN, "terminals": [terminal("TRAINED_ENCODER", mae=float("nan"))]})
    page = store.read_fs4_terminals({"plan_sha256": PLAN, "limit": 1})
    assert page["count"] == 1 and page["next_after"] == page["terminals"][0]["task"]["task_id"]
    rest = store.read_fs4_terminals({"plan_sha256": PLAN, "after": page["next_after"]})
    assert rest["next_after"] is None
    assert core.terminals_digest(t["terminal_sha256"] for t in page["terminals"] + rest["terminals"]) == receipt["terminals_sha256"]
    recon = store.reconcile_fs4({"plan_sha256": PLAN, "expected": {"total": 2, "by_population": {"EURUSD": 2}},
                                 "receipts": [receipt]})
    assert recon["stored"]["total"] == 2 and recon["stored"]["terminals_sha256"] == receipt["terminals_sha256"]
    assert recon["receipts_verified"] == 1 and recon["triples"]["partial"] == 1 and not recon["complete"]
    store.engine().dispose()
    local = wh.Warehouse(tmp_path / "backend.duckdb", read_only=True)
    assert local.readback_summary(PLAN)["terminals_sha256"] == receipt["terminals_sha256"]
    local.close()


def test_cli_render_migrate_dry_run_and_reconcile(tmp_path, capsys, monkeypatch):
    target = tmp_path / "cli.duckdb"
    assert wh.main(["migrate", "--duckdb", str(target), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] and plan["destructive_statements"] == 0 and sorted(plan["would_create"]) == sorted(core.ALL_RELATIONS)
    assert not target.exists() or wh.Warehouse(target, read_only=True).query(
        "SELECT count(*) AS n FROM information_schema.tables WHERE table_name = 'feature_extractibility_v1'")[0]["n"] == 0
    assert wh.main(["migrate", "--duckdb", str(target)]) == 0
    capsys.readouterr()
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"total": 1, "by_population": {"EURUSD": 1}}))
    w = wh.Warehouse(target)
    w.submit_terminals(PLAN, [terminal("RAW")])
    w.close()
    assert wh.main(["reconcile", "--warehouse", str(target), "--plan-sha256", PLAN, "--expected", str(expected)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["count_matches_expected"] and report["triples"]["partial"] == 1
    assert wh.main(["readback", "--warehouse", str(target), "--plan-sha256", PLAN, "--out", str(tmp_path / "rb.json")]) == 0
    assert json.loads((tmp_path / "rb.json").read_text())["total"] == 1
