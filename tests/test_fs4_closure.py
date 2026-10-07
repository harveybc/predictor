"""Acceptance tests for tools/fs4_closure.py: a throwaway controller queue (tools/fs4_campaign.py)
and a throwaway DuckDB warehouse; nothing contacts a service or a live store.

  * FS4-07: a second pass adopts the receipts and submits nothing again; a leased task (live or
    expired) never becomes a stored result; a lost receipt file is re-verified by replay, not by
    a second insert;
  * FS4-08: STATUS.json is derived from the task store (counts, active workers, last heartbeat,
    rate, ETA or null with the reason) and the `status` command never opens the warehouse;
  * FS4-09: an arm triple that disagrees in the task store blocks the closure and the warehouse
    quarantines the disagreeing terminal;
  * FS4-13: every COMPLETE terminal is written, read back and receipted; the closure is written
    only when every admitted task is COMPLETE or typed-refused, no result is non-finite, every
    triple agrees, every receipt exists and the warehouse reconciliation equals the task store;
    a technical failure blocks it; the closure is idempotent and digest-bound.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "olap" / "store" / "src", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for name in [n for n in sys.modules if n == "predictor_olap_store" or n.startswith("predictor_olap_store.")]:
    del sys.modules[name]

import fs4_campaign as campaign  # noqa: E402
import fs4_closure as closure  # noqa: E402
import fs4_warehouse as wh  # noqa: E402

core = closure.core
HEX = {k: c * 64 for k, c in zip(core.RESULT_DIGESTS, "abcde")}


def candidate_file(path: Path, population="EURUSD", members=("a", "b", "c")):
    path.write_text(json.dumps({
        "identity": "train-v1", "population_id": population, "final_selection": False,
        "candidates": [{"target_id": "short", "method": "MRMR", "k": len(members), "members": list(members),
                        "subset_sha256": f"subset-{population}"}]}))


def make_queue(tmp_path: Path, folds=("f1",)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "queue_v2.sqlite"
    eur = tmp_path / "eurusd.json"
    eth = tmp_path / "eth.json"
    candidate_file(eur)
    candidate_file(eth, "ETH", ("x",))
    campaign.initialize(db, [eur, eth], list(folds))
    return db


def result_for(task: dict, **over) -> dict:
    result = {"task_id": task["task_id"], "status": "COMPLETE", "seed": 0, **HEX, "population_n": 20,
              "metrics": {"mae": 0.1, "naive_mae": 0.2}, "cost": {"cpu_s": 1.0, "wall_s": 1.5},
              "training": {"chosen_epoch": 2, "updates": 10}}
    result.update(over)
    return result


def complete_all(db: Path, *, now: float = 1000.0, skip=lambda task: False, fail=lambda task: None) -> list[dict]:
    done = []
    clock = now
    while True:
        task = campaign.claim(db, "worker_a-slot1", now=clock)
        if task is None:
            return done
        clock += 1
        if skip(task):
            continue  # stays LEASED (a live lease), never completed in this test
        reason = fail(task)
        if reason:
            campaign.fail(db, "worker_a-slot1", task["task_id"], reason, now=clock)
            continue
        campaign.complete(db, "worker_a-slot1", result_for(task), now=clock + 10)
        done.append(task)
        clock += 10


def args_for(db: Path, tmp_path: Path, **over) -> SimpleNamespace:
    base = {"db": db, "warehouse": str(tmp_path / "wh.duckdb"), "state_root": tmp_path / "state",
            "host_role": "coordinator", "batch": 4, "rate_window_seconds": 7200.0,
            "refusal_pattern": closure.DEFAULT_REFUSAL_PATTERN}
    base.update(over)
    return SimpleNamespace(**base)


def rewrite_result(db: Path, task_id: str, mutate) -> None:
    """Bypass the controller (which refuses this) to plant a bad stored result."""
    con = sqlite3.connect(db)
    result = json.loads(con.execute("SELECT result FROM tasks WHERE task_id = ?", (task_id,)).fetchone()[0])
    mutate(result)
    con.execute("UPDATE tasks SET result = ? WHERE task_id = ?",
                (json.dumps(result, sort_keys=True, separators=(",", ":")), task_id))
    con.commit()
    con.close()


# ----------------------------------------------------------------------------- FS4-08
def test_fs4_08_status_comes_from_the_task_store_and_never_from_the_warehouse(tmp_path, monkeypatch):
    db = make_queue(tmp_path)
    args = args_for(db, tmp_path)
    monkeypatch.setattr(wh, "open_warehouse", lambda *a, **k: (_ for _ in ()).throw(AssertionError("warehouse opened")))
    status = closure.tick(args, submit=False, close=False, now=2000.0)
    assert status["schema"] == closure.SCHEMA_STATUS and status["source"] == "task_store"
    assert status["state"] == "PENDING_DISPATCH"
    assert status["expected"] == {"total": 12, "by_population": {"ETH": 3, "EURUSD": 9}}
    assert status["complete"] == 0 and status["pending"] == 12 and status["active"] == 0
    assert status["failed"] == {"total": 0, "technical": 0, "typed_refused": 0}
    assert status["eta_seconds"] is None and "no completed task" in status["eta_reason"]
    assert status["last_heartbeat"] is None and status["rate_tasks_per_hour"] is None
    assert json.loads((tmp_path / "state" / "STATUS.json").read_text()) == status
    # expected counts refuse a task store whose plan disagrees with its rows
    con = sqlite3.connect(db)
    con.execute("DELETE FROM tasks WHERE task_id = (SELECT task_id FROM tasks LIMIT 1)")
    con.commit()
    con.close()
    with pytest.raises(core.Refusal, match="plan declares"):
        closure.tick(args, submit=False, close=False, now=2000.0)


def test_fs4_08_active_workers_heartbeat_rate_and_eta_are_read_from_the_rows(tmp_path):
    db = make_queue(tmp_path)
    live = campaign.claim(db, "worker_b-gpu", now=5000.0)  # a live lease at now=5100
    campaign.heartbeat(db, "worker_b-gpu", live["task_id"], now=5050.0)
    done = []
    for _ in range(3):
        task = campaign.claim(db, "worker_a-slot1", now=5000.0)
        campaign.complete(db, "worker_a-slot1", result_for(task), now=5060.0)
        done.append(task)
    status = closure.tick(args_for(db, tmp_path), submit=False, close=False, now=5100.0)
    assert status["state"] == "RUNNING"
    assert status["complete"] == 3 and status["active"] == 1 and status["pending"] == 8
    assert status["workers"] == ["worker_b-gpu"]
    assert status["last_heartbeat"] == closure.iso(5050.0)
    assert status["rate_tasks_per_hour"] == pytest.approx(3 / 2.0)
    assert status["eta_seconds"] == round(60.0 * 9 / 1)  # median 60 s x 9 remaining / 1 active worker
    assert status["by_population"]["EURUSD"]["expected"] == 9
    # the lease expires: the task is pending again, no worker is active, the ETA says why
    later = closure.tick(args_for(db, tmp_path), submit=False, close=False, now=5050.0 + campaign.LEASE_SECONDS + 1)
    assert later["active"] == 0 and later["pending"] == 9 and later["eta_seconds"] is None
    assert "no active worker" in later["eta_reason"]


# ----------------------------------------------------------------------------- FS4-07 / FS4-13
def test_fs4_13_terminals_are_submitted_read_back_receipted_and_the_closure_is_written(tmp_path):
    db = make_queue(tmp_path)
    done = complete_all(db)
    assert len(done) == 12
    args = args_for(db, tmp_path)
    status = closure.tick(args, now=9000.0)
    assert status["state"] == "EXTRACTIBILITY_COMPLETE" and status["closure"]["closed"]
    assert status["warehouse"]["verified_receipts"] == 12 and status["warehouse"]["pending_submit"] == 0
    assert status["warehouse"]["last_pass"] == {"submitted": 12, "verified": 12, "quarantined": 0,
                                                "already_verified": 0, "errors": []}
    assert status["warehouse"]["target"] == "wh.duckdb"
    receipts = sorted((tmp_path / "state" / "receipts").glob("*.json"))
    assert len(receipts) == 12
    one = json.loads(receipts[0].read_text())
    assert one["readback_verified"] and one["receipt"]["schema"] == core.SCHEMA_RECEIPT
    closed = json.loads((tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").read_text())
    assert closed["state"] == "EXTRACTIBILITY_COMPLETE" and closed["schema"] == closure.SCHEMA_CLOSURE
    assert closed["admitted"] == {"total": 12, "by_population": {"ETH": 3, "EURUSD": 9}}
    assert closed["complete"] == 12 and closed["typed_refused"] == 0
    assert closed["reconciliation"]["complete"] and closed["reconciliation"]["triples"]["complete"] == 4
    assert closed["receipts"]["count"] == 12 and closed["final_selection"] is False
    body = {k: v for k, v in closed.items() if k not in ("generated_at", "closure_sha256")}
    assert closed["closure_sha256"] == closure.digest(body)
    assert closed["terminals_sha256"] == closed["reconciliation"]["stored"]["terminals_sha256"]
    # every metric is in the cube and equals the controller's stored result
    store = wh.Warehouse(tmp_path / "wh.duckdb", read_only=True)
    assert store.readback_summary(closed["plan_sha256"])["count"] == 12
    for task in done:
        back = store.read_terminals(closed["plan_sha256"], task_id=task["task_id"])[0]
        assert back["result"]["metrics"] == {"mae": 0.1, "naive_mae": 0.2}
        assert back["task"]["arm"] == task["arm"] and back["host_role"] == "coordinator"
    store.close()


def test_fs4_07_a_second_pass_adopts_receipts_and_a_lost_receipt_is_replayed_not_reinserted(tmp_path):
    db = make_queue(tmp_path)
    complete_all(db)
    args = args_for(db, tmp_path)
    first = closure.tick(args, now=9000.0)
    assert first["state"] == "EXTRACTIBILITY_COMPLETE"
    closed_bytes = (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").read_bytes()
    second = closure.tick(args, now=9100.0)
    assert second["state"] == "EXTRACTIBILITY_COMPLETE"
    assert "last_pass" not in second["warehouse"]  # closed: nothing is opened or resubmitted
    assert (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").read_bytes() == closed_bytes
    # before closure: a lost receipt file is re-verified by replay (duplicates_ignored), never a second row
    (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").unlink()
    lost = sorted((tmp_path / "state" / "receipts").glob("*.json"))[0]
    lost.unlink()
    third = closure.tick(args, now=9200.0)
    assert third["warehouse"]["last_pass"]["submitted"] == 1 and third["warehouse"]["last_pass"]["already_verified"] == 11
    assert third["warehouse"]["verified_receipts"] == 12
    store = wh.Warehouse(tmp_path / "wh.duckdb", read_only=True)
    assert store.readback_summary(first["plan_sha256"])["count"] == 12
    assert store.query("SELECT sum(inserted) AS n FROM fs4_load_receipt")[0]["n"] == 12
    store.close()


def test_fs4_07_a_leased_task_never_becomes_a_stored_result_and_blocks_closure(tmp_path):
    db = make_queue(tmp_path)
    complete_all(db, skip=lambda task: task["feature_id"] == "a" and task["arm"] == "RAW")
    status = closure.tick(args_for(db, tmp_path), now=2000.0)
    assert status["state"] == "RUNNING" and status["complete"] == 11 and status["active"] == 1
    assert status["warehouse"]["verified_receipts"] == 11
    assert not (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").exists()
    assert any("pending or running" in r for r in status["closure"]["reasons"])
    store = wh.Warehouse(tmp_path / "wh.duckdb", read_only=True)
    assert store.readback_summary(status["plan_sha256"])["count"] == 11
    store.close()


def test_fs4_13_a_technical_failure_blocks_and_a_typed_refusal_is_a_disposition(tmp_path):
    db = make_queue(tmp_path)
    complete_all(db, fail=lambda t: ("runner rc=1: NO_TRAIN_OBSERVATIONS for x/f1"
                                     if t["feature_id"] == "x" else None))
    args = args_for(db, tmp_path)
    status = closure.tick(args, now=3000.0)
    assert status["failed"] == {"total": 3, "technical": 0, "typed_refused": 3}
    assert status["state"] == "EXTRACTIBILITY_COMPLETE"
    closed = json.loads((tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").read_text())
    assert closed["complete"] == 9 and closed["typed_refused"] == 3
    assert {r["feature_id"] for r in closed["typed_refusals"]} == {"x"}
    assert closed["complete_by_population"] == {"EURUSD": 9}
    assert closed["reconciliation"]["expected"] == {"total": 9, "by_population": {"EURUSD": 9}}
    # a technical failure (lease exhausted / runner crash) is not a disposition
    db2 = make_queue(tmp_path / "two")
    complete_all(db2, fail=lambda t: "runner rc=137: Killed" if t["feature_id"] == "x" else None)
    status2 = closure.tick(args_for(db2, tmp_path / "two"), now=3000.0)
    assert status2["failed"] == {"total": 3, "technical": 3, "typed_refused": 0}
    assert status2["state"] == "BLOCKED" and not (tmp_path / "two" / "state" / "EXTRACTIBILITY_COMPLETE.json").exists()
    assert any("technical failures" in r for r in status2["closure"]["reasons"])
    assert status2["validation"]["technical_failures"][0]["reason"].startswith("runner rc=137")


def test_fs4_13_a_non_finite_stored_result_blocks_closure_and_is_quarantined(tmp_path):
    db = make_queue(tmp_path)
    done = complete_all(db)
    rewrite_result(db, done[0]["task_id"], lambda r: r["metrics"].__setitem__("mae", "not-a-number"))
    status = closure.tick(args_for(db, tmp_path), now=4000.0)
    assert status["state"] == "BLOCKED"
    assert status["validation"]["invalid_complete"][0]["task_id"] == done[0]["task_id"]
    assert status["warehouse"]["quarantined"] == 1 and status["warehouse"]["verified_receipts"] == 11
    quarantine = json.loads((tmp_path / "state" / "quarantine" / f"{done[0]['task_id']}.json").read_text())
    assert quarantine["reason"].startswith("INVALID_TERMINAL")
    assert not (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").exists()


# ----------------------------------------------------------------------------- FS4-09
def test_fs4_09_an_arm_triple_that_disagrees_blocks_closure_and_is_refused_by_the_warehouse(tmp_path):
    db = make_queue(tmp_path)
    done = complete_all(db)
    victim = next(t for t in done if t["feature_id"] == "b" and t["arm"] == "TRAINED_ENCODER")
    rewrite_result(db, victim["task_id"], lambda r: r.__setitem__("rows_sha256", "0" * 64))
    status = closure.tick(args_for(db, tmp_path), now=4000.0)
    assert status["state"] == "BLOCKED"
    bad = status["validation"]["inconsistent_triples"]
    assert bad == [{"population_id": "EURUSD", "identity": "train-v1", "feature_id": "b", "fold_id": "f1",
                    "arms": ["RANDOM_ENCODER", "RAW", "TRAINED_ENCODER"]}]
    assert any("arm triples disagree" in r for r in status["closure"]["reasons"])
    # the warehouse refused exactly the disagreeing terminal (the batch was isolated one by one)
    quarantined = {p.stem for p in (tmp_path / "state" / "quarantine").glob("*.json")}
    assert quarantined == {victim["task_id"]}
    reason = json.loads((tmp_path / "state" / "quarantine" / f"{victim['task_id']}.json").read_text())["reason"]
    assert "PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH" in reason
    store = wh.Warehouse(tmp_path / "wh.duckdb", read_only=True)
    assert store.readback_summary(status["plan_sha256"])["count"] == 11
    store.close()


def test_a_readback_mismatch_is_quarantined_not_receipted(tmp_path, monkeypatch):
    db = make_queue(tmp_path)
    complete_all(db)
    real_read = wh.Warehouse.read_terminals

    def lying_read(self, plan_sha256, **kw):
        out = real_read(self, plan_sha256, **kw)
        for doc in out:
            doc["terminal_sha256"] = "0" * 64
        return out

    monkeypatch.setattr(wh.Warehouse, "read_terminals", lying_read)
    status = closure.tick(args_for(db, tmp_path), now=4000.0)
    assert status["warehouse"]["quarantined"] == 12 and status["warehouse"]["verified_receipts"] == 0
    assert status["state"] == "BLOCKED"
    assert not (tmp_path / "state" / "EXTRACTIBILITY_COMPLETE.json").exists()


def test_cli_expected_and_status_print_json(tmp_path, capsys):
    db = make_queue(tmp_path)
    assert closure.main(["--db", str(db), "--warehouse", str(tmp_path / "w.duckdb"),
                         "--state-root", str(tmp_path / "s"), "expected"]) == 0
    assert json.loads(capsys.readouterr().out) == {"total": 12, "by_population": {"ETH": 3, "EURUSD": 9}}
    assert closure.main(["--db", str(db), "--warehouse", str(tmp_path / "w.duckdb"),
                         "--state-root", str(tmp_path / "s"), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "PENDING_DISPATCH"
    assert not (tmp_path / "w.duckdb").exists()
