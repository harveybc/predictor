#!/usr/bin/env python3
"""Durable set-level weekly controller for phase 4, the equivalent of ``tools/fs4_campaign.py``.

Plan sealed at ``init`` (frontier sets x VALIDATION weeks x input modes) only when the
extractibility closure exists; exclusive ``claim``; ``heartbeat``; ``complete`` verifies identity,
seed, finite paired metrics, cost and that every input-mode arm of one set/week scored IDENTICAL
rows with the same naive; ``status`` writes a generated ``STATUS.json`` from the task store alone;
``close`` aggregates on the full denominator, chooses winners by the predeclared rule, writes the
terminal rows to the warehouse and reads them back; ``freeze`` seals selector, model,
preprocessing and tie rule; ``open-test`` enumerates the TEST traversal exactly once.

The coordinator owns the SQLite file; remote workers call this over SSH and never copy it.
Scientific work is done by ``tools/fs4_weekly_wrapper.py run-task`` through ``tools/fs4_weekly_worker.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs4_frontier as F  # noqa: E402
from tools import fs4_weekly_wrapper as WW  # noqa: E402
from tools.business_objective_firewall import BusinessObjectiveFirewall  # noqa: E402
from tools.fs4_candidates import SCHEMA as CONSOLIDATED_SCHEMA  # noqa: E402
from tools.fs4_candidates import canonical, digest, write_atomic  # noqa: E402

SCHEMA = "fs4.weekly_campaign.v1"
STATUS_SCHEMA = "fs4.weekly_status.v1"
CLOSURE_SCHEMA = "fs4.weekly_selection_complete.v1"
FREEZE_SCHEMA = "fs4.test_freeze.v1"
WAREHOUSE_TABLE = "feature_weekly_selection_v1"
LEASE_SECONDS = 7200
MAX_ATTEMPTS = 3
DEFAULT_BAR_HOURS = {"EURUSD": 1, "ETH": 4}
MODE_ORDER = {m: i for i, m in enumerate(("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER"))}


class Refusal(ValueError):
    """A mismatch that must not become accepted evidence."""


# ----------------------------------------------------------------------------- store
def _open(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS campaign(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
            task_id TEXT PRIMARY KEY, split TEXT NOT NULL, population_id TEXT NOT NULL, set_id TEXT NOT NULL,
            input_mode TEXT NOT NULL, week_start TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'PENDING', owner TEXT, lease_until REAL, started_at REAL, finished_at REAL,
            result TEXT, attempt INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, split, input_mode, ordinal);
    """)
    return con


def _get(con, key):
    row = con.execute("SELECT value FROM campaign WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _set(con, key, value):
    con.execute("INSERT OR REPLACE INTO campaign VALUES(?, ?)", (key, canonical(value)))


def _plan(con) -> dict:
    plan = _get(con, "plan")
    if not plan:
        raise Refusal("NO_PLAN")
    return plan


def _out_dir(con) -> Path:
    return Path(_get(con, "out_dir"))


def _firewall_path(con) -> Path:
    return _out_dir(con) / "FIREWALL.json"


# ----------------------------------------------------------------------------- init
def _insert_tasks(con, tasks):
    con.executemany("""INSERT OR IGNORE INTO tasks(task_id, split, population_id, set_id, input_mode, week_start, ordinal, payload)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    [(t["task_id"], t["split"], t["population_id"], t["set_id"], t["input_mode"], t["week"]["start"],
                      t["week"]["ordinal"], canonical(t)) for t in tasks])


def initialize(db, consolidated_paths, seal_paths, extractibility_path, *, out_dir, validation_year: int,
               input_modes=("RAW",), bar_hours: dict | None = None) -> dict:
    extractibility_path = Path(extractibility_path)
    if not extractibility_path.is_file():
        raise Refusal(f"MISSING_EXTRACTIBILITY: {extractibility_path} (EXTRACTIBILITY_COMPLETE.json must exist before the weekly campaign)")
    ext = json.loads(extractibility_path.read_text())
    if ext.get("state") != "EXTRACTIBILITY_COMPLETE" or not ext.get("closure_sha256"):
        raise Refusal("EXTRACTIBILITY_NOT_COMPLETE")
    consolidated = []
    for p in consolidated_paths:
        c = json.loads(Path(p).read_text())
        if c.get("schema") != CONSOLIDATED_SCHEMA:
            raise Refusal(f"CONSOLIDATED_SCHEMA_INVALID: {p}")
        consolidated.append(c)
    seals = [F.load_seal(p) for p in seal_paths]
    for seal in seals:
        if seal["inputs"]["extractibility"]["closure_sha256"] != ext["closure_sha256"]:
            raise Refusal(f"FRONTIER_SEAL_WITHOUT_EXTRACTIBILITY: the {seal['population_id']} seal was sealed without (or with another) "
                          "extractibility closure; re-seal with --extractibility before initialising the weekly campaign")
    bar_hours = dict(DEFAULT_BAR_HOURS) | dict(bar_hours or {})
    try:
        plan = WW.build_plan(consolidated, seals, validation_year=validation_year, input_modes=tuple(input_modes),
                             bar_hours={c["population_id"]: bar_hours.get(c["population_id"], 1) for c in consolidated},
                             extractibility_sha256=ext["closure_sha256"])
    except WW.Refusal as exc:
        raise Refusal(str(exc)) from exc
    tasks = WW.enumerate_tasks(plan, "validation")
    out_dir = Path(out_dir)
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        prior = _get(con, "plan_sha256")
        if prior and prior != plan["plan_sha256"]:
            raise Refusal("PLAN_CHANGED_USE_NEW_QUEUE_PATH")
        if not prior:
            _set(con, "plan_sha256", plan["plan_sha256"])
            _set(con, "plan", plan)
            _set(con, "out_dir", str(out_dir))
            _set(con, "extractibility_closure_sha256", ext["closure_sha256"])
        _insert_tasks(con, tasks)
    write_atomic(out_dir / "WEEKLY_PLAN.json", plan)
    fw_path = out_dir / "FIREWALL.json"
    if not fw_path.is_file():
        firewall = BusinessObjectiveFirewall.create([f"test:{w['start']}" for w in plan["test_weeks"]])
        write_atomic(fw_path, json.loads(firewall.to_json()))
    return {"plan_sha256": plan["plan_sha256"], "tasks": len(tasks), "sets": len(plan["sets"]), "weeks": len(plan["weeks"]),
            "input_modes": plan["input_modes"], "denominator": plan["denominator"], "out_dir": str(out_dir)}


# ----------------------------------------------------------------------------- lease
def claim(db, owner, *, now=None, task_id=None, input_mode=None, split=None):
    if not owner or any(ch.isspace() for ch in owner):
        raise Refusal("INVALID_OWNER")
    now = time.time() if now is None else now
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("""UPDATE tasks SET state='FAILED', finished_at=?, result=? WHERE state='LEASED' AND lease_until<? AND attempt>=?""",
                    (now, canonical({"reason": "LEASE_EXPIRED_MAX_ATTEMPTS"}), now, MAX_ATTEMPTS))
        row = con.execute("""SELECT * FROM tasks WHERE (? IS NULL OR task_id=?) AND (? IS NULL OR input_mode=?) AND (? IS NULL OR split=?)
            AND (state='PENDING' OR (state='LEASED' AND lease_until<? AND attempt<?))
            ORDER BY CASE split WHEN 'validation' THEN 0 ELSE 1 END,
                     CASE input_mode WHEN 'RAW' THEN 0 WHEN 'RANDOM_ENCODER' THEN 1 ELSE 2 END, ordinal, set_id LIMIT 1""",
                          (task_id, task_id, input_mode, input_mode, split, split, now, MAX_ATTEMPTS)).fetchone()
        if row is None:
            return None
        con.execute("UPDATE tasks SET state='LEASED', owner=?, lease_until=?, started_at=?, attempt=attempt+1 WHERE task_id=?",
                    (owner, now + LEASE_SECONDS, now, row["task_id"]))
        return {**json.loads(row["payload"]), "lease_until": now + LEASE_SECONDS, "attempt": row["attempt"] + 1}


def heartbeat(db, owner, task_id, now=None):
    now = time.time() if now is None else now
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT state, owner, lease_until FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner or row["lease_until"] < now:
            raise Refusal("NO_LIVE_LEASE_FOR_HEARTBEAT")
        con.execute("UPDATE tasks SET lease_until=? WHERE task_id=?", (now + LEASE_SECONDS, task_id))
    return {"task_id": task_id, "lease_until": now + LEASE_SECONDS}


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_result(result: dict, payload: dict) -> None:
    if result.get("status") != "COMPLETE" or result.get("task_id") != payload["task_id"] or result.get("schema") != WW.RESULT_SCHEMA:
        raise Refusal("RESULT_IDENTITY_OR_STATUS_MISMATCH")
    for key in ("set_id", "input_mode", "population_id", "target_id", "split"):
        if result.get(key) != payload[key]:
            raise Refusal(f"RESULT_IDENTITY_MISMATCH: {key}")
    if result.get("week_start") != payload["week"]["start"]:
        raise Refusal("RESULT_IDENTITY_MISMATCH: week_start")
    if result.get("seed") != payload["seed"]:
        raise Refusal("SEED_MISMATCH")
    if result.get("disposition") not in ("COMPLETED", "FAILED"):
        raise Refusal("DISPOSITION_INVALID")
    if not _hex64(result.get("rows_sha256")):
        raise Refusal("INVALID_ROWS_SHA256")
    if type(result.get("n_scored")) is not int or result["n_scored"] < 0:
        raise Refusal("INVALID_N_SCORED")
    cost = result.get("cost")
    if not isinstance(cost, dict) or any(k not in cost for k in ("fit_seconds", "wall_seconds", "epochs", "peak_rss_bytes")):
        raise Refusal("MISSING_COST")
    if result["disposition"] == "FAILED":
        if not result.get("reason"):
            raise Refusal("FAILED_WEEK_NEEDS_REASON")
        canonical(result)
        return
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or any(k not in metrics for k in ("mae", "mse", "naive_mae", "naive_mse")):
        raise Refusal("MISSING_PAIRED_METRICS")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in metrics.values()):
        raise Refusal("INVALID_METRIC")
    if result["n_scored"] < 1:
        raise Refusal("COMPLETED_WEEK_WITHOUT_SCORED_ROWS")
    for key in ("model_sha256", "input_sha256", "code_sha256", "fit_population_digest"):
        if not _hex64(result.get(key)):
            raise Refusal(f"INVALID_{key.upper()}")
    canonical(result)


def complete(db, owner, result: dict, now=None):
    now = time.time() if now is None else now
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM tasks WHERE task_id=?", (result.get("task_id"),)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner or row["lease_until"] < now:
            raise Refusal("NO_LIVE_LEASE_FOR_RESULT")
        payload = json.loads(row["payload"])
        _validate_result(result, payload)
        peers = con.execute("""SELECT result FROM tasks WHERE state='COMPLETE' AND split=? AND population_id=? AND set_id=? AND week_start=?""",
                            (payload["split"], payload["population_id"], payload["set_id"], payload["week"]["start"])).fetchall()
        for peer in peers:
            prior = json.loads(peer["result"])
            if prior["rows_sha256"] != result["rows_sha256"] or prior.get("n_scored") != result.get("n_scored"):
                raise Refusal("PAIRED_ROWS_OR_NAIVE_MISMATCH: input-mode arms must score identical rows")
            if prior["disposition"] == "COMPLETED" and result["disposition"] == "COMPLETED" and \
                    prior["metrics"]["naive_mae"] != result["metrics"]["naive_mae"]:
                raise Refusal("PAIRED_ROWS_OR_NAIVE_MISMATCH: input-mode arms must share the paired naive")
        con.execute("UPDATE tasks SET state='COMPLETE', finished_at=?, result=? WHERE task_id=?", (now, canonical(result), row["task_id"]))
    return {"task_id": result["task_id"], "accepted": True, "disposition": result["disposition"]}


def fail(db, owner, task_id, reason, now=None):
    now = time.time() if now is None else now
    if not reason or len(reason) > 1000:
        raise Refusal("INVALID_REASON")
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner:
            raise Refusal("NO_LEASE_FOR_FAILURE")
        state = "FAILED" if row["attempt"] >= MAX_ATTEMPTS else "PENDING"
        con.execute("UPDATE tasks SET state=?, finished_at=?, result=?, owner=NULL, lease_until=NULL WHERE task_id=?",
                    (state, now, canonical({"reason": reason, "attempt": row["attempt"]}), task_id))
    return {"task_id": task_id, "failed": True, "terminal": state == "FAILED", "attempt": row["attempt"]}


# ----------------------------------------------------------------------------- status
def status(db, now=None, *, write=False, population=None, input_mode=None, split=None):
    now = time.time() if now is None else now
    with _open(db) as con:
        plan = _plan(con)
        out_dir = _out_dir(con)
        closure, freeze, test_opened = _get(con, "closure"), _get(con, "freeze"), _get(con, "test_opened")
        rows = con.execute("SELECT state, owner, lease_until, started_at, finished_at, population_id, input_mode, split, attempt FROM tasks").fetchall()
    counts = {"pending": 0, "active": 0, "complete": 0, "failed": 0}
    by_population: dict[str, dict] = {}
    workers = set()
    durations = []
    last_heartbeat = None
    for r in rows:
        if (population and r["population_id"] != population) or (input_mode and r["input_mode"] != input_mode) or (split and r["split"] != split):
            continue
        state = r["state"].lower()
        if state == "leased":
            state = "pending" if r["lease_until"] < now else "active"
        counts[state] += 1
        grp = by_population.setdefault(r["population_id"], {}).setdefault(r["input_mode"], {k: 0 for k in counts})
        grp[state] += 1
        if state == "active":
            workers.add(r["owner"])
            hb = r["lease_until"] - LEASE_SECONDS
            last_heartbeat = hb if last_heartbeat is None else max(last_heartbeat, hb)
        if state == "complete" and r["finished_at"] is not None and r["started_at"] is not None:
            durations.append(r["finished_at"] - r["started_at"])
    total = sum(counts.values())
    remaining = counts["pending"] + counts["active"]
    median = statistics.median(durations) if durations else None
    if remaining == 0:
        eta, reason, rate = 0, "all tasks terminal", None
    elif median is None:
        eta, reason, rate = None, "no measured task duration yet", None
    elif not workers:
        eta, reason, rate = None, "no active worker holds a lease", None
    else:
        rate = len(workers) * 3600.0 / max(median, 1e-9)
        eta, reason = int(round(median * remaining / len(workers))), "observed median task duration / active workers"
    if test_opened:
        state = "TEST_OPEN"
    elif freeze:
        state = "FROZEN"
    elif closure:
        state = "CLOSED"
    elif counts["active"] > 0:
        state = "RUNNING"
    elif total and remaining == 0:
        state = "ALL_TERMINAL"
    elif counts["complete"] + counts["failed"] > 0:
        state = "IDLE_WITH_PENDING"
    else:
        state = "PENDING"
    doc = {"schema": STATUS_SCHEMA, "state": state, "expected": total, "total": total, "pending": counts["pending"],
           "active": counts["active"], "running": counts["active"], "complete": counts["complete"], "failed": counts["failed"],
           "last_heartbeat": last_heartbeat, "workers": sorted(workers), "measured_tasks": len(durations),
           "median_task_seconds": median, "rate_tasks_per_hour": rate, "eta_seconds": eta, "eta_reason": reason,
           "by_population": by_population,
           "plan": {"plan_sha256": plan["plan_sha256"], "validation_year": plan["validation_year"], "input_modes": plan["input_modes"],
                    "sets": len(plan["sets"]), "weeks": len(plan["weeks"]), "denominator": plan["denominator"]},
           "closure_sha256": closure.get("closure_sha256") if closure else None,
           "freeze_sha256": freeze.get("freeze_sha256") if freeze else None, "test_opened_utc": test_opened,
           "scope": {"population": population, "input_mode": input_mode, "split": split},
           "generated_from": "task_store", "generated_at_epoch": now, "generated_utc": WW._now(), "final_selection": False}
    if write:
        write_atomic(out_dir / "STATUS.json", doc)
    return doc


def list_tasks(db, *, set_id=None, week_start=None, input_mode=None, split=None, state=None):
    with _open(db) as con:
        rows = con.execute("SELECT task_id, state, payload FROM tasks ORDER BY split, input_mode, ordinal, set_id").fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload"])
        if (set_id is None or p["set_id"] == set_id) and (week_start is None or p["week"]["start"] == week_start) and \
                (input_mode is None or p["input_mode"] == input_mode) and (split is None or p["split"] == split) and \
                (state is None or r["state"] == state):
            out.append({**p, "state": r["state"]})
    return out


# ----------------------------------------------------------------------------- close / freeze / test
def _warehouse_rows(plan: dict, results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        rows.append({"run_id": plan["plan_sha256"], "unit_id": r["set_id"], "row_key": r["task_id"], "task_id": r["task_id"],
                     "population_id": r["population_id"], "identity": r.get("identity"), "set_id": r["set_id"], "target_id": r["target_id"],
                     "input_mode": r["input_mode"], "split": r["split"], "week_start": r["week_start"], "disposition": r["disposition"],
                     "reason": r.get("reason"), "n_scored": r.get("n_scored"), "rows_sha256": r["rows_sha256"],
                     "fit_population_digest": r.get("fit_population_digest"), "n_features": r.get("n_features"),
                     "mae": (r.get("metrics") or {}).get("mae"), "mse": (r.get("metrics") or {}).get("mse"),
                     "naive_mae": (r.get("metrics") or {}).get("naive_mae"), "naive_mse": (r.get("metrics") or {}).get("naive_mse"),
                     "skill_mae": r.get("skill_mae"), "model_sha256": r.get("model_sha256"), "input_sha256": r.get("input_sha256"),
                     "code_sha256": r.get("code_sha256"), "seed": r.get("seed"), "cost": r.get("cost"),
                     "evaluation_mode": WW.BUSINESS_MODE, "update_mode": WW.UPDATE_MODE})
    return rows


def _warehouse_submit_and_readback(plan: dict, results: list[dict], warehouse_path) -> dict:
    from tools import fs_phase23_warehouse_adapter as A

    rows = _warehouse_rows(plan, results)
    wh = A.open_warehouse(Path(warehouse_path))
    try:
        receipt = wh.submit_rows(plan["plan_sha256"], WAREHOUSE_TABLE, rows, host_role="coordinator")
        back = wh.read_run(plan["plan_sha256"], WAREHOUSE_TABLE)
    finally:
        wh.close()
    local_digest = A.rows_digest(rows)
    back_digest = A.rows_digest(back)
    return {"table": WAREHOUSE_TABLE, "path": str(warehouse_path), "submitted": len(rows), "receipt": receipt,
            "readback": {"count": len(back), "rows_sha256": back_digest, "matches_store": back_digest == local_digest and len(back) == len(rows)},
            "store_rows_sha256": local_digest}


def _aggregates(plan: dict, results: list[dict], modes) -> dict:
    """population -> target -> mode -> {set_id: aggregate}."""
    weeks = [w["start"] for w in plan["weeks"]]
    n_features = {s["set_id"]: s["n_features"] for s in plan["sets"]}
    out: dict = {}
    for pop in sorted({s["population_id"] for s in plan["sets"]}):
        for target in sorted({s["target_id"] for s in plan["sets"] if s["population_id"] == pop}):
            for mode in modes:
                subset = [r for r in results if r["population_id"] == pop and r["target_id"] == target and r["input_mode"] == mode]
                if subset:
                    out.setdefault(pop, {}).setdefault(target, {})[mode] = WW.aggregate_results(subset, weeks, sets_n_features=n_features)
    return out


def stage2(db, now=None) -> dict:
    """Compute the stage-2 list mechanically from the stage-1 aggregates (docs/fs4/STAGE2_RULE.md) and enqueue it."""
    with _open(db) as con:
        plan = _plan(con)
        out_dir = _out_dir(con)
        prior = _get(con, "stage2")
        rows = con.execute("SELECT state, result FROM tasks WHERE split='validation' AND input_mode IN (%s)"
                           % ",".join("?" * len(plan["input_modes"])), list(plan["input_modes"])).fetchall()
    if WW.stage2_rule_sha256() != plan["stage2_rule_sha256"]:
        raise Refusal("STAGE2_RULE_CHANGED: docs/fs4/STAGE2_RULE.md differs from the digest sealed in the plan")
    pending = [r for r in rows if r["state"] != "COMPLETE"]
    if pending:
        raise Refusal(f"STAGE1_INCOMPLETE: {len(pending)} stage-1 validation tasks are not terminal")
    results = [json.loads(r["result"]) for r in rows]
    raw = {pop: {t: modes[plan["selection_arm"]] for t, modes in tt.items() if plan["selection_arm"] in modes}
           for pop, tt in _aggregates(plan, [r for r in results if r["input_mode"] in plan["input_modes"]], plan["input_modes"]).items()}
    body = WW.stage2_set_ids(plan, raw)
    if prior:
        if prior["list_sha256"] != body["list_sha256"]:
            raise Refusal("STAGE2_LIST_CHANGED: a different stage-2 list was already enqueued")
        return prior
    tasks = WW.enumerate_tasks(plan, "validation", set_ids=set(body["set_ids"]), modes=plan["stage2_modes"])
    body["tasks"] = len(tasks)
    body["computed_utc"] = WW._now()
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        if _get(con, "stage2"):
            raise Refusal("STAGE2_ALREADY_COMPUTED")
        _insert_tasks(con, tasks)
        _set(con, "stage2", body)
    write_atomic(out_dir / "STAGE2_LIST.json", body)
    return body


def close(db, *, warehouse_path, now=None) -> dict:
    now = time.time() if now is None else now
    with _open(db) as con:
        plan = _plan(con)
        out_dir = _out_dir(con)
        prior = _get(con, "closure")
        stage2_done = _get(con, "stage2")
        rows = con.execute("SELECT task_id, state, payload, result FROM tasks WHERE split='validation'").fetchall()
    if not stage2_done:
        raise Refusal("STAGE2_NOT_COMPUTED: run `stage2` once stage 1 is terminal; close needs the full denominator")
    not_terminal = [r["task_id"] for r in rows if r["state"] != "COMPLETE"]
    if not_terminal:
        raise Refusal(f"DENOMINATOR_INCOMPLETE: {len(not_terminal)} of {len(rows)} validation tasks are not terminal "
                      f"(first {not_terminal[:3]})")
    results = [json.loads(r["result"]) for r in rows]
    weeks = [w["start"] for w in plan["weeks"]]
    sets = {s["set_id"]: s for s in plan["sets"]}
    all_modes = list(plan["input_modes"]) + list(plan["stage2_modes"])
    by = _aggregates(plan, results, all_modes)
    aggregates: dict = {}
    winners: dict = {}
    comparison: dict = {}
    for pop, tt in by.items():
        for target, modes in tt.items():
            aggregates.setdefault(pop, {})[target] = modes
            arm = plan["selection_arm"]          # encoder arms never choose a winner (STAGE2_RULE.md)
            win = WW.choose_winner(modes[arm]) if arm in modes else None
            if win is not None:
                win = win | {"members": sets[win["set_id"]]["members"], "methods": sets[win["set_id"]]["methods"]}
            winners.setdefault(pop, {}).setdefault(target, {})[arm] = win
            cmp_ = WW.encoder_comparison(modes, arm)
            if cmp_:
                comparison.setdefault(pop, {})[target] = cmp_
    warehouse = _warehouse_submit_and_readback(plan, results, warehouse_path)
    if not warehouse["readback"]["matches_store"]:
        raise Refusal("WAREHOUSE_READBACK_MISMATCH")
    body = {"schema": CLOSURE_SCHEMA, "state": "WEEKLY_SELECTION_COMPLETE", "plan_sha256": plan["plan_sha256"],
            "evaluation_mode": WW.BUSINESS_MODE, "update_mode": WW.UPDATE_MODE, "validation_year": plan["validation_year"],
            "denominator": {"tasks": len(rows), "terminal": len(rows), "weeks": len(weeks), "sets": len(sets), "input_modes": all_modes,
                            "frontier": plan["denominator"]},
            "weeks_failed": sum(1 for r in results if r["disposition"] == "FAILED"),
            "aggregate_rule": WW.AGGREGATE_RULE, "tie_rule": WW.TIE_RULE, "selection_arm": plan["selection_arm"],
            "aggregates": aggregates, "winners": winners, "encoder_comparison": comparison,
            "stage2": {"list_sha256": stage2_done["list_sha256"], "sets": len(stage2_done["set_ids"]), "rule_sha256": stage2_done["stage2_rule_sha256"]},
            "test_opened": False, "final_selection": False,
            "warehouse": {k: v for k, v in warehouse.items() if k != "receipt"}}
    body["closure_sha256"] = digest({k: v for k, v in body.items() if k not in ("warehouse",)})
    if prior and prior.get("closure_sha256") == body["closure_sha256"]:
        return {**prior, "warehouse": body["warehouse"]}
    if prior:
        raise Refusal("CLOSURE_CHANGED: a different closure was already recorded")
    body["closed_utc"] = WW._now()
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        _set(con, "closure", body)
    write_atomic(out_dir / "WEEKLY_SELECTION_COMPLETE.json", body)
    status(db, now=now, write=True)
    return body


def freeze(db) -> dict:
    with _open(db) as con:
        plan = _plan(con)
        out_dir = _out_dir(con)
        closure = _get(con, "closure")
        if _get(con, "freeze"):
            raise Refusal("ALREADY_FROZEN: the procedure is sealed; reselection is forbidden")
    if not closure:
        raise Refusal("NOT_CLOSED: freeze requires WEEKLY_SELECTION_COMPLETE")
    body = {"schema": FREEZE_SCHEMA, "plan_sha256": plan["plan_sha256"], "closure_sha256": closure["closure_sha256"],
            "selector": {"frontier_rule_sha256": {p: v["frontier_rule_sha256"] for p, v in plan["populations"].items()},
                         "frontier_seal_sha256": {p: v["frontier_seal_sha256"] for p, v in plan["populations"].items()},
                         "aggregate_rule": WW.AGGREGATE_RULE, "selection_arm": plan["selection_arm"]},
            "model": {"trainer": plan["trainer"], "predictor_spec": plan["predictor_spec"], "predictor_spec_sha256": plan["predictor_spec_sha256"],
                      "budget_sha256": plan["budget_sha256"], "encoder_spec_sha256": plan["encoder_spec_sha256"], "seed": plan["seed"]},
            "preprocessing": {"rule": plan["preprocessing"], "support": plan["support"], "naive": plan["naive"], "scored_rows": plan["scored_rows"]},
            "tie_rule": WW.TIE_RULE, "aggregate_rule": WW.AGGREGATE_RULE,
            "winners": closure["winners"], "update_mode": WW.UPDATE_MODE, "evaluation_mode": WW.BUSINESS_MODE,
            "firewall_phase": "PROCEDURE_SEALED", "test_weeks_sealed": len(plan["test_weeks"])}
    body["freeze_sha256"] = digest(body)
    fw_path = out_dir / "FIREWALL.json"
    firewall = BusinessObjectiveFirewall.from_json(fw_path.read_text())
    firewall = firewall.seal_procedure(body["freeze_sha256"], {"freeze_sha256": body["freeze_sha256"], "plan_sha256": plan["plan_sha256"]},
                                       {"closure_sha256": closure["closure_sha256"]})
    body["frozen_utc"] = WW._now()
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        if _get(con, "freeze"):
            raise Refusal("ALREADY_FROZEN")
        _set(con, "freeze", body)
    write_atomic(fw_path, json.loads(firewall.to_json()))
    write_atomic(out_dir / "TEST_FREEZE.json", body)
    status(db, write=True)
    return body


def open_test(db) -> dict:
    with _open(db) as con:
        plan = _plan(con)
        out_dir = _out_dir(con)
        frozen = _get(con, "freeze")
        if _get(con, "test_opened"):
            raise Refusal("TEST_ALREADY_OPENED: the external traversal runs exactly once")
    if not frozen:
        raise Refusal("NOT_FROZEN: TEST stays sealed until selector, model, preprocessing and tie rule are frozen")
    winner_sets = sorted({w["set_id"] for pop in frozen["winners"].values() for target in pop.values() for w in target.values() if w})
    if not winner_sets:
        raise Refusal("NO_WINNER_TO_TEST")
    tasks = WW.enumerate_tasks(plan, "test", set_ids=set(winner_sets), test_authorization=frozen["freeze_sha256"])
    fw_path = out_dir / "FIREWALL.json"
    firewall = BusinessObjectiveFirewall.from_json(fw_path.read_text()).open_test_traversal()
    opened_utc = WW._now()
    with _open(db) as con:
        con.execute("BEGIN IMMEDIATE")
        if _get(con, "test_opened"):
            raise Refusal("TEST_ALREADY_OPENED")
        _insert_tasks(con, tasks)
        _set(con, "test_opened", opened_utc)
    write_atomic(fw_path, json.loads(firewall.to_json()))
    write_atomic(out_dir / "TEST_OPENED.json", {"schema": "fs4.test_opened.v1", "opened_utc": opened_utc, "freeze_sha256": frozen["freeze_sha256"],
                                                "test_tasks": len(tasks), "winner_sets": winner_sets, "opened_once": True})
    status(db, write=True)
    return {"test_tasks": len(tasks), "opened_once": True, "opened_utc": opened_utc, "winner_sets": winner_sets}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--consolidated", type=Path, action="append", required=True)
    init.add_argument("--frontier-seal", type=Path, action="append", required=True)
    init.add_argument("--extractibility", type=Path, required=True)
    init.add_argument("--out-dir", type=Path, required=True)
    init.add_argument("--validation-year", type=int, default=2024)
    init.add_argument("--input-mode", action="append", choices=list(MODE_ORDER), default=None)
    st = sub.add_parser("status")
    st.add_argument("--write", action="store_true", help="also write STATUS.json in the campaign out dir")
    st.add_argument("--population")
    st.add_argument("--input-mode")
    st.add_argument("--split")
    ls = sub.add_parser("list")
    ls.add_argument("--set-id")
    ls.add_argument("--week-start")
    ls.add_argument("--input-mode")
    ls.add_argument("--split")
    ls.add_argument("--state")
    take = sub.add_parser("claim")
    take.add_argument("--owner", required=True)
    take.add_argument("--task-id")
    take.add_argument("--input-mode", choices=list(MODE_ORDER))
    take.add_argument("--split", choices=("validation", "test"))
    done = sub.add_parser("complete")
    done.add_argument("--owner", required=True)
    bad = sub.add_parser("fail")
    bad.add_argument("--owner", required=True)
    bad.add_argument("--task-id", required=True)
    bad.add_argument("--reason", required=True)
    pulse = sub.add_parser("heartbeat")
    pulse.add_argument("--owner", required=True)
    pulse.add_argument("--task-id", required=True)
    cl = sub.add_parser("close")
    cl.add_argument("--warehouse", type=Path, required=True, help="local warehouse file (DuckDB when available, else SQLite)")
    sub.add_parser("stage2", help="compute and enqueue the stage-2 list from the stage-1 RAW aggregates (docs/fs4/STAGE2_RULE.md)")
    sub.add_parser("freeze")
    sub.add_parser("open-test")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            out = initialize(args.db, args.consolidated, args.frontier_seal, args.extractibility, out_dir=args.out_dir,
                             validation_year=args.validation_year, input_modes=tuple(args.input_mode or ("RAW",)))
        elif args.action == "status":
            out = status(args.db, write=args.write, population=args.population, input_mode=args.input_mode, split=args.split)
        elif args.action == "list":
            out = list_tasks(args.db, set_id=args.set_id, week_start=args.week_start, input_mode=args.input_mode, split=args.split, state=args.state)
        elif args.action == "claim":
            out = claim(args.db, args.owner, task_id=args.task_id, input_mode=args.input_mode, split=args.split)
        elif args.action == "complete":
            out = complete(args.db, args.owner, json.load(sys.stdin))
        elif args.action == "heartbeat":
            out = heartbeat(args.db, args.owner, args.task_id)
        elif args.action == "fail":
            out = fail(args.db, args.owner, args.task_id, args.reason)
        elif args.action == "stage2":
            out = stage2(args.db)
        elif args.action == "close":
            out = close(args.db, warehouse_path=args.warehouse)
        elif args.action == "freeze":
            out = freeze(args.db)
        else:
            out = open_test(args.db)
    except (Refusal, F.Refusal, WW.Refusal, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(canonical({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(2) from exc
    print(canonical(out))


if __name__ == "__main__":
    main()
