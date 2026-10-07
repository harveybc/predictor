#!/usr/bin/env python3
"""Durable Phase-4 extractibility queue; scientific work is done by a separate runner.

The coordinator owns this SQLite file. Remote workers invoke claim/complete/status
on the coordinator over SSH; they never copy or open the queue database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "fs4.extractibility.task.v1"
ARMS = ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER")
LEASE_SECONDS = 3600
MAX_ATTEMPTS = 3
NAFT = "NOT_AVAILABLE_FOR_TRAIN"
#: Refusal codes that mean "this feature has no usable TRAIN rows in this fold": a terminal disposition
#: that stays INSIDE the denominator, never retried and never dropped.
NAFT_CODE_RE = re.compile(r"^(NO_TRAIN_OBSERVATIONS|INSUFFICIENT_TRAIN_[A-Z0-9_]+)\b")


class Refusal(ValueError):
    """A mismatch that must not become accepted evidence."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


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
            task_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'PENDING', owner TEXT, lease_until REAL,
            started_at REAL, finished_at REAL, result TEXT, attempt INTEGER NOT NULL DEFAULT 0);
    """)
    return con


def initialize(path, candidate_paths, folds):
    """Deduplicate feature/arm/fold units across all candidate subsets."""
    if not folds or len(set(folds)) != len(folds) or any(not f for f in folds):
        raise Refusal("FOLDS_MUST_BE_UNIQUE_AND_NONEMPTY")
    records = []
    all_tasks = {}
    for candidate_path in candidate_paths:
        source = json.loads(Path(candidate_path).read_text())
        closure_path = Path(candidate_path).with_name("PHASE_3_FILTER_COMPLETE.json")
        if closure_path.exists():
            closure = json.loads(closure_path.read_text())
            if (closure.get("state") != "PHASE_3_FILTER_COMPLETE" or
                    closure.get("closure_sha256") != source.get("phase3_closure_sha256")):
                raise Refusal("PHASE3_CLOSURE_MISMATCH")
        elif source.get("phase3_closure_sha256"):
            raise Refusal("PHASE3_CLOSURE_MISSING")
        population = source.get("population_id")
        identity = source.get("identity")
        if not population or not identity or source.get("final_selection") is not False:
            raise Refusal("CANDIDATE_IDENTITY_OR_STATE_INVALID")
        candidates = source.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise Refusal("EMPTY_CANDIDATES")
        features = set()
        for candidate in candidates:
            if candidate.get("population_id", population) != population or candidate.get("identity", identity) != identity:
                raise Refusal("MIXED_POPULATION_OR_IDENTITY")
            members = candidate.get("members")
            if not isinstance(members, list) or not members or len(members) != len(set(members)):
                raise Refusal("INVALID_SUBSET")
            if not candidate.get("target_id") or not candidate.get("subset_sha256"):
                raise Refusal("CANDIDATE_MISSING_TARGET_OR_DIGEST")
            features.update(members)
        seasonal_context = sorted(f for f in features if f.startswith("cal."))
        trainable_features = features - set(seasonal_context)
        records.append({"population_id": population, "identity": identity,
                        "candidate_sha256": hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
                        "candidate_count": len(candidates), "unique_features": len(features),
                        "seasonal_context": seasonal_context,
                        "extractor_features": len(trainable_features)})
        for feature in sorted(trainable_features):
            for fold in folds:
                for arm in ARMS:
                    payload = {"schema": SCHEMA, "population_id": population,
                               "identity": identity, "feature_id": feature,
                               "fold_id": fold, "arm": arm, "seed": 0}
                    task_id = _digest(payload)
                    all_tasks[task_id] = payload
    plan = {"schema": "fs4.extractibility.plan.v1", "folds": list(folds),
            "sources": sorted(records, key=lambda x: x["population_id"]),
            "arms": list(ARMS), "task_count": len(all_tasks)}
    plan_id = _digest(plan)
    with _open(path) as con:
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute("SELECT value FROM campaign WHERE key='plan_id'").fetchone()
        if prior and prior[0] != plan_id:
            raise Refusal("PLAN_CHANGED_USE_NEW_QUEUE_PATH")
        con.execute("INSERT OR IGNORE INTO campaign VALUES('plan_id', ?)", (plan_id,))
        con.execute("INSERT OR IGNORE INTO campaign VALUES('plan', ?)", (_canonical(plan),))
        con.executemany("INSERT OR IGNORE INTO tasks(task_id,kind,payload) VALUES(?, 'extractibility', ?)",
                        [(task_id, _canonical(payload)) for task_id, payload in all_tasks.items()])
    return {"plan_sha256": plan_id, "tasks": len(all_tasks), "sources": records}


def claim(path, owner, kind="extractibility", now=None, task_id=None, arm=None):
    if not owner or any(ch.isspace() for ch in owner):
        raise Refusal("INVALID_OWNER")
    now = time.time() if now is None else now
    with _open(path) as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("""UPDATE tasks SET state='FAILED',finished_at=?,result=?
            WHERE state='LEASED' AND lease_until<? AND attempt>=?""",
            (now, _canonical({"reason": "LEASE_EXPIRED_MAX_ATTEMPTS"}), now, MAX_ATTEMPTS))
        row = con.execute("""SELECT * FROM tasks WHERE kind=? AND (? IS NULL OR task_id=?)
            AND (? IS NULL OR json_extract(payload,'$.arm')=?) AND
            (state='PENDING' OR (state='LEASED' AND lease_until<? AND attempt<?))
            ORDER BY CASE json_extract(payload,'$.arm')
                WHEN 'RAW' THEN 0 WHEN 'RANDOM_ENCODER' THEN 1 ELSE 2 END,
                task_id LIMIT 1""", (kind, task_id, task_id, arm, arm, now, MAX_ATTEMPTS)).fetchone()
        if row is None:
            return None
        con.execute("""UPDATE tasks SET state='LEASED',owner=?,lease_until=?,started_at=?,
            attempt=attempt+1 WHERE task_id=?""", (owner, now + LEASE_SECONDS, now, row["task_id"]))
        return {**json.loads(row["payload"]), "task_id": row["task_id"],
                "lease_until": now + LEASE_SECONDS, "attempt": row["attempt"] + 1}


def heartbeat(path, owner, task_id, now=None):
    now = time.time() if now is None else now
    with _open(path) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT state,owner,lease_until FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner or row["lease_until"] < now:
            raise Refusal("NO_LIVE_LEASE_FOR_HEARTBEAT")
        con.execute("UPDATE tasks SET lease_until=? WHERE task_id=?", (now + LEASE_SECONDS, task_id))
    return {"task_id": task_id, "lease_until": now + LEASE_SECONDS}


def _validate_result(result, payload):
    if result.get("status") != "COMPLETE" or result.get("task_id") != _digest(payload):
        raise Refusal("RESULT_IDENTITY_OR_STATUS_MISMATCH")
    if result.get("seed") != payload["seed"]:
        raise Refusal("SEED_MISMATCH")
    for key in ("input_sha256", "code_sha256", "model_sha256", "rows_sha256", "mask_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise Refusal(f"INVALID_{key.upper()}")
    if type(result.get("population_n")) is not int or result["population_n"] < 1:
        raise Refusal("INVALID_POPULATION")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or "mae" not in metrics or "naive_mae" not in metrics:
        raise Refusal("MISSING_PAIRED_METRICS")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in metrics.values()):
        raise Refusal("INVALID_METRIC")
    _canonical(result)


def complete(path, owner, result, now=None):
    now = time.time() if now is None else now
    with _open(path) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM tasks WHERE task_id=?", (result.get("task_id"),)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner or row["lease_until"] < now:
            raise Refusal("NO_LIVE_LEASE_FOR_RESULT")
        payload = json.loads(row["payload"])
        _validate_result(result, payload)
        peers = con.execute("""SELECT result FROM tasks WHERE state='COMPLETE' AND
            json_extract(payload,'$.population_id')=? AND
            json_extract(payload,'$.feature_id')=? AND
            json_extract(payload,'$.fold_id')=?""",
            (payload["population_id"], payload["feature_id"], payload["fold_id"])).fetchall()
        for peer in peers:
            prior = json.loads(peer["result"])
            if any(prior[key] != result[key] for key in
                   ("input_sha256", "rows_sha256", "mask_sha256", "population_n")) or \
                    prior["metrics"]["naive_mae"] != result["metrics"]["naive_mae"]:
                raise Refusal("PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH")
        con.execute("UPDATE tasks SET state='COMPLETE',finished_at=?,result=? WHERE task_id=?",
                    (now, _canonical(result), row["task_id"]))
    return {"task_id": result["task_id"], "accepted": True}


def fail(path, owner, task_id, reason, now=None, technical=False):
    """Record a failure. A typed (scientific) refusal is terminal on the first call. A technical
    failure returns the task to PENDING while attempts remain (MAX_ATTEMPTS in total) and becomes
    terminal FAILED only when they are exhausted; the last reason is kept either way."""
    now = time.time() if now is None else now
    if not reason or len(reason) > 1000:
        raise Refusal("INVALID_REASON")
    with _open(path) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None or row["state"] != "LEASED" or row["owner"] != owner:
            raise Refusal("NO_LEASE_FOR_FAILURE")
        if technical and row["attempt"] < MAX_ATTEMPTS:
            con.execute("""UPDATE tasks SET state='PENDING',owner=NULL,lease_until=NULL,result=?
                WHERE task_id=?""", (_canonical({"last_technical_failure": reason, "attempt": row["attempt"]}), task_id))
            return {"task_id": task_id, "failed": True, "retry": True, "attempt": row["attempt"]}
        if not technical and NAFT_CODE_RE.match(reason):
            code = NAFT_CODE_RE.match(reason).group(1)
            con.execute("UPDATE tasks SET state=?,finished_at=?,result=? WHERE task_id=?",
                        (NAFT, now, _canonical({"status": NAFT, "code": code, "reason": reason,
                                                "attempt": row["attempt"]}), task_id))
            return {"task_id": task_id, "failed": False, "state": NAFT, "retry": False}
        con.execute("UPDATE tasks SET state='FAILED',finished_at=?,result=? WHERE task_id=?",
                    (now, _canonical({"reason": reason, "attempt": row["attempt"]}), task_id))
    return {"task_id": task_id, "failed": True, "retry": False}


def status(path, now=None, population=None, feature=None, fold=None, arm=None, owner=None):
    now = time.time() if now is None else now
    with _open(path) as con:
        plan = con.execute("SELECT value FROM campaign WHERE key='plan'").fetchone()
        if not plan:
            raise Refusal("NO_PLAN")
        rows = con.execute("SELECT state,owner,lease_until,started_at,finished_at,payload,result FROM tasks").fetchall()
    states = {"pending": 0, "running": 0, "complete": 0, "failed": 0, "not_available_for_train": 0}
    by_population = {}
    active_owners = set()
    durations = []
    for row in rows:
        payload = json.loads(row["payload"])
        if (population is not None and payload["population_id"] != population or
                feature is not None and payload["feature_id"] != feature or
                fold is not None and payload["fold_id"] != fold or
                arm is not None and payload["arm"] != arm or
                owner is not None and row["owner"] != owner):
            continue
        group = by_population.setdefault(payload["population_id"], {key: 0 for key in states})
        state = "pending" if row["state"] == "LEASED" and row["lease_until"] < now else row["state"].lower()
        state = "running" if state == "leased" else state
        if state == "failed" and row["result"] and NAFT_CODE_RE.match(str(json.loads(row["result"]).get("reason") or "")):
            state = "not_available_for_train"  # a refusal recorded before the explicit state existed
        states[state] += 1
        group[state] += 1
        if state == "running":
            active_owners.add(row["owner"])
        if state == "complete":
            durations.append(row["finished_at"] - row["started_at"])
    remaining = states["pending"] + states["running"]
    eta = (round(statistics.median(durations) * remaining / len(active_owners))
           if durations and active_owners else (0 if remaining == 0 else None))
    return {"schema": "fs4.status.v1", "plan": json.loads(plan[0]), "total": sum(states.values()),
            **states, "by_population": by_population, "workers": sorted(active_owners),
            "scope": {"population": population, "feature": feature, "fold": fold,
                      "arm": arm, "owner": owner},
            "eta_seconds": eta, "eta_basis": ("complete" if remaining == 0 else
                "observed median task duration / active workers" if eta is not None else
                "insufficient measured throughput"),
            "final_selection": False}


def list_tasks(path, feature=None, fold=None, arm=None):
    with _open(path) as con:
        rows = con.execute("SELECT task_id,state,payload FROM tasks ORDER BY task_id").fetchall()
    selected = []
    for row in rows:
        payload = json.loads(row["payload"])
        if (feature is None or payload["feature_id"] == feature) and \
           (fold is None or payload["fold_id"] == fold) and \
           (arm is None or payload["arm"] == arm):
            selected.append({**payload, "task_id": row["task_id"], "state": row["state"]})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--candidates", type=Path, action="append", required=True)
    init.add_argument("--fold", action="append", required=True)
    scoped = sub.add_parser("status")
    scoped.add_argument("--population")
    scoped.add_argument("--feature")
    scoped.add_argument("--fold")
    scoped.add_argument("--arm", choices=ARMS)
    scoped.add_argument("--owner")
    listing = sub.add_parser("list")
    listing.add_argument("--feature")
    listing.add_argument("--fold")
    listing.add_argument("--arm", choices=ARMS)
    take = sub.add_parser("claim")
    take.add_argument("--owner", required=True)
    take.add_argument("--task-id", help="Claim a specific pilot task")
    take.add_argument("--arm", choices=ARMS, help="Restrict a worker slot to one arm")
    done = sub.add_parser("complete")
    done.add_argument("--owner", required=True)
    bad = sub.add_parser("fail")
    bad.add_argument("--owner", required=True)
    bad.add_argument("--task-id", required=True)
    bad.add_argument("--reason", required=True)
    bad.add_argument("--technical", action="store_true",
                     help="technical failure: retry up to MAX_ATTEMPTS; without it the failure is a terminal typed refusal")
    pulse = sub.add_parser("heartbeat")
    pulse.add_argument("--owner", required=True)
    pulse.add_argument("--task-id", required=True)
    args = parser.parse_args()
    try:
        if args.action == "init":
            out = initialize(args.db, args.candidates, args.fold)
        elif args.action == "status":
            out = status(args.db, population=args.population, feature=args.feature,
                         fold=args.fold, arm=args.arm, owner=args.owner)
        elif args.action == "list":
            out = list_tasks(args.db, args.feature, args.fold, args.arm)
        elif args.action == "claim":
            out = claim(args.db, args.owner, task_id=args.task_id, arm=args.arm)
        elif args.action == "complete":
            out = complete(args.db, args.owner, json.load(sys.stdin))
        elif args.action == "heartbeat":
            out = heartbeat(args.db, args.owner, args.task_id)
        else:
            out = fail(args.db, args.owner, args.task_id, args.reason, technical=args.technical)
    except (Refusal, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(_canonical({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
    print(_canonical(out))


if __name__ == "__main__":
    main()
