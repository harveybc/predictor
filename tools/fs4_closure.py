#!/usr/bin/env python3
"""Phase-4 closure follower: controller terminals -> warehouse -> receipts -> closure, with a
generated STATUS.json. Coordinator only; one pass per invocation (systemd timer, every 2 min).

Order: ``docs/handoffs/SATOSHI_FS4_EXECUTION_2026_10_07.md`` step 3; plan §5 (STATUS.json
contents) and §6 FS4-07/08/09/13.

What one ``tick`` does, in order, from the task store (``queue_v2.sqlite``, opened READ-ONLY;
this program never writes to the controller):

1. For every task in state COMPLETE whose receipt file is not yet verified: submit the terminal
   (task payload + result exactly as the controller stored it) to the warehouse through
   ``tools/fs4_warehouse.open_warehouse`` (DuckDB file or the service URL; token from
   ``WAREHOUSE_TOKEN``), read it back by task_id, compare the stored terminal digest with the
   local one and write ``<state>/receipts/<task_id>.json``. A refused terminal is written to
   ``<state>/quarantine/<task_id>.json`` with the typed reason and is never retried silently.
2. Write ``<state>/STATUS.json`` from the task store and the receipt directory, never from logs
   (FS4-08): state, expected, complete, failed (technical / typed-refused), active, pending,
   last heartbeat, rate, ETA or null with the reason, and warehouse coverage.
3. When every admitted task is COMPLETE or typed-refused, no result is non-finite, every arm
   triple shares rows/mask/input/population_n/naive_mae (FS4-09), every COMPLETE task has a
   verified receipt and the warehouse reconciliation agrees with the controller's counts and
   digest, write ``<state>/EXTRACTIBILITY_COMPLETE.json`` (generated from evidence; idempotent).

A typed refusal is a FAILED task whose stored reason carries a declared refusal code (the
runner's "no TRAIN observations" class). A technical failure (lease expiry, runner crash, a
non-JSON result) blocks closure: it is reported, never counted as a disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs4_warehouse as wh  # noqa: E402

core = wh.core
Refusal = wh.Refusal

SCHEMA_STATUS = "fs4.closure_status.v1"
SCHEMA_CLOSURE = "fs4.extractibility_complete.v1"
SCHEMA_RECEIPT_FILE = "fs4.closure_receipt.v1"
SCHEMA_QUARANTINE = "fs4.closure_quarantine.v1"
LEASE_SECONDS = 3600  # tools/fs4_campaign.py::LEASE_SECONDS; a heartbeat renews to now + LEASE_SECONDS
#: FAILED reasons that are scientific dispositions, not technical failures. The runner's typed
#: refusal for a feature/fold without TRAIN observations is the declared class.
DEFAULT_REFUSAL_PATTERN = r"\b(NO_TRAIN_OBSERVATIONS|INSUFFICIENT_TRAIN_[A-Z0-9_]+|TYPED_REFUSAL(?::[A-Z0-9_]+)?|REFUSED_[A-Z0-9_]+)\b"
TECHNICAL_MARKERS = ("LEASE_EXPIRED_MAX_ATTEMPTS", "WORKER_TIMEOUT", "LEASE_HEARTBEAT_FAILED",
                     "runner did not emit one JSON result")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


# --------------------------------------------------------------------------- the task store
class TaskStore:
    """The controller's SQLite, read-only. Nothing here can change a task."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise Refusal(f"task store {self.path} does not exist")
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA query_only = 1")

    def plan(self) -> tuple[str, dict]:
        rows = {r["key"]: r["value"] for r in self.con.execute("SELECT key, value FROM campaign")}
        if "plan_id" not in rows or "plan" not in rows:
            raise Refusal("task store has no plan (fs4_campaign.py init was never run)")
        plan = json.loads(rows["plan"])
        if digest(plan) != rows["plan_id"]:
            raise Refusal("task store plan does not match its plan_id")
        return rows["plan_id"], plan

    def tasks(self) -> list[dict]:
        out = []
        for r in self.con.execute("SELECT task_id, kind, payload, state, owner, lease_until, started_at,"
                                  " finished_at, result, attempt FROM tasks ORDER BY task_id"):
            payload = json.loads(r["payload"])
            out.append({"task_id": r["task_id"], "kind": r["kind"], "payload": payload, "state": r["state"],
                        "owner": r["owner"], "lease_until": r["lease_until"], "started_at": r["started_at"],
                        "finished_at": r["finished_at"], "attempt": r["attempt"],
                        "result": json.loads(r["result"]) if r["result"] else None})
        return out

    def close(self) -> None:
        self.con.close()


NAFT = "NOT_AVAILABLE_FOR_TRAIN"
NAFT_CODE_RE = re.compile(r"^(NO_TRAIN_OBSERVATIONS|INSUFFICIENT_TRAIN_[A-Z0-9_]+)\b")


def is_not_available(task: dict) -> bool:
    """A terminal NOT_AVAILABLE_FOR_TRAIN disposition: a FAILED row whose stored result says so (the controller's
    record), or whose reason starts with a no-TRAIN-rows code (rows written before the status was stored)."""
    if task["state"] == NAFT:
        return True
    if task["state"] == "FAILED" and isinstance(task.get("result"), dict) and task["result"].get("status") == NAFT:
        return True
    reason = str((task.get("result") or {}).get("reason") or "") if isinstance(task.get("result"), dict) else ""
    return task["state"] == "FAILED" and bool(NAFT_CODE_RE.match(reason))


def effective_state(task: dict, now: float) -> str:
    """The controller's own reading (``fs4_campaign.status``): an expired lease is pending again."""
    if is_not_available(task):
        return "not_available_for_train"
    if task["state"] == "LEASED":
        return "running" if (task["lease_until"] or 0) >= now else "pending"
    return task["state"].lower()


def is_typed_refusal(task: dict, pattern: re.Pattern) -> bool:
    if is_not_available(task):
        return False  # a NOT_AVAILABLE_FOR_TRAIN cell is its own terminal state, not a generic refusal
    if task["state"] != "FAILED" or not isinstance(task.get("result"), dict):
        return False
    reason = str(task["result"].get("reason") or "")
    if any(marker in reason for marker in TECHNICAL_MARKERS):
        return False
    return bool(pattern.search(reason))


def expected_counts(tasks: list[dict], plan: dict) -> dict:
    by_population: dict[str, int] = {}
    for task in tasks:
        pop = task["payload"]["population_id"]
        by_population[pop] = by_population.get(pop, 0) + 1
    expected = {"total": len(tasks), "by_population": dict(sorted(by_population.items()))}
    if plan.get("task_count") is not None and int(plan["task_count"]) != len(tasks):
        raise Refusal(f"task store holds {len(tasks)} tasks but the plan declares {plan['task_count']}")
    return expected


def terminal_document(task: dict, plan_sha256: str) -> dict:
    return {"schema": core.SCHEMA_TERMINAL, "plan_sha256": plan_sha256,
            "task": {**task["payload"], "task_id": task["task_id"]}, "result": task["result"],
            "owner": task["owner"], "attempt": task["attempt"],
            "started_at": task["started_at"], "finished_at": task["finished_at"]}


def local_terminal_sha256(task: dict) -> str:
    return hashlib.sha256(core.canonical_bytes(task["result"])).hexdigest()


def validate_complete(task: dict) -> str | None:
    """Re-validate a COMPLETE result with the store's rule; the reason when it is not acceptable."""
    try:
        core.prepare_terminal(terminal_document(task, "0" * 64))
    except Refusal as exc:
        return str(exc)
    return None


def triple_check(tasks: list[dict]) -> list[dict]:
    """FS4-09 from the task store: arm triples whose shared identity disagrees."""
    groups: dict[tuple, dict[str, tuple]] = {}
    for task in tasks:
        if task["state"] != "COMPLETE" or not isinstance(task.get("result"), dict):
            continue
        p, r = task["payload"], task["result"]
        key = (p["population_id"], p["identity"], p["feature_id"], p["fold_id"])
        sig = tuple(r.get(k) for k in core.SHARED_ACROSS_ARMS) + ((r.get("metrics") or {}).get("naive_mae"),)
        groups.setdefault(key, {})[p["arm"]] = sig
    bad = []
    for key, arms in sorted(groups.items()):
        if len(set(arms.values())) > 1:
            bad.append({"population_id": key[0], "identity": key[1], "feature_id": key[2], "fold_id": key[3],
                        "arms": sorted(arms)})
    return bad


def mixed_not_available(tasks: list[dict]) -> list[dict]:
    """The three arms of a feature-fold share their rows, so they must share the disposition: a cell that is
    NOT_AVAILABLE_FOR_TRAIN for one arm and measured for another is a defect, not a result."""
    groups: dict[tuple, set[str]] = {}
    for task in tasks:
        p = task["payload"]
        kind = "NAFT" if is_not_available(task) else ("COMPLETE" if task["state"] == "COMPLETE" else None)
        if kind:
            groups.setdefault((p["population_id"], p["identity"], p["feature_id"], p["fold_id"]), set()).add(kind)
    return [{"population_id": k[0], "identity": k[1], "feature_id": k[2], "fold_id": k[3],
             "arms": ["MIXED_NOT_AVAILABLE_AND_MEASURED"]} for k, kinds in sorted(groups.items()) if len(kinds) > 1]


def build_populations(tasks: list[dict]) -> dict:
    """populations.<POP>.features.<feature> for the weekly campaign: MEASURED with folds -> arm -> MAE, or
    NOT_AVAILABLE_FOR_TRAIN when no fold of the feature has TRAIN rows. A feature that is available only in
    some folds is MEASURED and carries {status: NOT_AVAILABLE_FOR_TRAIN} in the folds that are not."""
    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for task in tasks:
        p = task["payload"]
        fold = cells.setdefault(p["population_id"], {}).setdefault(p["feature_id"], {}).setdefault(p["fold_id"], {})
        if is_not_available(task):
            fold["status"] = NAFT
            fold["code"] = (task["result"] or {}).get("code") or NAFT_CODE_RE.match(str((task["result"] or {}).get("reason") or "")).group(1)
        elif task["state"] == "COMPLETE":
            fold[p["arm"]] = task["result"]["metrics"]["mae"]
            fold["naive_mae"] = task["result"]["metrics"]["naive_mae"]
            fold["population_n"] = task["result"]["population_n"]
    out: dict[str, Any] = {}
    for pop, features in sorted(cells.items()):
        record: dict[str, Any] = {}
        for feature, folds in sorted(features.items()):
            if folds and all(f.get("status") == NAFT for f in folds.values()):
                record[feature] = {"status": NAFT}
            else:
                record[feature] = {"status": "MEASURED", "folds": dict(sorted(folds.items()))}
        out[pop] = {"features": record,
                    "feature_count": len(record),
                    "measured": sum(1 for v in record.values() if v["status"] == "MEASURED"),
                    "not_available_for_train": sum(1 for v in record.values() if v["status"] == NAFT)}
    return out


# --------------------------------------------------------------------------- receipts on disk
class ReceiptDir:
    def __init__(self, state_root: Path):
        self.root = Path(state_root)
        self.receipts = self.root / "receipts"
        self.quarantine = self.root / "quarantine"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(parents=True, exist_ok=True)

    def verified(self, task_id: str) -> dict | None:
        path = self.receipts / f"{task_id}.json"
        if not path.is_file():
            return None
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if doc.get("schema") != SCHEMA_RECEIPT_FILE or doc.get("task_id") != task_id or not doc.get("readback_verified"):
            return None
        return doc

    def all_verified(self) -> dict[str, dict]:
        out = {}
        for path in sorted(self.receipts.glob("*.json")):
            doc = self.verified(path.stem)
            if doc is not None:
                out[path.stem] = doc
        return out

    def quarantined(self) -> dict[str, dict]:
        out = {}
        for path in sorted(self.quarantine.glob("*.json")):
            try:
                out[path.stem] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                out[path.stem] = {"reason": "unreadable quarantine file"}
        return out

    def write_receipt(self, task_id: str, doc: dict) -> None:
        atomic_json(self.receipts / f"{task_id}.json", {"schema": SCHEMA_RECEIPT_FILE, "task_id": task_id, **doc})
        stale = self.quarantine / f"{task_id}.json"
        if stale.exists():
            stale.unlink()

    def write_quarantine(self, task_id: str, reason: str, **extra) -> None:
        atomic_json(self.quarantine / f"{task_id}.json",
                    {"schema": SCHEMA_QUARANTINE, "task_id": task_id, "reason": reason,
                     "quarantined_at": core.now(), **extra})


# --------------------------------------------------------------------------- submission
def submit_pending(store: Any, receipts: ReceiptDir, tasks: list[dict], plan_sha256: str, *,
                   host_role: str | None, batch: int, log=lambda *_: None) -> dict:
    """Submit every COMPLETE task without a verified receipt; verify by readback; receipt or quarantine."""
    verified = receipts.all_verified()
    pending = [t for t in tasks if t["state"] == "COMPLETE" and t["task_id"] not in verified]
    outcome = {"submitted": 0, "verified": 0, "quarantined": 0, "already_verified": len(verified), "errors": []}
    for i in range(0, len(pending), batch):
        chunk = pending[i:i + batch]
        docs = []
        for task in chunk:
            problem = validate_complete(task)
            if problem:
                receipts.write_quarantine(task["task_id"], f"INVALID_TERMINAL: {problem}")
                outcome["quarantined"] += 1
                continue
            docs.append((task, terminal_document(task, plan_sha256)))
        if not docs:
            continue
        try:
            receipt = store.submit_terminals(plan_sha256, [d for _, d in docs], host_role=host_role)
            groups = [(docs, receipt)]
        except Refusal as exc:
            log(f"batch refused ({exc}); isolating terminals one by one")
            groups = []
            for task, doc in docs:
                try:
                    groups.append(([(task, doc)], store.submit_terminals(plan_sha256, [doc], host_role=host_role)))
                except Refusal as single:
                    receipts.write_quarantine(task["task_id"], f"WAREHOUSE_REFUSED: {single}")
                    outcome["quarantined"] += 1
        for members, receipt in groups:
            outcome["submitted"] += len(members)
            for task, _doc in members:
                expected_sha = local_terminal_sha256(task)
                back = store.read_terminals(plan_sha256, task_id=task["task_id"])
                if len(back) != 1 or back[0].get("terminal_sha256") != expected_sha \
                        or back[0].get("result") != task["result"]:
                    receipts.write_quarantine(task["task_id"], "READBACK_MISMATCH",
                                              expected_terminal_sha256=expected_sha,
                                              stored_terminal_sha256=(back[0].get("terminal_sha256") if back else None),
                                              receipt_sha256=receipt["receipt_sha256"])
                    outcome["quarantined"] += 1
                    continue
                receipts.write_receipt(task["task_id"], {
                    "plan_sha256": plan_sha256, "receipt_sha256": receipt["receipt_sha256"],
                    "receipt": receipt, "terminal_sha256": expected_sha, "readback_verified": True,
                    "backend": receipt.get("backend"), "verified_at": core.now(),
                    "population_id": task["payload"]["population_id"], "arm": task["payload"]["arm"]})
                outcome["verified"] += 1
    return outcome


# --------------------------------------------------------------------------- status (FS4-08)
def build_status(tasks: list[dict], plan_sha256: str, plan: dict, receipts: ReceiptDir, *, now: float,
                 rate_window: float, refusal_pattern: re.Pattern, warehouse: dict | None = None,
                 closure_path: Path | None = None) -> dict:
    expected = expected_counts(tasks, plan)
    counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0, "not_available_for_train": 0}
    by_population: dict[str, dict] = {}
    workers: set[str] = set()
    heartbeats: list[float] = []
    durations: list[float] = []
    recent: list[float] = []
    technical, typed = [], []
    invalid = []
    for task in tasks:
        state = effective_state(task, now)
        counts[state] += 1
        pop = by_population.setdefault(task["payload"]["population_id"],
                                       {"expected": 0, "pending": 0, "running": 0, "complete": 0, "failed": 0,
                                        "not_available_for_train": 0, "typed_refused": 0, "technical_failed": 0})
        pop["expected"] += 1
        pop[state] += 1
        if state == "running":
            workers.add(task["owner"])
            heartbeats.append(task["lease_until"] - LEASE_SECONDS)
        if state == "complete":
            if task["started_at"] is not None and task["finished_at"] is not None:
                durations.append(task["finished_at"] - task["started_at"])
                if task["finished_at"] >= now - rate_window:
                    recent.append(task["finished_at"])
            problem = validate_complete(task)
            if problem:
                invalid.append({"task_id": task["task_id"], "reason": problem})
        if state == "failed":
            entry = {"task_id": task["task_id"], "population_id": task["payload"]["population_id"],
                     "feature_id": task["payload"]["feature_id"], "fold_id": task["payload"]["fold_id"],
                     "arm": task["payload"]["arm"], "attempt": task["attempt"],
                     "reason": (task["result"] or {}).get("reason")}
            if is_typed_refusal(task, refusal_pattern):
                typed.append(entry)
                pop["typed_refused"] += 1
            else:
                technical.append(entry)
                pop["technical_failed"] += 1
    verified = receipts.all_verified()
    quarantined = receipts.quarantined()
    complete_ids = {t["task_id"] for t in tasks if t["state"] == "COMPLETE"}
    remaining = counts["pending"] + counts["running"]
    rate = len(recent) / (rate_window / 3600.0) if recent else None
    if remaining == 0:
        eta, eta_reason = 0, "no task pending or running"
    elif durations and workers:
        eta = round(statistics.median(durations) * remaining / len(workers))
        eta_reason = "observed median task duration x remaining / active workers (task store)"
    elif not durations:
        eta, eta_reason = None, "no completed task with measured duration yet"
    else:
        eta, eta_reason = None, "no active worker holds a live lease"
    triples_bad = triple_check(tasks) + mixed_not_available(tasks)
    reasons = []
    if remaining:
        reasons.append(f"{remaining} tasks pending or running")
    if technical:
        reasons.append(f"{len(technical)} technical failures (not dispositions)")
    if invalid:
        reasons.append(f"{len(invalid)} COMPLETE results fail validation (non-finite or malformed)")
    if triples_bad:
        reasons.append(f"{len(triples_bad)} arm triples disagree on rows/mask/input/population_n/naive_mae")
    missing_receipts = sorted(complete_ids - set(verified))
    if missing_receipts:
        reasons.append(f"{len(missing_receipts)} COMPLETE tasks without a verified warehouse receipt")
    if quarantined:
        reasons.append(f"{len(quarantined)} terminals quarantined")
    if warehouse and warehouse.get("error"):
        reasons.append(f"warehouse unreachable or refused: {warehouse['error']}")
    closed = closure_path is not None and closure_path.is_file()
    if closed:
        state = "EXTRACTIBILITY_COMPLETE"
    elif counts["complete"] == 0 and counts["running"] == 0 and counts["failed"] == 0 \
            and counts["not_available_for_train"] == 0:
        state = "PENDING_DISPATCH"
    elif technical or invalid or triples_bad or quarantined:
        state = "BLOCKED"
    elif remaining:
        state = "RUNNING"
    elif missing_receipts:
        state = "LOADING_WAREHOUSE"
    else:
        state = "READY_TO_CLOSE"
    status = {
        "schema": SCHEMA_STATUS, "generated_at": iso(now), "generated_epoch": now, "source": "task_store",
        "state": state, "plan_sha256": plan_sha256, "expected": expected,
        "complete": counts["complete"], "pending": counts["pending"], "active": counts["running"],
        "failed": {"total": counts["failed"], "technical": len(technical), "typed_refused": len(typed)},
        "not_available_for_train": {"total": counts["not_available_for_train"],
                                    "by_population": {p: v["not_available_for_train"] for p, v in sorted(by_population.items())}},
        "by_population": by_population, "workers": sorted(w for w in workers if w),
        "last_heartbeat": iso(max(heartbeats)) if heartbeats else None,
        "rate_tasks_per_hour": rate, "rate_window_seconds": rate_window,
        "eta_seconds": eta, "eta_reason": eta_reason,
        "warehouse": {**(warehouse or {}), "verified_receipts": len(verified),
                      "pending_submit": len(missing_receipts), "quarantined": len(quarantined)},
        "validation": {"invalid_complete": invalid[:20], "inconsistent_triples": triples_bad[:20],
                       "technical_failures": technical[:20]},
        "closure": {"ready": not reasons and not closed, "closed": closed, "reasons": reasons},
        "final_selection": False,
    }
    return status


# --------------------------------------------------------------------------- closure
def attempt_close(store: Any, tasks: list[dict], plan_sha256: str, plan: dict, receipts: ReceiptDir, status: dict,
                  closure_path: Path, *, refusal_pattern: re.Pattern) -> dict | None:
    """Write EXTRACTIBILITY_COMPLETE.json only from evidence; None when a condition fails."""
    if closure_path.is_file():
        return json.loads(closure_path.read_text())
    if not status["closure"]["ready"]:
        return None
    complete = [t for t in tasks if t["state"] == "COMPLETE"]
    typed = [t for t in tasks if is_typed_refusal(t, refusal_pattern)]
    naft = [t for t in tasks if is_not_available(t)]
    if len(complete) + len(typed) + len(naft) != len(tasks):
        return None
    expected_store = {"total": len(complete), "by_population": {}}
    for t in complete:
        pop = t["payload"]["population_id"]
        expected_store["by_population"][pop] = expected_store["by_population"].get(pop, 0) + 1
    verified = receipts.all_verified()
    receipt_docs = {}
    for doc in verified.values():
        receipt_docs[doc["receipt_sha256"]] = doc["receipt"]
    reconciliation = store.reconcile(plan_sha256, expected_store, list(receipt_docs.values()))
    local_digest = core.terminals_digest(local_terminal_sha256(t) for t in complete)
    if not reconciliation.get("complete") or reconciliation["stored"]["terminals_sha256"] != local_digest \
            or reconciliation.get("receipts_verified") != len(receipt_docs):
        status["closure"]["reasons"].append("warehouse reconciliation does not agree with the task store")
        status["closure"]["ready"] = False
        status["state"] = "BLOCKED"
        status["reconciliation"] = reconciliation
        return None
    closure = {
        "schema": SCHEMA_CLOSURE, "state": "EXTRACTIBILITY_COMPLETE", "plan_sha256": plan_sha256, "plan": plan,
        "admitted": status["expected"], "complete": len(complete),
        "typed_refused": len(typed),
        "not_available_for_train": len(naft),
        "denominator": {"admitted": len(tasks), "complete": len(complete), "not_available_for_train": len(naft),
                        "typed_refused": len(typed), "sum_equals_admitted": len(complete) + len(naft) + len(typed) == len(tasks)},
        "populations": build_populations(tasks),
        "typed_refusals": [{"task_id": t["task_id"], **{k: t["payload"][k] for k in ("population_id", "feature_id", "fold_id", "arm")},
                            "reason": (t["result"] or {}).get("reason")} for t in typed],
        "complete_by_population": expected_store["by_population"],
        "terminals_sha256": local_digest,
        "triples": reconciliation["triples"],
        "receipts": {"count": len(verified), "distinct_receipts": len(receipt_docs),
                     "receipts_sha256": core.terminals_digest(receipt_docs)},
        "reconciliation": reconciliation, "warehouse_backend": reconciliation.get("backend"),
        "generated_at": core.now(), "final_selection": False,
    }
    closure["closure_sha256"] = digest({k: v for k, v in closure.items() if k not in ("generated_at", "closure_sha256")})
    atomic_json(closure_path, closure)
    status["closure"]["closed"] = True
    status["closure"]["ready"] = False
    status["state"] = "EXTRACTIBILITY_COMPLETE"
    return closure


# --------------------------------------------------------------------------- one pass
def tick(args, *, submit: bool = True, close: bool = True, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    pattern = re.compile(args.refusal_pattern)
    state_root = Path(args.state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / "closure.log"

    def log(message: str) -> None:
        with log_path.open("a") as stream:
            stream.write(f"{iso(time.time())} {message}\n")

    tasks_store = TaskStore(args.db)
    try:
        plan_sha256, plan = tasks_store.plan()
        tasks = tasks_store.tasks()
    finally:
        tasks_store.close()
    receipts = ReceiptDir(state_root)
    closure_path = state_root / "EXTRACTIBILITY_COMPLETE.json"
    warehouse_info: dict[str, Any] = {"target": _role_only(args.warehouse)}
    store = None
    if (submit or close) and not closure_path.is_file():
        try:
            store = wh.open_warehouse(args.warehouse)
            warehouse_info["backend"] = store.backend
            if submit:
                warehouse_info["last_pass"] = submit_pending(store, receipts, tasks, plan_sha256, host_role=args.host_role,
                                                             batch=args.batch, log=log)
        except Refusal as exc:
            warehouse_info["error"] = f"refused: {exc}"
            log(f"warehouse refusal: {exc}")
        except Exception as exc:  # noqa: BLE001 - transport; reported in STATUS, retried next tick
            warehouse_info["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            log(f"warehouse unreachable: {exc}")
    status = build_status(tasks, plan_sha256, plan, receipts, now=now, rate_window=args.rate_window_seconds,
                          refusal_pattern=pattern, warehouse=warehouse_info, closure_path=closure_path)
    if close and store is not None and not warehouse_info.get("error"):
        try:
            closure = attempt_close(store, tasks, plan_sha256, plan, receipts, status, closure_path, refusal_pattern=pattern)
            if closure:
                log(f"EXTRACTIBILITY_COMPLETE written: {closure['closure_sha256']}")
        except Refusal as exc:
            status["closure"]["reasons"].append(f"closure refused: {exc}")
            status["closure"]["ready"] = False
            status["state"] = "BLOCKED"
    if store is not None:
        store.close()
    atomic_json(state_root / "STATUS.json", status)
    return status


def _role_only(target: str) -> str:
    """Never echo a host name: a URL is reported by scheme only, a file by its basename."""
    if target.startswith(("http://", "https://")):
        return "service"
    return Path(target).name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, type=Path, help="the controller's queue_v2.sqlite (opened read-only)")
    parser.add_argument("--warehouse", required=True, help="DuckDB file or service URL (token from WAREHOUSE_TOKEN)")
    parser.add_argument("--state-root", required=True, type=Path, help="receipts/, quarantine/, STATUS.json, closure")
    parser.add_argument("--host-role", default="coordinator", choices=core.HOST_ROLES)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--rate-window-seconds", type=float, default=7200.0)
    parser.add_argument("--refusal-pattern", default=DEFAULT_REFUSAL_PATTERN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tick", help="submit new terminals, verify readback, write STATUS.json, close when complete")
    sub.add_parser("status", help="write and print STATUS.json from the task store only (no warehouse call)")
    sub.add_parser("close", help="attempt the closure without submitting (what is receipted must suffice)")
    sub.add_parser("expected", help="print the controller's expected counts {total, by_population}")
    args = parser.parse_args(argv)
    try:
        if args.command == "expected":
            tasks_store = TaskStore(args.db)
            _, plan = tasks_store.plan()
            out = expected_counts(tasks_store.tasks(), plan)
            tasks_store.close()
        elif args.command == "status":
            out = tick(args, submit=False, close=False)
        elif args.command == "close":
            out = tick(args, submit=False, close=True)
        else:
            out = tick(args)
    except Refusal as exc:
        print(json.dumps({"refused": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(out, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
