"""Phase-4 extractibility terminals: additive DDL, idempotent submit, readback, reconciliation.

One implementation serves three callers so they cannot disagree about identity or digests:

* ``tools/fs4_warehouse.py`` on a local DuckDB file (throwaway tests, a local terminal store);
* the packaged backends (``predictor_olap_store.query.Plugin`` and the DuckDB provider) inside
  the warehouse service, through the host's ``/api/v2/fs4/*`` routes;
* ``tools/olap_duckdb_migrate.py``, whose snapshot boundary names the same relations.

The unit of storage is one controller terminal (``tools/fs4_campaign.py``): the task payload
the controller planned (``fs4.extractibility.task.v1``: population, identity, feature, fold,
arm, seed) and the COMPLETE result the worker delivered (``_validate_result``: digests of
input/code/model/rows/mask, population_n, paired ``mae``/``naive_mae``). The warehouse row is
keyed by ``task_id`` and that key is **recomputed** from the payload on every submission: a
document whose ``task_id`` is not the sha256 of its canonical payload is refused, so no row
can claim a task the controller never planned.

Identity: ``task_id`` PRIMARY KEY, plus UNIQUE over the full typed identity
``(plan_sha256, population_id, identity, feature_id, fold_id, arm, seed)``. CHECK constraints:
``arm`` in {RAW, RANDOM_ENCODER, TRAINED_ENCODER}, ``seed = 0``, ``population_n >= 1``, every
metric finite and non-negative. The same ``task_id`` with a different terminal digest is
refused; an identical replay is a no-op (duplicates_ignored). The three arms of one
(population, feature, fold) must share ``rows_sha256``, ``mask_sha256``, ``input_sha256``,
``population_n`` and ``naive_mae`` (FS4-09): a peer that disagrees is refused before any write,
and ``reconcile`` reports the triples that disagree in what is already stored.

Digest rule: ``terminal_sha256 = sha256(canonical_json(result))`` of the result exactly as the
controller stored it (sorted keys, compact, ascii, no NaN); ``terminals_sha256 =
sha256(''.join(sorted(terminal_sha256)))``. The SQL twin is
``sha256(string_agg(terminal_sha256, '' ORDER BY terminal_sha256))``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

SCHEMA_TASK = "fs4.extractibility.task.v1"
SCHEMA_TERMINAL = "fs4.extractibility.terminal.v1"
SCHEMA_RECEIPT = "fs4.warehouse_receipt.v1"
SCHEMA_RECONCILIATION = "fs4.warehouse_reconcile.v1"
MIGRATION_ID = "fs4_0001"

TABLE = "feature_extractibility_v1"
RECEIPT_TABLE = "fs4_load_receipt"
RECEIPT_TASK_TABLE = "fs4_load_receipt_task"
VIEWS = ("fs4_extractibility_coverage", "fs4_extractibility_triples")
ALL_TABLES = (TABLE, RECEIPT_TABLE, RECEIPT_TASK_TABLE)
ALL_RELATIONS = ALL_TABLES + VIEWS

ARMS = ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER")
HOST_ROLES = ("coordinator", "worker_a", "worker_b")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()

#: The task payload fields, in the controller's order; ``task_id = sha256(canonical(payload))``.
TASK_FIELDS = ("schema", "population_id", "identity", "feature_id", "fold_id", "arm", "seed")
#: Result fields the controller validates; all must be present and well formed.
RESULT_DIGESTS = ("input_sha256", "code_sha256", "model_sha256", "rows_sha256", "mask_sha256")
#: What the three arms of one feature x fold must share (FS4-09).
SHARED_ACROSS_ARMS = ("input_sha256", "rows_sha256", "mask_sha256", "population_n")

#: Typed identity columns (UNIQUE together) -> source in the terminal document.
IDENTITY = ("plan_sha256", "population_id", "identity", "feature_id", "fold_id", "arm", "seed")
#: Column -> type. Order here is the column order of the table after the identity.
_VALUE_TYPES = (
    ("rows_sha256", "TEXT NOT NULL"), ("mask_sha256", "TEXT NOT NULL"),
    ("input_sha256", "TEXT NOT NULL"), ("code_sha256", "TEXT NOT NULL"),
    ("model_sha256", "TEXT NOT NULL"), ("population_n", "BIGINT NOT NULL"),
    ("mae", "DOUBLE NOT NULL"), ("naive_mae", "DOUBLE NOT NULL"), ("mse", "DOUBLE"),
    ("chosen_epoch", "INTEGER"), ("updates", "BIGINT"),
    ("cpu_s", "DOUBLE"), ("wall_s", "DOUBLE"), ("peak_ram_bytes", "BIGINT"), ("peak_vram_bytes", "BIGINT"),
    ("host_role", "TEXT"), ("owner", "TEXT"), ("attempt", "INTEGER"),
    ("started_at", "DOUBLE"), ("finished_at", "DOUBLE"),
    ("terminal_sha256", "TEXT NOT NULL"), ("task_json", "TEXT NOT NULL"), ("result_json", "TEXT NOT NULL"),
)
VALUE_COLUMNS = tuple(name for name, _ in _VALUE_TYPES)
COLUMNS = ("task_id", *IDENTITY, *VALUE_COLUMNS)


class Refusal(ValueError):
    """A typed refusal: the request is wrong, not the store. Answers 400 at the host."""


# --------------------------------------------------------------------------- canonical bytes
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def terminals_digest(terminal_digests: Iterable[str]) -> str:
    """sha256 of the sorted concatenation of per-terminal digests; EMPTY_DIGEST for none."""
    return hashlib.sha256("".join(sorted(terminal_digests)).encode("ascii")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def task_id_of(payload: dict) -> str:
    """The controller's rule (``tools/fs4_campaign.py::_digest`` over the planned payload)."""
    return digest({k: payload[k] for k in TASK_FIELDS})


# --------------------------------------------------------------------------- DDL
def _finite(column: str, dialect: str) -> str:
    """A finite, non-negative DOUBLE. DuckDB has isfinite(); SQLite stores NaN as NULL (caught by
    NOT NULL) and compares infinities, so the magnitude bound is the finiteness test there."""
    if dialect == "duckdb":
        return f"isfinite({column}) AND {column} >= 0"
    return f"{column} = {column} AND abs({column}) < 1e308 AND {column} >= 0"


def ddl(qualified: Callable[[str], str] = lambda n: n, dialect: str = "duckdb") -> list[str]:
    """Additive DDL: CREATE TABLE/VIEW IF NOT EXISTS only. Nothing dropped or altered."""
    t = qualified
    create_view = "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
    arms = ", ".join(f"'{a}'" for a in ARMS)
    roles = ", ".join(f"'{r}'" for r in HOST_ROLES)
    parts = [" task_id TEXT PRIMARY KEY",
             " plan_sha256 TEXT NOT NULL", " population_id TEXT NOT NULL", " identity TEXT NOT NULL",
             " feature_id TEXT NOT NULL", " fold_id TEXT NOT NULL",
             f" arm TEXT NOT NULL CHECK (arm IN ({arms}))",
             " seed INTEGER NOT NULL CHECK (seed = 0)"]
    parts += [f" {name} {kind}" for name, kind in _VALUE_TYPES]
    parts += [" stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
              " CHECK (population_n >= 1)",
              f" CHECK ({_finite('mae', dialect)})",
              f" CHECK ({_finite('naive_mae', dialect)})",
              f" CHECK (mse IS NULL OR ({_finite('mse', dialect)}))",
              " CHECK (chosen_epoch IS NULL OR chosen_epoch >= 0)",
              " CHECK (updates IS NULL OR updates >= 0)",
              " CHECK (cpu_s IS NULL OR cpu_s >= 0)", " CHECK (wall_s IS NULL OR wall_s >= 0)",
              " CHECK (peak_ram_bytes IS NULL OR peak_ram_bytes >= 0)",
              " CHECK (peak_vram_bytes IS NULL OR peak_vram_bytes >= 0)",
              f" CHECK (host_role IS NULL OR host_role IN ({roles}))",
              f" UNIQUE ({', '.join(IDENTITY)})"]
    statements = [
        f"CREATE TABLE IF NOT EXISTS {t(TABLE)} ({','.join(parts)})",
        f"CREATE TABLE IF NOT EXISTS {t(RECEIPT_TABLE)} ("
        " receipt_sha256 TEXT PRIMARY KEY, plan_sha256 TEXT NOT NULL,"
        " terminal_count INTEGER NOT NULL, inserted INTEGER NOT NULL,"
        " duplicates_ignored INTEGER NOT NULL, terminals_sha256 TEXT NOT NULL,"
        " host_role TEXT, submitted_at TEXT NOT NULL)",
        f"CREATE TABLE IF NOT EXISTS {t(RECEIPT_TASK_TABLE)} ("
        " receipt_sha256 TEXT NOT NULL, task_id TEXT NOT NULL, terminal_sha256 TEXT NOT NULL,"
        " PRIMARY KEY (receipt_sha256, task_id))",
        f"{create_view} {t('fs4_extractibility_coverage')} AS"
        " SELECT plan_sha256, population_id, arm, count(*) AS terminals,"
        " count(DISTINCT feature_id) AS features, count(DISTINCT fold_id) AS folds,"
        " min(mae) AS min_mae, max(mae) AS max_mae"
        f" FROM {t(TABLE)} GROUP BY plan_sha256, population_id, arm",
        f"{create_view} {t('fs4_extractibility_triples')} AS"
        " SELECT plan_sha256, population_id, identity, feature_id, fold_id,"
        " count(*) AS arms, count(DISTINCT rows_sha256) AS rows_digests,"
        " count(DISTINCT mask_sha256) AS mask_digests, count(DISTINCT input_sha256) AS input_digests,"
        " count(DISTINCT population_n) AS population_ns, count(DISTINCT naive_mae) AS naive_maes"
        f" FROM {t(TABLE)} GROUP BY plan_sha256, population_id, identity, feature_id, fold_id",
    ]
    return statements


def render_sql() -> str:
    """The migration file content: the same statements, for humans."""
    head = (
        f"-- {MIGRATION_ID}: additive schema for feature-selection phase 4 (extractibility).\n"
        "-- GENERATED from predictor_olap_store.fs4_store.ddl(); do not edit by hand.\n"
        "-- Every statement is CREATE ... IF NOT EXISTS. Nothing is dropped, altered or rewritten.\n"
        "-- Identity: task_id (PRIMARY KEY) = sha256 of the controller's canonical task payload,\n"
        "-- AND UNIQUE over (plan_sha256, population_id, identity, feature_id, fold_id, arm, seed).\n"
        "-- CHECK: arm in {RAW, RANDOM_ENCODER, TRAINED_ENCODER}, seed = 0, population_n >= 1,\n"
        "-- mae / naive_mae / mse finite and non-negative. terminal_sha256 digests the result as\n"
        "-- the controller stored it; result_json holds that JSON; stored_at is never digested.\n\n")
    return head + ";\n\n".join(ddl()) + ";\n"


# --------------------------------------------------------------------------- execution shim
def _is_sqlalchemy(conn) -> bool:
    return hasattr(conn, "exec_driver_sql")


def _run(conn, sql: str, params: Iterable[Any] = ()):
    params = tuple(params)
    if _is_sqlalchemy(conn):
        result = conn.exec_driver_sql(sql, params if params else ())
        if result.returns_rows:
            return result.fetchall()
        return []
    cursor = conn.execute(sql, list(params))
    try:
        return cursor.fetchall()
    except Exception:  # a statement that returns nothing
        return []


def _run_many(conn, sql: str, rows: list[tuple]) -> None:
    if not rows:
        return
    if _is_sqlalchemy(conn):
        conn.exec_driver_sql(sql, rows)
    else:
        conn.executemany(sql, rows)


def list_relations(conn) -> list[str]:
    rows = _run(conn, "SELECT table_name FROM information_schema.tables "
                      "WHERE table_schema = current_schema() ORDER BY 1")
    return [r[0] for r in rows]


def apply_migration(conn, qualified: Callable[[str], str] = lambda n: n, dialect: str = "duckdb") -> dict:
    """Apply the additive migration; a second application creates nothing."""
    before = set(list_relations(conn))
    for statement in ddl(qualified, dialect):
        _run(conn, statement)
    after = set(list_relations(conn))
    missing = [r for r in ALL_TABLES if r not in after]
    if missing:
        raise RuntimeError(f"migration applied but relations missing: {missing}")
    return {"migration_id": MIGRATION_ID, "created": sorted(after - before),
            "already_present": sorted(before & set(ALL_RELATIONS)), "relations": list(ALL_RELATIONS)}


# --------------------------------------------------------------------------- validation
def _check_hex(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise Refusal(f"{name} must be a 64-hex SHA-256, got {value!r}")
    return value


def _finite_number(name: str, value: Any, *, required: bool) -> float | None:
    if value is None:
        if required:
            raise Refusal(f"{name} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise Refusal(f"{name} must be finite and non-negative, got {value!r}")
    return float(value)


def _non_negative_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{name} must be an integer, got {type(value).__name__}")
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise Refusal(f"{name} must be an integer, got {value!r}")
        value = int(value)
    if value < 0:
        raise Refusal(f"{name} must be non-negative, got {value!r}")
    return int(value)


def _pick(mapping: dict, *keys: str):
    """The first present key among ``keys`` in ``mapping`` (dotted keys descend)."""
    for key in keys:
        node: Any = mapping
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    return None


def prepare_terminal(document: dict) -> dict:
    """Validate one terminal document; return the storage record.

    ``document``: {"schema"?: SCHEMA_TERMINAL, "plan_sha256", "task": payload (+ "task_id"),
    "result": the COMPLETE result as the controller stored it, "owner"?, "attempt"?,
    "started_at"?, "finished_at"?, "host_role"?}. Cost and training metadata are read from the
    result's ``cost`` / ``training`` sub-documents (or their top-level fallbacks) and are
    nullable; the metrics are not.
    """
    if not isinstance(document, dict):
        raise Refusal("a terminal must be a mapping")
    if document.get("schema", SCHEMA_TERMINAL) != SCHEMA_TERMINAL:
        raise Refusal(f"terminal schema must be {SCHEMA_TERMINAL}")
    plan_sha256 = _check_hex("plan_sha256", document.get("plan_sha256"))
    task = document.get("task")
    result = document.get("result")
    if not isinstance(task, dict) or not isinstance(result, dict):
        raise Refusal("a terminal carries a task mapping and a result mapping")
    for field in TASK_FIELDS:
        if field not in task or task[field] is None or task[field] == "":
            raise Refusal(f"task.{field} is required")
    if task["schema"] != SCHEMA_TASK:
        raise Refusal(f"task.schema must be {SCHEMA_TASK}, got {task['schema']!r}")
    if task["arm"] not in ARMS:
        raise Refusal(f"task.arm must be one of {list(ARMS)}, got {task['arm']!r}")
    if type(task["seed"]) is not int or task["seed"] != 0:
        raise Refusal(f"task.seed must be the single scientific seed 0, got {task['seed']!r}")
    for field in ("population_id", "identity", "feature_id", "fold_id"):
        if not isinstance(task[field], str):
            raise Refusal(f"task.{field} must be a string")
    if task["fold_id"].lower() in ("validation", "valid", "test", "holdout"):
        raise Refusal(f"task.fold_id={task['fold_id']!r}; validation and test stay closed")
    task_id = task_id_of(task)
    declared = task.get("task_id", document.get("task_id", task_id))
    if declared != task_id:
        raise Refusal(f"task_id {str(declared)[:12]}... is not the sha256 of the canonical task payload "
                      f"({task_id[:12]}...): not a planned task")
    if result.get("status") != "COMPLETE":
        raise Refusal(f"result.status must be COMPLETE, got {result.get('status')!r}")
    if result.get("task_id") != task_id:
        raise Refusal("result.task_id does not name the task")
    if result.get("seed") != 0:
        raise Refusal(f"result.seed must be 0, got {result.get('seed')!r}")
    digests = {name: _check_hex(f"result.{name}", result.get(name)) for name in RESULT_DIGESTS}
    population_n = result.get("population_n")
    if type(population_n) is not int or population_n < 1:
        raise Refusal(f"result.population_n must be a positive integer, got {population_n!r}")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise Refusal("result.metrics must be a mapping with mae and naive_mae")
    mae = _finite_number("result.metrics.mae", metrics.get("mae"), required=True)
    naive_mae = _finite_number("result.metrics.naive_mae", metrics.get("naive_mae"), required=True)
    for name, value in metrics.items():
        _finite_number(f"result.metrics.{name}", value, required=True)
    mse = _finite_number("result.metrics.mse", metrics.get("mse"), required=False)
    host_role = document.get("host_role", _pick(result, "host_role"))
    if host_role is not None and host_role not in HOST_ROLES:
        raise Refusal(f"host_role must be one of {list(HOST_ROLES)}; no host names")
    extracted = {
        "chosen_epoch": _non_negative_int("chosen_epoch", _pick(result, "training.chosen_epoch", "chosen_epoch", "training.best_epoch")),
        "updates": _non_negative_int("updates", _pick(result, "training.updates", "updates", "training.update_count")),
        "cpu_s": _finite_number("cost.cpu_s", _pick(result, "cost.cpu_s", "cpu_s"), required=False),
        "wall_s": _finite_number("cost.wall_s", _pick(result, "cost.wall_s", "wall_s"), required=False),
        "peak_ram_bytes": _non_negative_int("cost.peak_ram_bytes", _pick(result, "cost.peak_ram_bytes", "cost.peak_ram", "peak_ram_bytes")),
        "peak_vram_bytes": _non_negative_int("cost.peak_vram_bytes", _pick(result, "cost.peak_vram_bytes", "cost.peak_vram", "peak_vram_bytes")),
    }
    try:
        result_json = canonical_bytes(result).decode("ascii")
        task_json = canonical_bytes({k: task[k] for k in TASK_FIELDS}).decode("ascii")
    except ValueError as exc:
        raise Refusal(f"terminal is not canonical JSON: {exc}") from None
    record = {
        "task_id": task_id, "plan_sha256": plan_sha256,
        "population_id": task["population_id"], "identity": task["identity"],
        "feature_id": task["feature_id"], "fold_id": task["fold_id"], "arm": task["arm"], "seed": 0,
        **digests, "population_n": population_n, "mae": mae, "naive_mae": naive_mae, "mse": mse,
        **extracted,
        "host_role": host_role, "owner": document.get("owner"),
        "attempt": _non_negative_int("attempt", document.get("attempt")),
        "started_at": _finite_number("started_at", document.get("started_at"), required=False),
        "finished_at": _finite_number("finished_at", document.get("finished_at"), required=False),
        "terminal_sha256": hashlib.sha256(result_json.encode("ascii")).hexdigest(),
        "task_json": task_json, "result_json": result_json,
    }
    return record


def _shared_signature(record: dict) -> tuple:
    return tuple(record[k] for k in SHARED_ACROSS_ARMS) + (record["naive_mae"],)


# --------------------------------------------------------------------------- submit
def submit_terminals(conn, plan_sha256: str, terminals: list[dict], *, host_role: str | None = None,
                     manage_transaction: bool = True, backend: str = "duckdb",
                     qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Store terminals for ``plan_sha256`` and return the receipt.

    Refuses before writing: empty batch, a terminal of another plan, a malformed terminal, one
    task_id with two contents in the batch, a task already stored with a different terminal
    digest, an arm whose rows/mask/input/population_n/naive_mae differ from a stored or batched
    peer of the same (population, identity, feature, fold). Writes are one transaction.
    """
    t = qualified
    plan_sha256 = _check_hex("plan_sha256", plan_sha256)
    if not terminals:
        raise Refusal("submit_terminals received no terminals")
    if host_role is not None and host_role not in HOST_ROLES:
        raise Refusal(f"host_role must be one of {list(HOST_ROLES)}; no host names")
    records: dict[str, dict] = {}
    for raw in terminals:
        if isinstance(raw, dict) and raw.get("plan_sha256") not in (None, plan_sha256):
            raise Refusal(f"foreign plan identity: terminal carries plan {str(raw.get('plan_sha256'))[:12]}..., "
                          f"submission is for {plan_sha256[:12]}...")
        rec = prepare_terminal({**raw, "plan_sha256": plan_sha256} if isinstance(raw, dict) else raw)
        if host_role is not None and rec["host_role"] is None:
            rec["host_role"] = host_role
        prior = records.get(rec["task_id"])
        if prior is not None and prior["terminal_sha256"] != rec["terminal_sha256"]:
            raise Refusal(f"the batch carries task {rec['task_id'][:12]}... with two different terminals")
        records[rec["task_id"]] = rec
    # FS4-09 inside the batch
    by_triple: dict[tuple, tuple] = {}
    for rec in records.values():
        key = (rec["population_id"], rec["identity"], rec["feature_id"], rec["fold_id"])
        sig = _shared_signature(rec)
        if by_triple.setdefault(key, sig) != sig:
            raise Refusal(f"PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH in the batch for {key[0]}/{key[2]}/{key[3]}")
    ids = list(records)
    stored: dict[str, str] = {}
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i + 1000]
        marks = ",".join("?" * len(chunk))
        for task_id, sha in _run(conn, f"SELECT task_id, terminal_sha256 FROM {t(TABLE)}"
                                       f" WHERE task_id IN ({marks})", chunk):
            stored[task_id] = sha
    conflicts = [x for x, sha in stored.items() if records[x]["terminal_sha256"] != sha]
    if conflicts:
        raise Refusal(f"{len(conflicts)} tasks already stored with a different terminal; "
                      f"first {conflicts[0][:12]}...; nothing written")
    new = [records[x] for x in ids if x not in stored]
    # FS4-09 against what is stored
    shared_cols = ", ".join(SHARED_ACROSS_ARMS)
    for rec in new:
        peers = _run(conn, f"SELECT task_id, {shared_cols}, naive_mae FROM {t(TABLE)} WHERE plan_sha256 = ?"
                           " AND population_id = ? AND identity = ? AND feature_id = ? AND fold_id = ?",
                     [plan_sha256, rec["population_id"], rec["identity"], rec["feature_id"], rec["fold_id"]])
        for peer in peers:
            if tuple(peer[1:]) != _shared_signature(rec):
                raise Refusal(f"PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH: task {rec['task_id'][:12]}... "
                              f"({rec['arm']}) disagrees with stored peer {peer[0][:12]}... for "
                              f"{rec['population_id']}/{rec['feature_id']}/{rec['fold_id']}; nothing written")
    receipt = {
        "schema": SCHEMA_RECEIPT, "backend": backend, "plan_sha256": plan_sha256,
        "terminal_count": len(terminals), "distinct_tasks": len(records),
        "inserted": len(new), "duplicates_ignored": len(records) - len(new),
        "terminals_sha256": terminals_digest(r["terminal_sha256"] for r in records.values()),
        "task_ids": sorted(records), "host_role": host_role, "submitted_at": now(),
    }
    receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    managed = manage_transaction and not _is_sqlalchemy(conn)
    if managed:
        _run(conn, "BEGIN TRANSACTION")
    try:
        _run_many(conn, f"INSERT INTO {t(TABLE)} ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                  [tuple(r[c] for c in COLUMNS) for r in new])
        _run(conn, f"INSERT INTO {t(RECEIPT_TABLE)} (receipt_sha256, plan_sha256, terminal_count, inserted,"
                   " duplicates_ignored, terminals_sha256, host_role, submitted_at)"
                   " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             [receipt["receipt_sha256"], plan_sha256, receipt["terminal_count"], receipt["inserted"],
              receipt["duplicates_ignored"], receipt["terminals_sha256"], host_role, receipt["submitted_at"]])
        _run_many(conn, f"INSERT INTO {t(RECEIPT_TASK_TABLE)} (receipt_sha256, task_id, terminal_sha256) VALUES (?, ?, ?)",
                  [(receipt["receipt_sha256"], x, records[x]["terminal_sha256"]) for x in sorted(records)])
        if managed:
            _run(conn, "COMMIT")
    except Exception:
        if managed:
            _run(conn, "ROLLBACK")
        raise
    return receipt


def verify_receipt(conn, receipt: dict, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Accept a receipt only when its digest, plan and stored digest agree."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA_RECEIPT:
        raise Refusal("not a phase-4 warehouse receipt")
    if digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != receipt.get("receipt_sha256"):
        raise Refusal("receipt digest does not match its body")
    rows = _run(conn, f"SELECT plan_sha256, terminals_sha256 FROM {qualified(RECEIPT_TABLE)} WHERE receipt_sha256 = ?",
                [receipt["receipt_sha256"]])
    if not rows:
        raise Refusal("the store holds no such receipt")
    if rows[0][0] != receipt.get("plan_sha256"):
        raise Refusal("receipt carries a plan identity the store does not hold for it")
    if rows[0][1] != receipt.get("terminals_sha256"):
        raise Refusal("stored receipt digest differs from the presented one")
    return {"accepted": True, "receipt_sha256": receipt["receipt_sha256"]}


# --------------------------------------------------------------------------- readback
def _document_of(row: tuple) -> dict:
    task_id, plan_sha256, task_json, result_json, owner, attempt, started_at, finished_at, host_role, terminal_sha256 = row
    task = json.loads(task_json)
    return {"schema": SCHEMA_TERMINAL, "plan_sha256": plan_sha256, "task": {**task, "task_id": task_id},
            "result": json.loads(result_json), "owner": owner, "attempt": attempt,
            "started_at": started_at, "finished_at": finished_at, "host_role": host_role,
            "terminal_sha256": terminal_sha256}


def read_terminals(conn, plan_sha256: str, *, task_id: str | None = None, population_id: str | None = None,
                   feature_id: str | None = None, fold_id: str | None = None, arm: str | None = None,
                   after: str | None = None, limit: int | None = None,
                   qualified: Callable[[str], str] = lambda n: n) -> list[dict]:
    """Terminals exactly as submitted (task + result), ordered by task_id; optional filters/paging."""
    plan_sha256 = _check_hex("plan_sha256", plan_sha256)
    sql = ("SELECT task_id, plan_sha256, task_json, result_json, owner, attempt, started_at, finished_at,"
           f" host_role, terminal_sha256 FROM {qualified(TABLE)} WHERE plan_sha256 = ?")
    params: list[Any] = [plan_sha256]
    for column, value in (("task_id", task_id), ("population_id", population_id), ("feature_id", feature_id),
                          ("fold_id", fold_id), ("arm", arm)):
        if value is not None:
            sql += f" AND {column} = ?"
            params.append(value)
    if after is not None:
        sql += " AND task_id > ?"
        params.append(after)
    sql += " ORDER BY task_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [_document_of(tuple(r)) for r in _run(conn, sql, params)]


def stored_summary_sql(plan_sha256: str, qualified: Callable[[str], str] = lambda n: n) -> str:
    pid = plan_sha256.replace("'", "''")
    return (f"SELECT count(*) AS n, coalesce(sha256(string_agg(terminal_sha256, '' ORDER BY terminal_sha256)),"
            f" '{EMPTY_DIGEST}') AS terminals_sha256 FROM {qualified(TABLE)} WHERE plan_sha256 = '{pid}' LIMIT 1")


def reconcile(conn, plan_sha256: str, expected: dict | None = None, receipts: list[dict] | None = None,
              *, backend: str = "duckdb", qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Stored counts and digests for one plan against the controller's expected counts.

    ``expected``: {"total": int, "by_population": {population: int}} from the controller's task
    store. The report says whether every stored arm-triple agrees (FS4-09), whether every stored
    task is covered by a receipt, and whether counts meet the expectation; ``complete`` is the
    conjunction.
    """
    t = qualified
    plan_sha256 = _check_hex("plan_sha256", plan_sha256)
    expected = dict(expected or {})
    n, sha = _run(conn, stored_summary_sql(plan_sha256, t))[0]
    by_population = {r[0]: int(r[1]) for r in _run(
        conn, f"SELECT population_id, count(*) FROM {t(TABLE)} WHERE plan_sha256 = ? GROUP BY 1 ORDER BY 1", [plan_sha256])}
    by_arm = {r[0]: int(r[1]) for r in _run(
        conn, f"SELECT arm, count(*) FROM {t(TABLE)} WHERE plan_sha256 = ? GROUP BY 1 ORDER BY 1", [plan_sha256])}
    by_population_arm = {}
    for pop, arm, k in _run(conn, f"SELECT population_id, arm, count(*) FROM {t(TABLE)} WHERE plan_sha256 = ?"
                                  " GROUP BY 1, 2 ORDER BY 1, 2", [plan_sha256]):
        by_population_arm.setdefault(pop, {})[arm] = int(k)
    triples = _run(conn, "SELECT population_id, identity, feature_id, fold_id, arms, rows_digests, mask_digests,"
                         f" input_digests, population_ns, naive_maes FROM {t('fs4_extractibility_triples')}"
                         " WHERE plan_sha256 = ? ORDER BY 1, 2, 3, 4", [plan_sha256])
    inconsistent, partial, full = [], 0, 0
    for pop, ident, feat, fold, arms, rd, md, idg, pn, nm in triples:
        if max(rd, md, idg, pn, nm) > 1:
            inconsistent.append({"population_id": pop, "identity": ident, "feature_id": feat, "fold_id": fold,
                                 "arms": int(arms), "rows_digests": int(rd), "mask_digests": int(md),
                                 "input_digests": int(idg), "population_ns": int(pn), "naive_maes": int(nm)})
        if int(arms) == len(ARMS):
            full += 1
        else:
            partial += 1
    without_receipt = _run(conn, f"SELECT count(*) FROM {t(TABLE)} x WHERE x.plan_sha256 = ? AND NOT EXISTS"
                                 f" (SELECT 1 FROM {t(RECEIPT_TASK_TABLE)} r WHERE r.task_id = x.task_id"
                                 " AND r.terminal_sha256 = x.terminal_sha256)", [plan_sha256])[0][0]
    issued = _run(conn, f"SELECT count(*), coalesce(sum(inserted), 0) FROM {t(RECEIPT_TABLE)} WHERE plan_sha256 = ?",
                  [plan_sha256])[0]
    verified = [verify_receipt(conn, r, t) for r in (receipts or [])]
    counts_match = None
    if expected.get("total") is not None:
        counts_match = int(n) == int(expected["total"])
        for pop, k in (expected.get("by_population") or {}).items():
            counts_match = counts_match and by_population.get(pop, 0) == int(k)
    report = {"schema": SCHEMA_RECONCILIATION, "backend": backend, "plan_sha256": plan_sha256,
              "stored": {"total": int(n), "terminals_sha256": sha, "by_population": by_population,
                         "by_arm": by_arm, "by_population_arm": by_population_arm},
              "expected": expected or None, "count_matches_expected": counts_match,
              "triples": {"complete": full, "partial": partial, "inconsistent": len(inconsistent),
                          "inconsistent_examples": inconsistent[:20]},
              "receipts": {"issued": int(issued[0]), "receipted_inserted": int(issued[1]),
                           "tasks_without_receipt": int(without_receipt),
                           "receipts_cover_store": int(without_receipt) == 0},
              "receipts_verified": len(verified),
              "complete": bool(counts_match) and not inconsistent and int(without_receipt) == 0 and partial == 0,
              "reconciled_at": now()}
    report["reconciliation_sha256"] = digest({k: v for k, v in report.items() if k != "reconciled_at"})
    return report


# --------------------------------------------------------------------------- host documents
def write_document(conn, document: dict, *, backend: str, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """The host write route body: {"plan_sha256", "terminals": [...], "host_role"?}."""
    if not isinstance(document, dict):
        raise Refusal("the request body must be a JSON object")
    terminals = document.get("terminals")
    if not isinstance(terminals, list):
        raise Refusal("terminals must be a list")
    return submit_terminals(conn, document.get("plan_sha256"), terminals, host_role=document.get("host_role"),
                            manage_transaction=False, backend=backend, qualified=qualified)


def read_document(conn, document: dict, *, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """The host read route: {"plan_sha256", "task_id"?, "population_id"?, "feature_id"?, "fold_id"?,
    "arm"?, "after"?, "limit"?} -> terminals + cursor."""
    if not isinstance(document, dict):
        raise Refusal("the request must be a JSON object")
    limit = document.get("limit")
    limit = 1000 if limit is None else int(limit)
    if limit < 1 or limit > 5000:
        raise Refusal("limit must be between 1 and 5000")
    if document.get("arm") is not None and document["arm"] not in ARMS:
        raise Refusal(f"arm must be one of {list(ARMS)}")
    rows = read_terminals(conn, document.get("plan_sha256"), task_id=document.get("task_id"),
                          population_id=document.get("population_id"), feature_id=document.get("feature_id"),
                          fold_id=document.get("fold_id"), arm=document.get("arm"),
                          after=document.get("after"), limit=limit, qualified=qualified)
    next_after = rows[-1]["task"]["task_id"] if len(rows) == limit else None
    return {"terminals": rows, "count": len(rows), "next_after": next_after}


def reconcile_document(conn, document: dict, *, backend: str, qualified: Callable[[str], str] = lambda n: n) -> dict:
    if not isinstance(document, dict) or not document.get("plan_sha256"):
        raise Refusal("the request must carry plan_sha256")
    return reconcile(conn, document["plan_sha256"], document.get("expected"), document.get("receipts"),
                     backend=backend, qualified=qualified)
