"""Phase-2/3 feature-selection storage: additive DDL, submit, readback, reconciliation.

One implementation serves three callers so they cannot disagree about identity or digests:

* ``tools/fs_phase23_warehouse.py`` on a local DuckDB file (throwaway tests, local terminal
  store, snapshot copies);
* the packaged backend (``predictor_olap_store.query.Plugin`` and the DuckDB provider) inside
  the warehouse service, through the host's ``/api/v2/fs-phase23/*`` routes;
* ``tools/olap_duckdb_migrate.py``, whose snapshot boundary names the same relations.

Interface contract (what the driver ``tools/feature_pairwise_campaign.py`` relies on):

    submit_rows(conn, run_id, table, rows, ...) -> receipt
        receipt: run_id, table, row_count, rows_sha256, receipt_sha256, inserted,
                 duplicates_ignored, backend, unit_id, population_id, run_sha256, ...
    read_run(conn, run_id, table, unit_id=None) -> [row dict exactly as submitted]
    reconcile(conn, run_id) -> {"run_id", "tables": {table: {"count", "rows_sha256", ...}}}

Digest rule (adopted from the driver's adapter so receipts, readback and reconciliation
share one arithmetic): ``row_sha256 = sha256(canonical_json(row))`` of the row **as
submitted** (sorted keys, compact separators, ascii, no NaN) and
``rows_sha256 = sha256(''.join(sorted(row_sha256)))``. The SQL twin is
``sha256(string_agg(row_sha256, '' ORDER BY row_sha256))`` with the empty-set digest for
an empty relation.

Identity: every table has a UNIQUE ``(run_id, row_key)`` — the driver's key — AND a
``row_identity_sha256`` PRIMARY KEY computed from the typed identity columns of subplan §2
extracted from the row (ordered pair, population, fold, lag, metric/method, parameter and
code digests, target/horizon, k, group/cluster id). Two rows cannot share either key. The
same identity with different content is refused; an identical replay is a no-op.

Rows carry a foreign identity when their ``run_id`` differs from the submission's or their
``population_id`` differs from the population the run is bound to; both are refused before
any write. A run is bound on first submission (the driver registers nothing) or explicitly
through ``register_run`` with the contract digests.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

SCHEMA_RECEIPT = "fs_phase23.warehouse_receipt.v2"
SCHEMA_RECONCILIATION = "fs_phase23.warehouse_reconcile.v2"
MIGRATION_ID = "fs_phase23_0001"

RUN_TABLE = "fs_phase23_run"
RECEIPT_TABLE = "fs_phase23_load_receipt"
FACT_TABLES = (
    "feature_pair_metrics", "feature_pair_stability", "feature_pair_gate",
    "feature_alias_groups", "feature_redundancy_clusters",
    "feature_filter_rankings", "feature_filter_subsets",
)
VIEWS = ("fs_phase23_coverage",)
ALL_TABLES = (RUN_TABLE, RECEIPT_TABLE, *FACT_TABLES)
ALL_RELATIONS = ALL_TABLES + VIEWS

STATES = ("MEASURED", "INSUFFICIENT_SUPPORT", "NOT_APPLICABLE", "FAILED")
HOST_ROLES = ("coordinator", "worker_a", "worker_b")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()

#: Typed identity columns per table -> the row field they are read from (subplan §2).
IDENTITY: dict[str, dict[str, str]] = {
    "feature_pair_metrics": {
        "run_id": "run_id", "population_id": "population_id", "feature_left": "left",
        "feature_right": "right", "fold": "fold_id", "metric": "metric", "lag": "lag_hours",
        "method": "method", "params_sha256": "params_sha256", "code_sha256": "code_sha256"},
    "feature_pair_stability": {
        "run_id": "run_id", "population_id": "population_id", "feature_left": "left",
        "feature_right": "right", "metric": "metric", "lag": "lag_hours", "method": "method",
        "params_sha256": "params_sha256", "code_sha256": "code_sha256"},
    "feature_pair_gate": {
        "run_id": "run_id", "population_id": "population_id", "feature_left": "left",
        "feature_right": "right", "fold": "fold_id", "method": "method",
        "params_sha256": "params_sha256", "code_sha256": "code_sha256"},
    "feature_alias_groups": {
        "run_id": "run_id", "population_id": "population_id", "group_id": "alias_group_id"},
    "feature_redundancy_clusters": {
        "run_id": "run_id", "population_id": "population_id", "cluster_id": "cluster_id"},
    "feature_filter_rankings": {
        "run_id": "run_id", "population_id": "population_id", "target_id": "target_id",
        "horizon": "horizon_hours", "method": "method", "params_sha256": "params_sha256",
        "feature_id": "feature_id"},
    "feature_filter_subsets": {
        "run_id": "run_id", "population_id": "population_id", "target_id": "target_id",
        "horizon": "horizon_hours", "method": "method", "params_sha256": "params_sha256",
        "k": "k"},
}

#: Queryable value columns per table -> row field (nullable; the payload is authoritative).
VALUES: dict[str, dict[str, str]] = {
    "feature_pair_metrics": {"metric_value": "value", "shared_support": "support_n", "state": "state"},
    "feature_pair_stability": {"metric_value": "mean", "valid_folds": "valid_folds", "state": "state"},
    "feature_pair_gate": {"gate_state": "gate_state", "shared_support": "shared_support"},
    "feature_alias_groups": {"representative": "representative", "disposition": "disposition",
                             "members_json": "members"},
    "feature_redundancy_clusters": {"representative": "representative", "size": "size",
                                    "members_json": "members"},
    "feature_filter_rankings": {"rank": "rank", "score": "score", "causal_label": "causal_label"},
    "feature_filter_subsets": {"subset_sha256": "subset_sha256", "label": "label",
                               "members_json": "members", "is_final_selection": "is_final_selection"},
}

_TYPES = {
    "run_id": "TEXT NOT NULL", "population_id": "TEXT NOT NULL", "feature_left": "TEXT NOT NULL",
    "feature_right": "TEXT NOT NULL", "fold": "TEXT NOT NULL", "metric": "TEXT NOT NULL",
    "lag": "INTEGER NOT NULL", "method": "TEXT NOT NULL", "params_sha256": "TEXT NOT NULL",
    "code_sha256": "TEXT NOT NULL", "group_id": "TEXT NOT NULL", "cluster_id": "TEXT NOT NULL",
    "target_id": "TEXT NOT NULL", "horizon": "INTEGER NOT NULL", "feature_id": "TEXT NOT NULL",
    "k": "INTEGER NOT NULL",
    "metric_value": "DOUBLE", "shared_support": "BIGINT", "state": "TEXT", "valid_folds": "INTEGER",
    "gate_state": "TEXT", "representative": "TEXT", "disposition": "TEXT", "members_json": "TEXT",
    "size": "INTEGER", "rank": "INTEGER", "score": "DOUBLE", "causal_label": "TEXT",
    "subset_sha256": "TEXT", "label": "TEXT", "is_final_selection": "BOOLEAN",
}
_COMMON_HEAD = ("row_identity_sha256", "row_sha256", "row_key", "unit_id")
_COMMON_TAIL = ("host_role", "shard_id", "payload")


class Refusal(ValueError):
    """A typed refusal: the request is wrong, not the store. Answers 400 at the host."""


# --------------------------------------------------------------------------- canonical bytes
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def row_sha256(row: dict) -> str:
    return hashlib.sha256(canonical_bytes(row)).hexdigest()


def rows_digest(row_digests: Iterable[str]) -> str:
    """sha256 of the sorted concatenation of per-row digests; EMPTY_DIGEST for no rows."""
    return hashlib.sha256("".join(sorted(row_digests)).encode("ascii")).hexdigest()


def rows_sha256(rows: Iterable[dict]) -> str:
    return rows_digest(row_sha256(r) for r in rows)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pair_denominator(n: int) -> int:
    if n < 2:
        raise Refusal(f"a population of {n} features has no pairs")
    return n * (n - 1) // 2


def order_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise Refusal(f"{a!r} paired with itself is not a pair")
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------- DDL
def columns(table: str) -> tuple[str, ...]:
    return (*_COMMON_HEAD, *IDENTITY[table], *VALUES[table], *_COMMON_TAIL)


def ddl(qualified: Callable[[str], str] = lambda n: n, dialect: str = "duckdb") -> list[str]:
    """Additive DDL: CREATE TABLE/VIEW IF NOT EXISTS only. Nothing dropped or altered."""
    t = qualified
    create_view = "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
    statements = [
        f"CREATE TABLE IF NOT EXISTS {t(RUN_TABLE)} ("
        " run_id TEXT PRIMARY KEY, run_sha256 TEXT NOT NULL UNIQUE, population_id TEXT NOT NULL,"
        " phase TEXT NOT NULL CHECK (phase IN ('PHASE_2', 'PHASE_3', 'PHASE_2_3')),"
        " registration TEXT NOT NULL CHECK (registration IN ('CONTRACT_BOUND', 'FIRST_SUBMISSION')),"
        " contract_sha256 TEXT, campaign_sha256 TEXT, code_sha256 TEXT, input_sha256 TEXT,"
        " expected_json TEXT NOT NULL, created_at TEXT NOT NULL)",
        f"CREATE TABLE IF NOT EXISTS {t(RECEIPT_TABLE)} ("
        " receipt_sha256 TEXT PRIMARY KEY,"
        " run_id TEXT NOT NULL, population_id TEXT NOT NULL,"
        " table_name TEXT NOT NULL, unit_id TEXT, row_count INTEGER NOT NULL,"
        " distinct_identities INTEGER NOT NULL, inserted INTEGER NOT NULL,"
        " duplicates_ignored INTEGER NOT NULL, rows_sha256 TEXT NOT NULL, host_role TEXT,"
        " shard_id TEXT, submitted_at TEXT NOT NULL)",
    ]
    for table in FACT_TABLES:
        parts = [
            " row_identity_sha256 TEXT PRIMARY KEY", " row_sha256 TEXT NOT NULL",
            " row_key TEXT NOT NULL", " unit_id TEXT"]
        for column in IDENTITY[table]:
            parts.append(f" {column} {_TYPES[column]}")
        for column in VALUES[table]:
            parts.append(f" {column} {_TYPES[column]}")
        parts += [" host_role TEXT", " shard_id TEXT", " payload TEXT NOT NULL",
                  " stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"]
        if "feature_left" in IDENTITY[table]:
            parts.append(" CHECK (feature_left < feature_right)")
        if "lag" in IDENTITY[table]:
            parts.append(" CHECK (lag >= 0)")
        if "state" in VALUES[table]:
            parts.append(" CHECK (state IS NULL OR state IN ('MEASURED', 'INSUFFICIENT_SUPPORT',"
                         " 'NOT_APPLICABLE', 'FAILED'))")
        parts.append(" UNIQUE (run_id, row_key)")
        parts.append(f" UNIQUE ({', '.join(IDENTITY[table])})")
        statements.append(f"CREATE TABLE IF NOT EXISTS {t(table)} ({','.join(parts)})")
    counts = ", ".join(
        f"(SELECT count(*) FROM {t(table)} x WHERE x.run_id = r.run_id) AS {table}_rows"
        for table in FACT_TABLES)
    statements.append(
        f"{create_view} {t('fs_phase23_coverage')} AS SELECT r.run_id, r.population_id,"
        f" r.phase, r.registration, {counts},"
        f" (SELECT count(*) FROM {t(RECEIPT_TABLE)} l WHERE l.run_id = r.run_id) AS receipts"
        f" FROM {t(RUN_TABLE)} r")
    return statements


def render_sql() -> str:
    """The migration file content: the same statements, one per line group, for humans."""
    head = (
        f"-- {MIGRATION_ID}: additive schema for feature-selection phases 2 and 3.\n"
        "-- GENERATED from predictor_olap_store.fs_phase23_store.ddl(); do not edit by hand.\n"
        "-- Every statement is CREATE ... IF NOT EXISTS. Nothing is dropped, altered or rewritten.\n"
        "-- Identity: UNIQUE (run_id, row_key) AND UNIQUE over the typed identity columns;\n"
        "-- row_identity_sha256 (PRIMARY KEY) digests those columns; row_sha256 digests the\n"
        "-- submitted row (canonical JSON); payload holds that JSON; stored_at is never digested.\n\n")
    return head + ";\n\n".join(ddl()) + ";\n"


# --------------------------------------------------------------------------- execution shim
def _is_sqlalchemy(conn) -> bool:
    return hasattr(conn, "exec_driver_sql")


def _run(conn, sql: str, params: Iterable[Any] = ()):
    """Execute with qmark parameters on a raw DuckDB connection or a SQLAlchemy connection."""
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


def list_relations(conn) -> list[str]:
    rows = _run(conn, "SELECT table_name FROM information_schema.tables "
                      "WHERE table_schema = current_schema() ORDER BY 1")
    return [r[0] for r in rows]


# --------------------------------------------------------------------------- runs
def _check_hex(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise Refusal(f"{name} must be a 64-hex SHA-256, got {value!r}")
    return value


def _run_body(run: dict) -> dict:
    return {k: run.get(k) for k in ("run_id", "population_id", "phase", "registration",
                                    "contract_sha256", "campaign_sha256", "code_sha256",
                                    "input_sha256", "expected_json")}


def register_run(conn, run: dict, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Bind a run to a population (and, when given, to the contract digests).

    Re-registering an identical run is a no-op; a different document under the same run_id is
    refused. A run bound by FIRST_SUBMISSION may later be upgraded to CONTRACT_BOUND only with
    the same population_id.
    """
    for key in ("run_id", "population_id"):
        if not run.get(key):
            raise Refusal(f"run.{key} is required")
    phase = run.get("phase") or "PHASE_2_3"
    if phase not in ("PHASE_2", "PHASE_3", "PHASE_2_3"):
        raise Refusal("run.phase must be PHASE_2, PHASE_3 or PHASE_2_3")
    registration = run.get("registration") or "CONTRACT_BOUND"
    if registration == "CONTRACT_BOUND":
        for key in ("contract_sha256", "campaign_sha256", "code_sha256", "input_sha256"):
            _check_hex(f"run.{key}", run.get(key))
    expected = run.get("expected_json") or {}
    if not isinstance(expected, dict):
        raise Refusal("run.expected_json must map table -> expected row count")
    body = dict(_run_body(run), phase=phase, registration=registration,
                expected_json=canonical_bytes(expected).decode("ascii"))
    run_sha = digest(body)
    t = qualified
    stored = _run(conn, f"SELECT run_sha256, population_id, registration FROM {t(RUN_TABLE)} "
                        "WHERE run_id = ?", [run["run_id"]])
    if stored:
        if stored[0][0] == run_sha:
            return {"run_id": run["run_id"], "run_sha256": run_sha, "already_registered": True}
        if stored[0][2] == "FIRST_SUBMISSION" and registration == "CONTRACT_BOUND" \
                and stored[0][1] == run["population_id"]:
            _run(conn, f"UPDATE {t(RUN_TABLE)} SET run_sha256 = ?, phase = ?, registration = ?,"
                       " contract_sha256 = ?, campaign_sha256 = ?, code_sha256 = ?, input_sha256 = ?,"
                       " expected_json = ? WHERE run_id = ?",
                 [run_sha, phase, registration, body["contract_sha256"], body["campaign_sha256"],
                  body["code_sha256"], body["input_sha256"], body["expected_json"], run["run_id"]])
            return {"run_id": run["run_id"], "run_sha256": run_sha, "already_registered": False,
                    "upgraded": True}
        raise Refusal(f"run {run['run_id']!r} is already registered with a different identity "
                      f"({stored[0][0][:12]}... vs {run_sha[:12]}...)")
    _run(conn, f"INSERT INTO {t(RUN_TABLE)} (run_id, run_sha256, population_id, phase, registration,"
               " contract_sha256, campaign_sha256, code_sha256, input_sha256, expected_json, created_at)"
               " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
         [run["run_id"], run_sha, run["population_id"], phase, registration,
          body["contract_sha256"], body["campaign_sha256"], body["code_sha256"],
          body["input_sha256"], body["expected_json"], now()])
    return {"run_id": run["run_id"], "run_sha256": run_sha, "already_registered": False}


def get_run(conn, run_id: str, qualified: Callable[[str], str] = lambda n: n) -> dict | None:
    keys = ("run_id", "run_sha256", "population_id", "phase", "registration", "contract_sha256",
            "campaign_sha256", "code_sha256", "input_sha256", "expected_json", "created_at")
    rows = _run(conn, f"SELECT {', '.join(keys)} FROM {qualified(RUN_TABLE)} WHERE run_id = ?", [run_id])
    if not rows:
        return None
    out = dict(zip(keys, rows[0]))
    out["expected_json"] = json.loads(out["expected_json"] or "{}")
    return out


# --------------------------------------------------------------------------- rows
def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else canonical_bytes(value).decode("ascii")


def prepare_row(table: str, row: dict) -> dict:
    """Validate one submitted row; return the storage record (identity, digests, payload)."""
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}; expected one of {FACT_TABLES}")
    if not isinstance(row, dict):
        raise Refusal(f"{table}: a row must be a mapping")
    key = row.get("row_key")
    if not isinstance(key, str) or not key:
        raise Refusal(f"{table}: every row needs a row_key")
    identity = {}
    for column, field in IDENTITY[table].items():
        value = row.get(field)
        if value is None or value == "":
            raise Refusal(f"{table}: identity field {field!r} is required")
        if column in ("params_sha256", "code_sha256"):
            _check_hex(f"{table}.{field}", value)
        if column in ("lag", "horizon", "k"):
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise Refusal(f"{table}: {field} must be an integer") from None
            if column == "lag" and value < 0:
                raise Refusal(f"{table}: lag_hours must be non-negative")
        identity[column] = value
    if "feature_left" in identity and not identity["feature_left"] < identity["feature_right"]:
        raise Refusal(f"{table}: pair must be ordered left < right, got "
                      f"({identity['feature_left']!r}, {identity['feature_right']!r})")
    for field in ("split", "fold_id"):
        value = row.get(field)
        if isinstance(value, str) and value.lower() in ("validation", "valid", "test", "holdout"):
            raise Refusal(f"{table}: {field}={value!r}; validation and test stay closed")
    if row.get("host_role") is not None and row["host_role"] not in HOST_ROLES:
        raise Refusal(f"{table}: host_role must be one of {list(HOST_ROLES)}; no host names")
    values = {}
    for column, field in VALUES[table].items():
        value = row.get(field)
        if column == "state" and value is not None and value not in STATES:
            raise Refusal(f"{table}: state {value!r} not in {STATES}")
        if column == "members_json":
            value = _json_or_none(value)
        elif column in ("rank", "size", "valid_folds") and value is not None:
            value = int(value)
        elif column == "shared_support" and value is not None:
            value = int(value)
        elif column in ("metric_value", "score") and value is not None:
            value = float(value)
        elif column == "is_final_selection" and value is not None:
            value = bool(value)
        values[column] = value
    payload = canonical_bytes(row).decode("ascii")
    record = {"row_identity_sha256": digest(identity),
              "row_sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
              "row_key": key, "unit_id": row.get("unit_id"), **identity, **values,
              "host_role": row.get("host_role"), "shard_id": row.get("shard_id"), "payload": payload}
    return record


def submit_rows(conn, run_id: str, table: str, rows: list[dict], *, host_role: str | None = None,
                shard_id: str | None = None, manage_transaction: bool = True,
                backend: str = "duckdb", qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Store rows for ``run_id`` in ``table`` and return the receipt.

    Refuses before writing: unknown table, empty batch, unknown run when the batch does not
    bind one, foreign run_id or population_id, malformed rows, one identity with two contents
    in the batch, an identity already stored with different content. Writes are one
    transaction (managed here on a raw connection, by the caller on SQLAlchemy).
    """
    t = qualified
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}")
    if not run_id:
        raise Refusal("run_id is required")
    if not rows:
        raise Refusal("submit_rows received no rows")
    records: dict[str, dict] = {}
    keys: dict[str, str] = {}
    populations = set()
    unit_ids = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise Refusal(f"{table}: a row must be a mapping")
        if raw.get("run_id") != run_id:
            raise Refusal(f"foreign run identity: row carries run_id {raw.get('run_id')!r}, "
                          f"submission is for {run_id!r}")
        rec = prepare_row(table, raw)
        if host_role is not None and rec["host_role"] is None:
            rec["host_role"] = host_role
        if shard_id is not None and rec["shard_id"] is None:
            rec["shard_id"] = shard_id
        populations.add(rec["population_id"])
        unit_ids.add(rec["unit_id"])
        prior = records.get(rec["row_identity_sha256"])
        if prior is not None and prior["row_sha256"] != rec["row_sha256"]:
            raise Refusal("the batch carries one identity with two different contents: "
                          f"{rec['row_identity_sha256'][:12]}...")
        other = keys.get(rec["row_key"])
        if other is not None and other != rec["row_identity_sha256"]:
            raise Refusal(f"the batch carries row_key {rec['row_key'][:12]}... under two identities")
        keys[rec["row_key"]] = rec["row_identity_sha256"]
        records[rec["row_identity_sha256"]] = rec
    if len(populations) != 1:
        raise Refusal(f"a submission binds one population; the batch carries {sorted(populations)}")
    population_id = populations.pop()
    run = get_run(conn, run_id, t)
    if run is None:
        register_run(conn, {"run_id": run_id, "population_id": population_id,
                            "phase": "PHASE_2_3", "registration": "FIRST_SUBMISSION"}, t)
        run = get_run(conn, run_id, t)
    if run["population_id"] != population_id:
        raise Refusal(f"foreign population identity: rows carry {population_id!r}, run "
                      f"{run_id!r} is bound to {run['population_id']!r}")
    stored_by_identity: dict[str, str] = {}
    stored_by_key: dict[str, str] = {}
    idents = list(records)
    for i in range(0, len(idents), 1000):
        chunk = idents[i:i + 1000]
        marks = ",".join("?" * len(chunk))
        for ident, sha, key in _run(conn, f"SELECT row_identity_sha256, row_sha256, row_key FROM {t(table)}"
                                          f" WHERE row_identity_sha256 IN ({marks})", chunk):
            stored_by_identity[ident] = sha
        ks = [records[x]["row_key"] for x in chunk]
        for ident, key in _run(conn, f"SELECT row_identity_sha256, row_key FROM {t(table)}"
                                     f" WHERE run_id = ? AND row_key IN ({marks})", [run_id, *ks]):
            stored_by_key[key] = ident
    conflicts = [x for x, sha in stored_by_identity.items() if records[x]["row_sha256"] != sha]
    if conflicts:
        raise Refusal(f"{len(conflicts)} identities already stored with different content; "
                      f"first {conflicts[0][:12]}...; nothing written")
    key_conflicts = [k for k, ident in stored_by_key.items() if keys.get(k) != ident]
    if key_conflicts:
        raise Refusal(f"{len(key_conflicts)} row_keys already stored under another identity; "
                      f"first {key_conflicts[0][:12]}...; nothing written")
    new = [records[x] for x in idents if x not in stored_by_identity]
    cols = columns(table)
    unit_id = next(iter(unit_ids)) if len(unit_ids) == 1 else None
    receipt = {
        "schema": SCHEMA_RECEIPT, "backend": backend, "run_id": run_id, "run_sha256": run["run_sha256"],
        "population_id": population_id, "table": table, "unit_id": unit_id,
        "row_count": len(rows), "distinct_identities": len(records),
        "inserted": len(new), "duplicates_ignored": len(records) - len(new),
        "rows_sha256": rows_digest(r["row_sha256"] for r in records.values()),
        "host_role": host_role, "shard_id": shard_id, "submitted_at": now(),
    }
    receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    managed = manage_transaction and not _is_sqlalchemy(conn)
    if managed:
        _run(conn, "BEGIN TRANSACTION")
    try:
        _run_many(conn, f"INSERT INTO {t(table)} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                  [tuple(r[c] for c in cols) for r in new])
        _run(conn, f"INSERT INTO {t(RECEIPT_TABLE)} (receipt_sha256, run_id, population_id, table_name,"
                   " unit_id, row_count, distinct_identities, inserted, duplicates_ignored, rows_sha256,"
                   " host_role, shard_id, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
             [receipt["receipt_sha256"], run_id, population_id, table, unit_id, receipt["row_count"],
              receipt["distinct_identities"], receipt["inserted"], receipt["duplicates_ignored"],
              receipt["rows_sha256"], host_role, shard_id, receipt["submitted_at"]])
        if managed:
            _run(conn, "COMMIT")
    except Exception:
        if managed:
            _run(conn, "ROLLBACK")
        raise
    return receipt


def verify_receipt(conn, receipt: dict, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Accept a receipt only when its digest, run, population and stored digest agree."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA_RECEIPT:
        raise Refusal("not a phase-2/3 warehouse receipt")
    if digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != receipt.get("receipt_sha256"):
        raise Refusal("receipt digest does not match its body")
    run = get_run(conn, receipt.get("run_id"), qualified)
    if run is None:
        raise Refusal(f"receipt names an unknown run {receipt.get('run_id')!r}")
    if receipt.get("population_id") != run["population_id"]:
        raise Refusal(f"receipt carries foreign population {receipt.get('population_id')!r}; "
                      f"run {run['run_id']!r} is bound to {run['population_id']!r}")
    if receipt.get("run_sha256") != run["run_sha256"]:
        raise Refusal("receipt carries a run identity digest the store does not hold")
    rows = _run(conn, f"SELECT rows_sha256 FROM {qualified(RECEIPT_TABLE)} WHERE receipt_sha256 = ?",
                [receipt["receipt_sha256"]])
    if not rows:
        raise Refusal("the store holds no such receipt")
    if rows[0][0] != receipt.get("rows_sha256"):
        raise Refusal("stored receipt digest differs from the presented one")
    return {"accepted": True, "receipt_sha256": receipt["receipt_sha256"]}


# --------------------------------------------------------------------------- readback
def read_run(conn, run_id: str, table: str, unit_id: str | None = None, *, after: str | None = None,
             limit: int | None = None, qualified: Callable[[str], str] = lambda n: n) -> list[dict]:
    """Rows exactly as submitted (the payload), ordered by row_key; optional unit filter/paging."""
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}")
    if not run_id:
        raise Refusal("run_id is required")
    sql = f"SELECT payload FROM {qualified(table)} WHERE run_id = ?"
    params: list[Any] = [run_id]
    if unit_id is not None:
        sql += " AND unit_id = ?"
        params.append(unit_id)
    if after is not None:
        sql += " AND row_key > ?"
        params.append(after)
    sql += " ORDER BY row_key"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [json.loads(r[0]) for r in _run(conn, sql, params)]


def stored_summary_sql(table: str, run_id: str, qualified: Callable[[str], str] = lambda n: n) -> str:
    rid = run_id.replace("'", "''")
    return (f"SELECT count(*) AS n, coalesce(sha256(string_agg(row_sha256, '' ORDER BY row_sha256)),"
            f" '{EMPTY_DIGEST}') AS rows_sha256 FROM {qualified(table)} WHERE run_id = '{rid}' LIMIT 1")


def stored_summary(conn, table: str, run_id: str,
                   qualified: Callable[[str], str] = lambda n: n,
                   fetch_size: int = 8192) -> tuple[int, str]:
    """Return the stored row count and canonical digest with bounded Python memory.

    The digest contract is SHA-256 over the sorted concatenation of the stored
    row digests. ``sha256(string_agg(... ORDER BY ...))`` is equivalent, but it
    materializes hundreds of megabytes for large campaigns before hashing and
    exhausted the production warehouse during the EURUSD phase-2 closure.
    Fetching the ordered values in chunks preserves the exact bytes while the
    database may spill its sort and Python retains only one chunk.
    """
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}")
    if fetch_size < 1:
        raise ValueError("fetch_size must be positive")
    sql = f"SELECT row_sha256 FROM {qualified(table)} WHERE run_id = ? ORDER BY row_sha256"
    if _is_sqlalchemy(conn):
        result = conn.exec_driver_sql(sql, (run_id,))
    else:
        result = conn.execute(sql, [run_id])
    count = 0
    hasher = hashlib.sha256()
    while True:
        batch = result.fetchmany(fetch_size)
        if not batch:
            break
        for row in batch:
            hasher.update(row[0].encode("ascii"))
        count += len(batch)
    return count, hasher.hexdigest()


def reconcile(conn, run_id: str, receipts: list[dict] | None = None, expected: dict | None = None,
              *, backend: str = "duckdb", qualified: Callable[[str], str] = lambda n: n) -> dict:
    """Stored counts and digests per table for one run, against declared expectations and receipts."""
    run = get_run(conn, run_id, qualified)
    if run is None:
        raise Refusal(f"run {run_id!r} is not registered; nothing to reconcile")
    expected = dict(run["expected_json"]) if expected is None else dict(expected)
    tables = {}
    complete = True
    for table in FACT_TABLES:
        n, sha = stored_summary(conn, table, run_id, qualified)
        issued = _run(conn, f"SELECT count(*), coalesce(sum(inserted), 0) FROM {qualified(RECEIPT_TABLE)}"
                            " WHERE run_id = ? AND table_name = ?", [run_id, table])[0]
        entry = {"count": int(n), "rows_sha256": sha, "receipts": int(issued[0]),
                 "receipted_inserted": int(issued[1]), "expected": expected.get(table),
                 "receipts_cover_store": int(issued[1]) == int(n)}
        if expected.get(table) is not None:
            entry["count_matches_expected"] = int(n) == int(expected[table])
            complete = complete and entry["count_matches_expected"]
        complete = complete and entry["receipts_cover_store"]
        tables[table] = entry
    verified = [verify_receipt(conn, r, qualified) for r in (receipts or [])]
    report = {"schema": SCHEMA_RECONCILIATION, "backend": backend, "run_id": run_id,
              "run_sha256": run["run_sha256"], "population_id": run["population_id"],
              "phase": run["phase"], "registration": run["registration"], "tables": tables,
              "receipts_verified": len(verified), "complete": complete, "reconciled_at": now()}
    report["reconciliation_sha256"] = digest({k: v for k, v in report.items() if k != "reconciled_at"})
    return report


# --------------------------------------------------------------------------- host documents
def write_document(conn, document: dict, *, backend: str, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """The host route body: {"run_id", "table", "rows", "unit_id"?, "host_role"?, "shard_id"?}."""
    if not isinstance(document, dict):
        raise Refusal("the request body must be a JSON object")
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise Refusal("rows must be a list")
    return submit_rows(conn, document.get("run_id"), document.get("table"), rows,
                       host_role=document.get("host_role"), shard_id=document.get("shard_id"),
                       manage_transaction=False, backend=backend, qualified=qualified)


def read_document(conn, document: dict, *, qualified: Callable[[str], str] = lambda n: n) -> dict:
    """The host read route: {"run_id", "table", "unit_id"?, "after"?, "limit"?} -> rows + cursor."""
    if not isinstance(document, dict):
        raise Refusal("the request must be a JSON object")
    limit = int(document.get("limit") or 5000)
    if limit < 1 or limit > 10000:
        raise Refusal("limit must be between 1 and 10000")
    rows = read_run(conn, document.get("run_id"), document.get("table"), document.get("unit_id"),
                    after=document.get("after"), limit=limit, qualified=qualified)
    next_after = rows[-1]["row_key"] if len(rows) == limit else None
    return {"rows": rows, "count": len(rows), "next_after": next_after}


def reconcile_document(conn, document: dict, *, backend: str, qualified: Callable[[str], str] = lambda n: n) -> dict:
    if not isinstance(document, dict) or not document.get("run_id"):
        raise Refusal("the request must carry run_id")
    return reconcile(conn, document["run_id"], document.get("receipts"), document.get("expected"),
                     backend=backend, qualified=qualified)
