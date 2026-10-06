#!/usr/bin/env python3
"""Phase-2/3 feature-selection warehouse: contract, additive migration, submit, readback, snapshot.

Subplan: ``docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md``
(§2 identities, §3.3 closure, §6 acceptance) and the order
``docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md`` (§B, §D, §F, §G).

What this module owns, and what it refuses to own:

* **the frozen populations.** ``contract`` derives ``CONTRACT.json`` from the phase-1 closure
  artifacts (plan, final envelope, fold files, frozen TRAIN parquet files). Nothing in the
  contract is typed by hand: feature lists, targets, folds and digests are read, and the pair
  denominators ``n*(n-1)/2`` are computed and asserted against the declared 66,795 and 3,403.
* **the additive schema.** ``olap/migrations/fs_phase23/0001_fs_phase23_additive.sql`` is the
  single source of DDL. ``DDL`` below is parsed from that file, so the tool and any backend
  that applies the file cannot drift apart. Every statement is ``CREATE ... IF NOT EXISTS``;
  nothing is dropped, altered or rewritten.
* **the write semantics.** ``submit_rows`` stores rows keyed by a content-derived identity with
  a UNIQUE key over every identity column of subplan §2. An identical replay is a no-op; the
  same identity with different content rejects the whole batch; a row or receipt carrying a
  run or population identity other than the registered one is refused before any write.
* **readback.** ``read_run``, ``reconcile`` and ``readback_report`` compute order-independent
  digests with the same SQL expression locally and through the live service's read-only
  ``/api/v1/query`` route, so a local terminal and the warehouse are compared by the same
  arithmetic.
* **the physical snapshot.** ``snapshot`` compresses a *copy* of the cube to ``.duckdb.zst``,
  records SHA-256 of both forms, and writes the release-asset manifest the repository keeps.
  It refuses a path that looks like a live store (a sibling write-ahead log) because the live
  cube has exactly one owner, the warehouse service.

This module never opens the configured production cube. Tests use temporary files only.
All host references are roles (``coordinator``, ``worker_a``, ``worker_b``); no host name,
address, credential or private path is written into any artifact this module produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_CONTRACT = "fs_phase23_contract.v1"
SCHEMA_RECEIPT = "fs_phase23_load_receipt.v1"
SCHEMA_RECONCILIATION = "fs_phase23_reconciliation.v1"
SCHEMA_READBACK = "fs_phase23_readback_report.v1"
SCHEMA_SNAPSHOT = "olap_snapshot_manifest.v2"
MIGRATION_ID = "fs_phase23_0001"

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_SQL = REPO_ROOT / "olap" / "migrations" / "fs_phase23" / "0001_fs_phase23_additive.sql"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HOST_ROLES = ("coordinator", "worker_a", "worker_b")

STATES = ("MEASURED", "INSUFFICIENT_SUPPORT", "NOT_APPLICABLE", "FAILED")
ALIAS_DISPOSITIONS = ("ALIAS_BYTE_EXACT", "ALIAS_NUMERIC_TOLERANCE", "AFFINE_EXACT",
                      "MONOTONE_NEAR_PERFECT", "DISTINCT")
FILTER_METHODS = ("clustering_spearman", "mrmr_mi", "jmi",
                  "ALL_ADMISSIBLE", "univariate_mi", "CAUSAL_SUPPORTED", "RANDOM_K")

#: Run dimension and receipt table, then the four fact families of subplan §3.3 / §4.
RUN_TABLE = "fs_phase23_run"
RECEIPT_TABLE = "fs_phase23_load_receipt"
FACT_TABLES = ("feature_pair_metrics", "feature_alias_groups",
               "feature_redundancy_clusters", "feature_filter_rankings")
ALL_TABLES = (RUN_TABLE, RECEIPT_TABLE, *FACT_TABLES)

#: Identity columns per table: the UNIQUE key of subplan §2. ``row_identity_sha256`` is the
#: digest of exactly these columns, in this order, and the table also carries a UNIQUE
#: constraint over them so a second row with the same identity cannot be inserted by any path.
IDENTITY = {
    "feature_pair_metrics": (
        "run_id", "population_id", "feature_left", "feature_right", "fold", "lag",
        "method", "params_sha256", "code_sha256", "input_sha256"),
    "feature_alias_groups": (
        "run_id", "population_id", "group_id", "feature_id", "fold", "method",
        "params_sha256", "code_sha256", "input_sha256"),
    "feature_redundancy_clusters": (
        "run_id", "population_id", "fold", "method", "params_sha256", "code_sha256",
        "input_sha256", "cluster_id", "feature_id"),
    "feature_filter_rankings": (
        "run_id", "population_id", "target_id", "horizon", "fold", "method",
        "params_sha256", "code_sha256", "input_sha256", "feature_id"),
}

#: Every stored column, in table order. ``row_sha256`` digests these (minus itself and
#: ``row_identity_sha256``); ``stored_at`` is never part of a digest.
COLUMNS = {
    "feature_pair_metrics": (
        "row_identity_sha256", "row_sha256", "run_id", "population_id", "feature_left",
        "feature_right", "split", "fold", "lag", "method", "params_sha256", "code_sha256",
        "input_sha256", "shared_support", "metric_value", "state", "host_role", "shard_id",
        "terminal_sha256", "extra_json"),
    "feature_alias_groups": (
        "row_identity_sha256", "row_sha256", "run_id", "population_id", "group_id",
        "feature_id", "split", "fold", "method", "params_sha256", "code_sha256",
        "input_sha256", "disposition", "representative", "shared_support", "host_role",
        "shard_id", "terminal_sha256", "extra_json"),
    "feature_redundancy_clusters": (
        "row_identity_sha256", "row_sha256", "run_id", "population_id", "split", "fold",
        "method", "params_sha256", "code_sha256", "input_sha256", "cluster_id", "feature_id",
        "representative", "linkage_distance", "threshold", "host_role", "shard_id",
        "terminal_sha256", "extra_json"),
    "feature_filter_rankings": (
        "row_identity_sha256", "row_sha256", "run_id", "population_id", "target_id",
        "horizon", "split", "fold", "method", "params_sha256", "code_sha256", "input_sha256",
        "feature_id", "rank", "score", "relevance_term", "redundancy_term",
        "complementarity_term", "causal_term", "cost_term", "k_membership_json", "state",
        "host_role", "shard_id", "terminal_sha256", "extra_json"),
}

REQUIRED = {
    "feature_pair_metrics": ("run_id", "population_id", "feature_left", "feature_right",
                             "fold", "lag", "method", "params_sha256", "code_sha256",
                             "input_sha256", "shared_support", "state"),
    "feature_alias_groups": ("run_id", "population_id", "group_id", "feature_id", "fold",
                             "method", "params_sha256", "code_sha256", "input_sha256",
                             "disposition", "representative"),
    "feature_redundancy_clusters": ("run_id", "population_id", "fold", "method",
                                    "params_sha256", "code_sha256", "input_sha256",
                                    "cluster_id", "feature_id", "representative"),
    "feature_filter_rankings": ("run_id", "population_id", "target_id", "horizon", "fold",
                                "method", "params_sha256", "code_sha256", "input_sha256",
                                "feature_id", "rank", "state"),
}


class Refusal(ValueError):
    """A typed refusal: the request is wrong, not the store."""


# --------------------------------------------------------------------------- canonical bytes
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pair_denominator(n: int) -> int:
    """Unique unordered pairs of ``n`` features: n(n-1)/2."""
    if n < 2:
        raise Refusal(f"a population of {n} features has no pairs")
    return n * (n - 1) // 2


def order_pair(a: str, b: str) -> tuple[str, str]:
    """The ordered (left, right) identity of a pair. Equal names are not a pair."""
    if a == b:
        raise Refusal(f"{a!r} paired with itself is not a pair")
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------- the DDL
def load_ddl(path: Path = MIGRATION_SQL) -> list[str]:
    """Statements of the migration file, in order, comments stripped."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    statements = [s.strip() for s in "\n".join(lines).split(";")]
    return [s for s in statements if s]


def apply_migration(conn, path: Path = MIGRATION_SQL) -> dict:
    """Apply the additive migration. Idempotent: a second application changes nothing."""
    before = set(list_tables(conn))
    for statement in load_ddl(path):
        if not statement.upper().startswith(("CREATE TABLE IF NOT EXISTS",
                                             "CREATE INDEX IF NOT EXISTS",
                                             "CREATE OR REPLACE VIEW")):
            raise Refusal(f"migration {MIGRATION_ID} is additive only; refusing: "
                          f"{statement[:60]!r}")
        conn.execute(statement)
    after = set(list_tables(conn))
    missing = [t for t in ALL_TABLES if t not in after]
    if missing:
        raise RuntimeError(f"migration applied but tables missing: {missing}")
    return {"migration_id": MIGRATION_ID, "sql_sha256": file_sha256(path),
            "created": sorted(after - before), "already_present": sorted(before & set(ALL_TABLES)),
            "tables": list(ALL_TABLES)}


def list_tables(conn) -> list[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema() ORDER BY 1").fetchall()
    return [r[0] for r in rows]


def migration_plan(conn) -> dict:
    """What applying the migration WOULD do, without doing it (the dry run)."""
    present = set(list_tables(conn))
    return {"migration_id": MIGRATION_ID, "sql_sha256": file_sha256(MIGRATION_SQL),
            "would_create": [t for t in ALL_TABLES if t not in present],
            "already_present": [t for t in ALL_TABLES if t in present],
            "statements": len(load_ddl()), "destructive_statements": 0}


# --------------------------------------------------------------------------- rows
def _check_hex(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.match(value):
        raise Refusal(f"{name} must be a 64-hex SHA-256, got {value!r}")
    return value


def normalise_row(table: str, row: dict) -> dict:
    """Validate one row and return it with identity and content digests attached.

    Column order of the input is irrelevant: digests are computed over canonical JSON with
    sorted keys, so two producers that emit the same values in a different order agree.
    """
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}; expected one of {FACT_TABLES}")
    if not isinstance(row, dict):
        raise Refusal("a row must be a mapping")
    unknown = sorted(set(row) - set(COLUMNS[table]))
    if unknown:
        raise Refusal(f"{table}: unknown columns {unknown}")
    for column in REQUIRED[table]:
        if row.get(column) is None:
            raise Refusal(f"{table}: {column} is required")
    for column in ("params_sha256", "code_sha256", "input_sha256"):
        _check_hex(column, row[column])
    if row.get("terminal_sha256") is not None:
        _check_hex("terminal_sha256", row["terminal_sha256"])
    split = row.get("split", "train")
    if split != "train":
        raise Refusal(f"{table}: split must be 'train' (validation/test stay closed), "
                      f"got {split!r}")
    if row.get("host_role") is not None and row["host_role"] not in _HOST_ROLES:
        raise Refusal(f"{table}: host_role must be a role {list(_HOST_ROLES)}, "
                      f"got {row['host_role']!r}: no host names enter the warehouse")
    if table == "feature_pair_metrics":
        if not row["feature_left"] < row["feature_right"]:
            raise Refusal(f"{table}: pair must be ordered feature_left < feature_right, got "
                          f"({row['feature_left']!r}, {row['feature_right']!r}); use order_pair()")
        if row["state"] not in STATES:
            raise Refusal(f"{table}: state {row['state']!r} not in {STATES}")
        if int(row["shared_support"]) < 0 or int(row["lag"]) < 0:
            raise Refusal(f"{table}: shared_support and lag must be non-negative")
        if row["state"] == "MEASURED" and row.get("metric_value") is None:
            raise Refusal(f"{table}: a MEASURED row carries a metric_value")
    if table == "feature_alias_groups" and row["disposition"] not in ALIAS_DISPOSITIONS:
        raise Refusal(f"{table}: disposition {row['disposition']!r} not in {ALIAS_DISPOSITIONS}")
    if table == "feature_filter_rankings":
        if row["method"] not in FILTER_METHODS:
            raise Refusal(f"{table}: method {row['method']!r} not in {FILTER_METHODS}")
        if row["state"] not in STATES:
            raise Refusal(f"{table}: state {row['state']!r} not in {STATES}")
        if int(row["rank"]) < 1:
            raise Refusal(f"{table}: rank is 1-based")
    out = {column: row.get(column) for column in COLUMNS[table]}
    out["split"] = "train"
    for json_column in ("extra_json", "k_membership_json"):
        if json_column in out and out[json_column] is not None and not isinstance(out[json_column], str):
            out[json_column] = canonical_bytes(out[json_column]).decode("ascii")
    identity = {column: out[column] for column in IDENTITY[table]}
    out["row_identity_sha256"] = digest(identity)
    body = {column: out[column] for column in COLUMNS[table]
            if column not in ("row_identity_sha256", "row_sha256")}
    out["row_sha256"] = digest(body)
    return out


def rows_digest(row_digests: Iterable[str]) -> str:
    """Order-independent digest of a set of row digests: sha256 of the sorted concatenation.

    The SQL twin is ``sha256(string_agg(row_sha256, '' ORDER BY row_sha256))`` and the two must
    agree; ``test_fs_phase23_warehouse`` checks that they do.
    """
    return hashlib.sha256("".join(sorted(row_digests)).encode("ascii")).hexdigest()


# --------------------------------------------------------------------------- runs
def register_run(conn, run: dict) -> dict:
    """Register a run (population identity bound) before any row may be submitted.

    ``run`` carries: run_id, population_id, phase ('PHASE_2' | 'PHASE_3'), contract_sha256,
    campaign_sha256, code_sha256, input_sha256, expected_json (declared expected counts per
    table, used by ``reconcile``). Re-registering an identical run is a no-op; a different
    document under the same run_id is refused.
    """
    for key in ("run_id", "population_id", "phase", "contract_sha256", "campaign_sha256",
                "code_sha256", "input_sha256"):
        if not run.get(key):
            raise Refusal(f"run.{key} is required")
    if run["phase"] not in ("PHASE_2", "PHASE_3"):
        raise Refusal("run.phase must be PHASE_2 or PHASE_3")
    for key in ("contract_sha256", "campaign_sha256", "code_sha256", "input_sha256"):
        _check_hex(key, run[key])
    expected = run.get("expected_json") or {}
    if not isinstance(expected, dict):
        raise Refusal("run.expected_json must be a mapping table -> expected row count")
    body = {k: run[k] for k in ("run_id", "population_id", "phase", "contract_sha256",
                                "campaign_sha256", "code_sha256", "input_sha256")}
    body["expected_json"] = canonical_bytes(expected).decode("ascii")
    run_sha = digest(body)
    stored = conn.execute(
        f"SELECT run_sha256 FROM {RUN_TABLE} WHERE run_id = ?", [run["run_id"]]).fetchone()
    if stored:
        if stored[0] != run_sha:
            raise Refusal(f"run {run['run_id']!r} is already registered with a different "
                          f"identity ({stored[0][:12]}… vs {run_sha[:12]}…)")
        return {"run_id": run["run_id"], "run_sha256": run_sha, "already_registered": True}
    conn.execute(
        f"INSERT INTO {RUN_TABLE} (run_id, run_sha256, population_id, phase, contract_sha256, "
        "campaign_sha256, code_sha256, input_sha256, expected_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [run["run_id"], run_sha, run["population_id"], run["phase"], run["contract_sha256"],
         run["campaign_sha256"], run["code_sha256"], run["input_sha256"], body["expected_json"],
         now()])
    return {"run_id": run["run_id"], "run_sha256": run_sha, "already_registered": False}


def get_run(conn, run_id: str) -> dict:
    row = conn.execute(
        f"SELECT run_id, run_sha256, population_id, phase, contract_sha256, campaign_sha256, "
        f"code_sha256, input_sha256, expected_json, created_at FROM {RUN_TABLE} WHERE run_id = ?",
        [run_id]).fetchone()
    if not row:
        raise Refusal(f"run {run_id!r} is not registered; register_run first")
    keys = ("run_id", "run_sha256", "population_id", "phase", "contract_sha256",
            "campaign_sha256", "code_sha256", "input_sha256", "expected_json", "created_at")
    out = dict(zip(keys, row))
    out["expected_json"] = json.loads(out["expected_json"] or "{}")
    return out


# --------------------------------------------------------------------------- submit
def submit_rows(conn, run_id: str, table: str, rows: list[dict], *, host_role: str | None = None,
                shard_id: str | None = None) -> dict:
    """Store rows for a registered run and return a receipt with counts and digests.

    Refuses, before writing anything: an unregistered run; a row whose ``run_id`` or
    ``population_id`` differs from the registered run (a foreign identity); a malformed row;
    two rows with one identity and different content inside the batch; and a row whose
    identity is already stored with different content. Identical rows already stored are
    counted as ``already_stored`` and not written again. The write is one transaction.
    """
    run = get_run(conn, run_id)
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}")
    if not rows:
        raise Refusal("submit_rows received no rows")
    prepared: dict[str, dict] = {}
    for raw in rows:
        row = dict(raw)
        if host_role is not None:
            row.setdefault("host_role", host_role)
        if shard_id is not None:
            row.setdefault("shard_id", shard_id)
        row = normalise_row(table, row)
        if row["run_id"] != run_id:
            raise Refusal(f"foreign run identity: row carries run_id {row['run_id']!r}, "
                          f"submission is for {run_id!r}")
        if row["population_id"] != run["population_id"]:
            raise Refusal(f"foreign population identity: row carries {row['population_id']!r}, "
                          f"run {run_id!r} is bound to {run['population_id']!r}")
        prior = prepared.get(row["row_identity_sha256"])
        if prior is not None and prior["row_sha256"] != row["row_sha256"]:
            raise Refusal("the batch carries one identity with two different contents: "
                          f"{row['row_identity_sha256'][:12]}…")
        prepared[row["row_identity_sha256"]] = row
    identities = list(prepared)
    stored = {}
    for i in range(0, len(identities), 1000):
        chunk = identities[i:i + 1000]
        placeholders = ",".join("?" * len(chunk))
        for ident, sha in conn.execute(
                f"SELECT row_identity_sha256, row_sha256 FROM {table} "
                f"WHERE row_identity_sha256 IN ({placeholders})", chunk).fetchall():
            stored[ident] = sha
    conflicts = [ident for ident, sha in stored.items() if prepared[ident]["row_sha256"] != sha]
    if conflicts:
        raise Refusal(f"{len(conflicts)} identities already stored with different content; "
                      f"first {conflicts[0][:12]}…; nothing written")
    new = [prepared[ident] for ident in identities if ident not in stored]
    columns = COLUMNS[table]
    conn.execute("BEGIN TRANSACTION")
    try:
        if new:
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                [[row[c] for c in columns] for row in new])
        receipt = {
            "schema": SCHEMA_RECEIPT, "run_id": run_id, "run_sha256": run["run_sha256"],
            "population_id": run["population_id"], "table": table,
            "submitted": len(rows), "distinct_identities": len(identities),
            "stored_new": len(new), "already_stored": len(identities) - len(new),
            "rows_sha256": rows_digest(row["row_sha256"] for row in prepared.values()),
            "host_role": host_role, "shard_id": shard_id, "submitted_at": now(),
        }
        receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "submitted_at"})
        conn.execute(
            f"INSERT INTO {RECEIPT_TABLE} (receipt_sha256, run_id, population_id, table_name, "
            "submitted, distinct_identities, stored_new, already_stored, rows_sha256, host_role, "
            "shard_id, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [receipt["receipt_sha256"], run_id, run["population_id"], table, receipt["submitted"],
             receipt["distinct_identities"], receipt["stored_new"], receipt["already_stored"],
             receipt["rows_sha256"], host_role, shard_id, receipt["submitted_at"]])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return receipt


def verify_receipt(conn, receipt: dict) -> dict:
    """A receipt is accepted only if its run, population and digest match the store."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA_RECEIPT:
        raise Refusal("not a phase-2/3 load receipt")
    body = {k: v for k, v in receipt.items() if k not in ("submitted_at", "receipt_sha256")}
    if digest(body) != receipt.get("receipt_sha256"):
        raise Refusal("receipt digest does not match its body")
    run = get_run(conn, receipt["run_id"])
    if receipt.get("population_id") != run["population_id"]:
        raise Refusal(f"receipt carries foreign population {receipt.get('population_id')!r}; "
                      f"run {run['run_id']!r} is bound to {run['population_id']!r}")
    if receipt.get("run_sha256") != run["run_sha256"]:
        raise Refusal("receipt carries a run identity digest the store does not hold")
    row = conn.execute(
        f"SELECT rows_sha256 FROM {RECEIPT_TABLE} WHERE receipt_sha256 = ?",
        [receipt["receipt_sha256"]]).fetchone()
    if not row:
        raise Refusal("the store holds no such receipt")
    if row[0] != receipt["rows_sha256"]:
        raise Refusal("stored receipt digest differs from the presented one")
    return {"accepted": True, "receipt_sha256": receipt["receipt_sha256"]}


# --------------------------------------------------------------------------- readback
def read_run(conn, run_id: str, table: str) -> list[dict]:
    if table not in FACT_TABLES:
        raise Refusal(f"unknown table {table!r}")
    get_run(conn, run_id)
    columns = COLUMNS[table]
    rows = conn.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE run_id = ? ORDER BY row_identity_sha256",
        [run_id]).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def _stored_summary_sql(table: str, run_id: str) -> str:
    rid = run_id.replace("'", "''")
    return (f"SELECT count(*) AS n, "
            f"coalesce(sha256(string_agg(row_sha256, '' ORDER BY row_sha256)), '') AS rows_sha256 "
            f"FROM {table} WHERE run_id = '{rid}' LIMIT 1")


def reconcile(conn, run_id: str, receipts: list[dict] | None = None,
              expected: dict[str, int] | None = None) -> dict:
    """Expected versus stored counts and digests, per table.

    ``expected`` counts default to the run's declared ``expected_json``. When receipts are
    given, each is verified and its digest is compared with the store; the union of receipt
    identities is also required to equal the stored population of the table (a receipt that
    was never issued, or rows that arrived without a receipt, both fail).
    """
    run = get_run(conn, run_id)
    expected = dict(run["expected_json"]) if expected is None else dict(expected)
    tables = {}
    complete = True
    for table in FACT_TABLES:
        n, sha = conn.execute(_stored_summary_sql(table, run_id)).fetchone()
        rows_sha = sha if n else rows_digest([])
        issued = conn.execute(
            f"SELECT count(*), coalesce(sum(stored_new), 0) FROM {RECEIPT_TABLE} "
            f"WHERE run_id = ? AND table_name = ?", [run_id, table]).fetchone()
        entry = {"stored": int(n), "stored_rows_sha256": rows_sha,
                 "receipts": int(issued[0]), "receipted_new_rows": int(issued[1]),
                 "expected": expected.get(table)}
        if expected.get(table) is not None:
            entry["count_matches_expected"] = int(n) == int(expected[table])
            complete &= entry["count_matches_expected"]
        entry["receipts_cover_store"] = int(issued[1]) == int(n)
        complete &= entry["receipts_cover_store"]
        tables[table] = entry
    verified = []
    if receipts:
        for receipt in receipts:
            verified.append(verify_receipt(conn, receipt))
    report = {"schema": SCHEMA_RECONCILIATION, "run_id": run_id, "run_sha256": run["run_sha256"],
              "population_id": run["population_id"], "phase": run["phase"],
              "tables": tables, "receipts_verified": len(verified), "complete": complete,
              "generated_at": now()}
    report["reconciliation_sha256"] = digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def _readback_sql(table: str, run_id: str | None) -> str:
    where = f"WHERE run_id = '{run_id.replace(chr(39), chr(39)*2)}'" if run_id else ""
    return (f"SELECT population_id, method, coalesce(host_role, 'UNDECLARED') AS host_role, "
            f"fold, count(*) AS n, "
            f"sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) AS rows_sha256, "
            f"sum(CASE WHEN state = 'MEASURED' THEN 1 ELSE 0 END) AS measured, "
            f"sum(CASE WHEN state = 'INSUFFICIENT_SUPPORT' THEN 1 ELSE 0 END) AS insufficient, "
            f"sum(CASE WHEN state = 'FAILED' THEN 1 ELSE 0 END) AS failed "
            f"FROM {table} {where} GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4 LIMIT 100000")


def _readback_sql_no_state(table: str, run_id: str | None) -> str:
    where = f"WHERE run_id = '{run_id.replace(chr(39), chr(39)*2)}'" if run_id else ""
    return (f"SELECT population_id, method, coalesce(host_role, 'UNDECLARED') AS host_role, "
            f"fold, count(*) AS n, "
            f"sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) AS rows_sha256, "
            f"NULL AS measured, NULL AS insufficient, NULL AS failed "
            f"FROM {table} {where} GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4 LIMIT 100000")


def readback_report(query, run_id: str | None = None, *, source: str = "local") -> dict:
    """Counts and digests per asset (population) × method × host role × fold, per table.

    ``query`` is any callable ``sql -> list[dict]``: ``local_query(conn)`` for a file, or
    ``Service.query`` for the live read-only route. The SQL is identical in both cases, so a
    terminal and the warehouse are compared by one expression. This is the document that
    ``close-phase2`` / ``close-phase3`` consume.
    """
    tables = {}
    for table in FACT_TABLES:
        sql = _readback_sql(table, run_id) if table in ("feature_pair_metrics", "feature_filter_rankings") \
            else _readback_sql_no_state(table, run_id)
        groups = []
        for row in query(sql):
            groups.append({"population_id": row["population_id"], "method": row["method"],
                           "host_role": row["host_role"], "fold": row["fold"], "n": int(row["n"]),
                           "rows_sha256": row["rows_sha256"],
                           "measured": None if row["measured"] is None else int(row["measured"]),
                           "insufficient_support": None if row["insufficient"] is None else int(row["insufficient"]),
                           "failed": None if row["failed"] is None else int(row["failed"])})
        by_asset: dict[str, int] = {}
        by_method: dict[str, int] = {}
        by_host: dict[str, int] = {}
        for g in groups:
            by_asset[g["population_id"]] = by_asset.get(g["population_id"], 0) + g["n"]
            by_method[g["method"]] = by_method.get(g["method"], 0) + g["n"]
            by_host[g["host_role"]] = by_host.get(g["host_role"], 0) + g["n"]
        tables[table] = {"groups": groups, "total": sum(g["n"] for g in groups),
                         "table_rows_sha256": rows_digest(g["rows_sha256"] for g in groups),
                         "by_asset": by_asset, "by_method": by_method, "by_host_role": by_host}
    report = {"schema": SCHEMA_READBACK, "source": source, "run_id": run_id, "tables": tables,
              "generated_at": now()}
    report["report_sha256"] = digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def compare_readback(local: dict, remote: dict) -> dict:
    """Two readback reports agree when every (table, group) has the same n and digest."""
    differences = []
    for table in FACT_TABLES:
        left = {(g["population_id"], g["method"], g["host_role"], g["fold"]): g
                for g in local["tables"][table]["groups"]}
        right = {(g["population_id"], g["method"], g["host_role"], g["fold"]): g
                 for g in remote["tables"][table]["groups"]}
        for key in sorted(set(left) | set(right)):
            a, b = left.get(key), right.get(key)
            if a is None or b is None or a["n"] != b["n"] or a["rows_sha256"] != b["rows_sha256"]:
                differences.append({"table": table, "group": list(key),
                                    "local": None if a is None else {"n": a["n"], "rows_sha256": a["rows_sha256"]},
                                    "remote": None if b is None else {"n": b["n"], "rows_sha256": b["rows_sha256"]}})
    return {"agree": not differences, "differences": differences}


def local_query(conn):
    def run(sql: str) -> list[dict]:
        cursor = conn.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    return run


class Service:
    """Read-only access to the running warehouse through its query route. No write method."""

    def __init__(self, url: str, token_env: str = "WAREHOUSE_TOKEN"):
        self.url = url.rstrip("/")
        self.token = os.environ.get(token_env)
        if not self.token:
            raise Refusal(f"{token_env} is not set: the service token comes from the "
                          "environment, never from an argument or a file in the repository")

    def query(self, sql: str) -> list[dict]:
        request = urllib.request.Request(
            f"{self.url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=300) as handle:
            body = json.load(handle)
        return body.get("rows") or []


# --------------------------------------------------------------------------- the contract
def _targets_from_envelope(rows: dict) -> list[dict]:
    seen = {}
    for family in ("selection_decisions", "information_metrics", "pair_relations"):
        for row in rows.get(family, []):
            if row.get("target_id") is not None:
                seen[row["target_id"]] = int(row["horizon"])
    return [{"target_id": t, "horizon": h} for t, h in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))]


def _causal_supported(rows: dict) -> list[dict]:
    out = []
    for row in rows.get("selection_decisions", []):
        if row.get("decision") == "SELECTED":
            out.append({"feature_id": row["feature_id"], "target_id": row["target_id"],
                        "horizon": int(row["horizon"]), "method": row.get("method"),
                        "phase1_evidence_sha256": row.get("evidence_sha256")})
    return sorted(out, key=lambda r: (r["feature_id"], r["horizon"], r["target_id"]))


def build_population(population_id: str, run_id: str, plan_path: Path, envelope_path: Path,
                     folds_path: Path, features_parquet: Path, targets_parquet: Path,
                     completion_path: Path, *, expected_features: int, expected_targets: int,
                     expected_pairs: int, train_period: list, fold_names: list[str] | None,
                     causal_note: str | None = None) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    envelope = json.loads(Path(envelope_path).read_text(encoding="utf-8"))
    if "envelope" in envelope and "rows" not in envelope:
        envelope = envelope["envelope"]
    folds_doc = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    completion = json.loads(Path(completion_path).read_text(encoding="utf-8"))
    run = envelope["run"]
    if run["run_id"] != run_id:
        raise Refusal(f"{population_id}: envelope run_id {run['run_id']!r} is not {run_id!r}")
    if plan["population_id"] != population_id:
        raise Refusal(f"plan population {plan['population_id']!r} is not {population_id!r}")
    features = sorted(item["feature_id"] for item in plan["items"])
    if len(features) != expected_features or len(set(features)) != expected_features:
        raise Refusal(f"{population_id}: plan holds {len(features)} features, expected {expected_features}")
    targets = _targets_from_envelope(envelope["rows"])
    if len(targets) != expected_targets:
        raise Refusal(f"{population_id}: envelope names {len(targets)} targets, expected {expected_targets}")
    pairs = pair_denominator(len(features))
    if pairs != expected_pairs:
        raise Refusal(f"{population_id}: n(n-1)/2 = {pairs}, declared {expected_pairs}")
    folds = []
    for index, fold in enumerate(folds_doc["folds"]):
        name = fold.get("name") or (fold_names[index] if fold_names else f"inner_{index + 1}")
        folds.append({"name": name, "train_rows": list(fold["train_rows"]),
                      "validation_rows": list(fold.get("validation_rows") or fold.get("val_rows")),
                      "train_time": fold.get("train_time"), "validation_time": fold.get("val_time"),
                      "label_purge_rows": fold.get("label_purge_h")})
    row_population_ids = sorted({r.get("population_id") for fam in envelope["rows"].values()
                                 for r in fam if r.get("population_id")})
    causal = _causal_supported(envelope["rows"])
    distinct_causal = sorted({c["feature_id"] for c in causal})
    return {
        "population_id": population_id,
        "phase1_run_id": run_id,
        "phase1_identity": {
            "run_id": run_id,
            "campaign_sha256": run["campaign_sha256"], "code_sha256": run["code_sha256"],
            "input_sha256": run["input_sha256"], "inventory_sha256": run["inventory_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "final_envelope_sha256": envelope.get("envelope_sha256"),
            "row_population_id_values": row_population_ids,
            "phase1_completion": completion,
        },
        "feature_count": len(features), "features": features,
        "target_count": len(targets), "targets": targets,
        "target_pack": plan.get("target_pack"),
        "pair_denominator": pairs, "pair_denominator_rule": "n*(n-1)/2 over the frozen feature list",
        "train_period": train_period,
        "fold_count": len(folds), "folds": folds,
        "fold_source_schema": folds_doc.get("schema"),
        "sources": [
            {"role": "phase1_plan", "sha256": file_sha256(plan_path)},
            {"role": "phase1_final_envelope", "sha256": file_sha256(envelope_path)},
            {"role": "phase1_folds", "sha256": file_sha256(folds_path)},
            {"role": "train_features_parquet", "sha256": file_sha256(features_parquet)},
            {"role": "train_targets_parquet", "sha256": file_sha256(targets_parquet)},
            {"role": "phase1_completion", "sha256": file_sha256(completion_path)},
        ],
        "causal_supported": {
            "label": "CAUSAL_SUPPORTED",
            "meaning": "phase-1 causal-ladder support for a feature-target pair; a control variant "
                       "of the phase-3 ranking, NOT a final selection and NOT a predictive claim",
            "pair_count": len(causal), "distinct_feature_count": len(distinct_causal),
            "distinct_features": distinct_causal, "pairs": causal,
            "note": causal_note,
        },
    }


def build_contract(args) -> dict:
    eurusd = build_population(
        "EURUSD", "phase1-eurusd-final:94d20c038d55e152", args.eurusd_plan, args.eurusd_envelope,
        args.eurusd_folds, args.eurusd_features, args.eurusd_targets, args.eurusd_completion,
        expected_features=366, expected_targets=14, expected_pairs=66_795,
        train_period=["2012-05-01T00:00:00Z", "2024-01-01T00:00:00Z"], fold_names=None)
    eth = build_population(
        "ETH", "phase1-final:ETH:29d2f745f5d9e87c", args.eth_plan, args.eth_envelope,
        args.eth_folds, args.eth_features, args.eth_targets, args.eth_completion,
        expected_features=83, expected_targets=6, expected_pairs=3_403,
        train_period=[None, "2024-01-01T00:00:00Z"],
        fold_names=["inner_1", "inner_2", "inner_3"],
        causal_note="ETH phase-1 is PROVISIONAL_DEVELOPMENT under its point-in-time caveat; "
                    "no point-in-time admissibility, production selection or EURUSD comparability "
                    "claim is carried by this contract")
    if eurusd["causal_supported"]["distinct_feature_count"] != 12 or eurusd["causal_supported"]["pair_count"] != 34:
        raise Refusal("EURUSD causal support is not the 34-pair / 12-feature phase-1 record")
    if eth["causal_supported"]["distinct_feature_count"] != 3:
        raise Refusal("ETH causal support is not the three-feature phase-1 record")
    contract = {
        "schema": SCHEMA_CONTRACT,
        "frozen_on": args.frozen_on,
        "authority": [
            "docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md",
            "docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md",
        ],
        "populations": {"EURUSD": eurusd, "ETH": eth},
        "rules": {
            "pair_identity": "feature_left < feature_right (lexicographic); a pair is stored once; "
                             "lagged relations carry lag >= 0 under the same ordered identity",
            "alias_evidence": "an alias group contributes ONE independent observation; aliases never "
                              "count as independent evidence in any pair denominator, cluster, ranking "
                              "or stability statistic; the disposition and representative are stored, "
                              "no column is deleted",
            "splits": "only split='train' rows exist; validation and test remain closed; fitting, "
                      "discretisation, scaling and estimators resolve inside each TRAIN fold",
            "causal_supported": "the 12 EURUSD and 3 ETH phase-1 features are CAUSAL_SUPPORTED candidates "
                                "feeding a declared ranking variant and the CAUSAL_SUPPORTED control; "
                                "they are not the final selection; NOT_IDENTIFIED is zero causal "
                                "evidence, not rejection",
            "row_identity": "run_id, population_id, ordered pair (or feature), fold, lag, method, "
                            "params_sha256, code_sha256, input_sha256 -> row_identity_sha256; UNIQUE",
            "lags_hours": [0, 1, 2, 6, 24, 48, 168],
            "k_path": [4, 8, 12, 16, 24, 32],
            "states": list(STATES),
            "host_references": "roles only: coordinator, worker_a, worker_b",
        },
        "warehouse": {
            "migration_id": MIGRATION_ID,
            "migration_sql": str(MIGRATION_SQL.relative_to(REPO_ROOT)),
            "migration_sql_sha256": file_sha256(MIGRATION_SQL) if MIGRATION_SQL.exists() else None,
            "tables": list(ALL_TABLES),
        },
    }
    contract["contract_sha256"] = digest({k: v for k, v in contract.items() if k != "contract_sha256"})
    return contract


def verify_contract(contract: dict) -> dict:
    """Re-check the arithmetic and labels of a contract document (used by tests and closures)."""
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if digest(body) != contract.get("contract_sha256"):
        raise Refusal("contract digest does not match its body")
    out = {}
    for pid, pop in contract["populations"].items():
        n = len(pop["features"])
        if n != pop["feature_count"] or len(set(pop["features"])) != n:
            raise Refusal(f"{pid}: feature list is not {pop['feature_count']} unique names")
        if pair_denominator(n) != pop["pair_denominator"]:
            raise Refusal(f"{pid}: denominator mismatch")
        if len(pop["targets"]) != pop["target_count"]:
            raise Refusal(f"{pid}: target count mismatch")
        if pop["causal_supported"]["label"] != "CAUSAL_SUPPORTED":
            raise Refusal(f"{pid}: causal features must be labelled CAUSAL_SUPPORTED")
        unknown = set(pop["causal_supported"]["distinct_features"]) - set(pop["features"])
        if unknown:
            raise Refusal(f"{pid}: causal features outside the population: {sorted(unknown)}")
        out[pid] = {"features": n, "pairs": pop["pair_denominator"], "targets": len(pop["targets"]),
                    "folds": pop["fold_count"], "causal_supported_features": pop["causal_supported"]["distinct_feature_count"]}
    return out


# --------------------------------------------------------------------------- snapshot
def _zstd_compress(source: Path, target: Path, zstd_bin: str | None, level: int) -> str:
    try:
        import zstandard  # type: ignore
    except ImportError:
        zstandard = None
    if zstandard is not None:
        compressor = zstandard.ZstdCompressor(level=level)
        with source.open("rb") as src, target.open("wb") as dst:
            compressor.copy_stream(src, dst)
        return f"python-zstandard {zstandard.__version__}"
    binary = zstd_bin or shutil.which("zstd")
    if not binary:
        raise Refusal("neither the zstandard module nor a zstd binary is available")
    subprocess.run([binary, "-q", f"-{level}", "-T1", "--force", "-o", str(target), str(source)],
                   check=True)
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    return version or "zstd"


def snapshot(source: Path, out_dir: Path, *, tag: str, repo: str, phase: str, zstd_bin: str | None,
             level: int = 19, warehouse_runs: dict | None = None, expected_tables: Iterable[str] = ALL_TABLES) -> dict:
    """Compress a cube COPY to .duckdb.zst, digest both forms, write the release-asset manifest."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise Refusal(f"snapshot source does not exist: {source.name}")
    wal = source.with_name(source.name + ".wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise Refusal("the source has a non-empty write-ahead log beside it: that is a live store, "
                      "not a snapshot copy. Take the copy with tools/olap_duckdb_migrate.py snapshot "
                      "(it CHECKPOINTs the copy) and point this command at the copy")
    import duckdb  # local import: the contract subcommand must not need it
    conn = duckdb.connect(str(source), read_only=True)
    try:
        present = set(list_tables(conn))
        counts = {}
        for table in expected_tables:
            if table in present:
                counts[table] = {"rows": int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])}
                if table in FACT_TABLES:
                    n, sha = conn.execute(
                        f"SELECT count(*), coalesce(sha256(string_agg(row_sha256, '' ORDER BY row_sha256)), '') "
                        f"FROM {table}").fetchone()
                    counts[table]["rows_sha256"] = sha if n else rows_digest([])
            else:
                counts[table] = {"rows": None, "missing": True}
    finally:
        conn.close()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = out_dir / f"{source.stem}.duckdb.zst"
    tool = _zstd_compress(source, asset, zstd_bin, level)
    manifest = {
        "schema": SCHEMA_SNAPSHOT, "phase": phase, "created_on": datetime.now(timezone.utc).date().isoformat(),
        "database": "DuckDB",
        "snapshot_file": source.name, "snapshot_size_bytes": source.stat().st_size,
        "snapshot_sha256": file_sha256(source),
        "compressed_asset": asset.name, "compressed_size_bytes": asset.stat().st_size,
        "compressed_sha256": file_sha256(asset), "compressor": tool, "zstd_level": level,
        "release_tag": tag, "release_url": f"https://github.com/{repo}/releases/tag/{tag}",
        "asset_url": f"https://github.com/{repo}/releases/download/{tag}/{asset.name}",
        "relations": counts, "warehouse_runs": warehouse_runs or {},
        "missing_relations": [t for t, c in counts.items() if c.get("missing")],
        "custody_policy": "The repository retains this manifest, the schema and the SHA-256; the "
                          "compressed binary is a release asset. Restore requires SHA-256 "
                          "verification of the asset and of the decompressed file before opening.",
        "publish_commands": [
            f"gh release create {tag} --repo {repo} --title '{phase} OLAP snapshot' --notes-file RELEASE_NOTES.md",
            f"gh release upload {tag} {asset.name} --repo {repo} --clobber",
        ],
    }
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    (out_dir / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    (out_dir / f"{asset.name}.sha256").write_text(f"{manifest['compressed_sha256']}  {asset.name}\n",
                                                  encoding="utf-8")
    return manifest


def verify_snapshot_asset(asset: Path, manifest: dict, *, zstd_bin: str | None, work_dir: Path) -> dict:
    """Check the asset digest, decompress, check the file digest, open read-only, count rows."""
    asset = Path(asset)
    if file_sha256(asset) != manifest["compressed_sha256"]:
        raise Refusal("asset SHA-256 does not match the manifest; refusing to open")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    restored = work_dir / manifest["snapshot_file"]
    try:
        import zstandard  # type: ignore
        with asset.open("rb") as src, restored.open("wb") as dst:
            zstandard.ZstdDecompressor().copy_stream(src, dst)
    except ImportError:
        binary = zstd_bin or shutil.which("zstd")
        if not binary:
            raise Refusal("no zstd available to decompress")
        subprocess.run([binary, "-q", "-d", "--force", "-o", str(restored), str(asset)], check=True)
    if file_sha256(restored) != manifest["snapshot_sha256"]:
        raise Refusal("decompressed file SHA-256 does not match the manifest")
    import duckdb
    conn = duckdb.connect(str(restored), read_only=True)
    try:
        observed = {}
        for table, expected in manifest["relations"].items():
            if expected.get("missing"):
                continue
            observed[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            if observed[table] != expected["rows"]:
                raise Refusal(f"{table}: {observed[table]} rows, manifest says {expected['rows']}")
    finally:
        conn.close()
    return {"verified": True, "restored": restored.name, "relations": observed}


# --------------------------------------------------------------------------- CLI
def _connect(path: str, read_only: bool = False):
    import duckdb
    return duckdb.connect(str(Path(path).expanduser()), read_only=read_only,
                          config={"memory_limit": "256MB", "threads": 1})


def _dump(document: dict, out: str | None) -> None:
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
    sys.stdout.write(text if not out else f"wrote {out}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="derive CONTRACT.json from phase-1 artifacts")
    for pop in ("eurusd", "eth"):
        for role in ("plan", "envelope", "folds", "features", "targets", "completion"):
            contract.add_argument(f"--{pop}-{role}", required=True, type=Path)
    contract.add_argument("--frozen-on", default=datetime.now(timezone.utc).date().isoformat())
    contract.add_argument("--out", required=True)

    verify = sub.add_parser("verify-contract", help="re-check a contract's arithmetic and labels")
    verify.add_argument("contract", type=Path)

    migrate = sub.add_parser("migrate", help="apply (or plan) the additive migration on a DuckDB file")
    migrate.add_argument("--duckdb", required=True)
    migrate.add_argument("--dry-run", action="store_true", help="report the plan; write nothing")
    migrate.add_argument("--out")

    rb = sub.add_parser("readback", help="counts/digests per asset x method x host role")
    rb.add_argument("--duckdb", help="a local file (opened read-only)")
    rb.add_argument("--service-url", help="the live store's read-only query route")
    rb.add_argument("--token-env", default="WAREHOUSE_TOKEN")
    rb.add_argument("--run-id")
    rb.add_argument("--out")

    rc = sub.add_parser("reconcile", help="expected vs stored counts and digests for one run")
    rc.add_argument("--duckdb", required=True)
    rc.add_argument("--run-id", required=True)
    rc.add_argument("--receipts", nargs="*", type=Path, default=[])
    rc.add_argument("--out")

    cmp_ = sub.add_parser("compare-readback", help="do two readback reports agree?")
    cmp_.add_argument("local", type=Path)
    cmp_.add_argument("remote", type=Path)

    snap = sub.add_parser("snapshot", help="compress a cube COPY to .duckdb.zst with manifest")
    snap.add_argument("--source", required=True, help="a checkpointed copy, never the live cube")
    snap.add_argument("--out-dir", required=True)
    snap.add_argument("--tag", required=True)
    snap.add_argument("--repo", default="harveybc/predictor")
    snap.add_argument("--phase", required=True)
    snap.add_argument("--zstd-bin")
    snap.add_argument("--level", type=int, default=19)
    snap.add_argument("--publish", action="store_true", help="run the gh commands after writing")

    vs = sub.add_parser("verify-snapshot", help="verify an asset against its manifest")
    vs.add_argument("--asset", required=True, type=Path)
    vs.add_argument("--manifest", required=True, type=Path)
    vs.add_argument("--work-dir", required=True, type=Path)
    vs.add_argument("--zstd-bin")

    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            document = build_contract(args)
            verify_contract(document)
            _dump(document, args.out)
        elif args.command == "verify-contract":
            _dump(verify_contract(json.loads(args.contract.read_text(encoding="utf-8"))), None)
        elif args.command == "migrate":
            if args.dry_run and not Path(args.duckdb).expanduser().exists():
                _dump({"migration_id": MIGRATION_ID, "sql_sha256": file_sha256(MIGRATION_SQL),
                       "would_create": list(ALL_TABLES), "already_present": [],
                       "statements": len(load_ddl()), "destructive_statements": 0,
                       "note": "target file does not exist; a dry run creates nothing"}, args.out)
            else:
                conn = _connect(args.duckdb, read_only=args.dry_run)
                try:
                    _dump(migration_plan(conn) if args.dry_run else apply_migration(conn), args.out)
                finally:
                    conn.close()
        elif args.command == "readback":
            if bool(args.duckdb) == bool(args.service_url):
                raise Refusal("give exactly one of --duckdb or --service-url")
            if args.duckdb:
                conn = _connect(args.duckdb, read_only=True)
                try:
                    _dump(readback_report(local_query(conn), args.run_id, source="local"), args.out)
                finally:
                    conn.close()
            else:
                service = Service(args.service_url, args.token_env)
                _dump(readback_report(service.query, args.run_id, source="service"), args.out)
        elif args.command == "reconcile":
            conn = _connect(args.duckdb, read_only=True)
            try:
                receipts = [json.loads(p.read_text(encoding="utf-8")) for p in args.receipts]
                _dump(reconcile(conn, args.run_id, receipts or None), args.out)
            finally:
                conn.close()
        elif args.command == "compare-readback":
            result = compare_readback(json.loads(args.local.read_text()), json.loads(args.remote.read_text()))
            _dump(result, None)
            return 0 if result["agree"] else 3
        elif args.command == "snapshot":
            manifest = snapshot(Path(args.source), Path(args.out_dir), tag=args.tag, repo=args.repo,
                                phase=args.phase, zstd_bin=args.zstd_bin, level=args.level)
            _dump(manifest, None)
            if args.publish:
                for command in manifest["publish_commands"]:
                    subprocess.run(command, shell=True, check=True, cwd=args.out_dir)
        elif args.command == "verify-snapshot":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            _dump(verify_snapshot_asset(args.asset, manifest, zstd_bin=args.zstd_bin, work_dir=args.work_dir), None)
    except Refusal as exc:
        sys.stderr.write(f"REFUSED: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
