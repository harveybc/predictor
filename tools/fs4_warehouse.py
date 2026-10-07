#!/usr/bin/env python3
"""Phase-4 extractibility warehouse: additive migration, idempotent submit, readback, reconcile.

Plan: ``docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md``
(§5 automation, §6 FS4-06/07/08/09/13) and the order
``docs/handoffs/SATOSHI_FS4_EXECUTION_2026_10_07.md`` (step 3).

The storage semantics live in ONE place, ``predictor_olap_store.fs4_store`` (the packaged
backend the warehouse service loads); this module imports them, so the local file path used
by tests and the closure follower, and the service path behind ``/api/v2/fs4/*``, are the
same code. What this module adds:

* ``open_warehouse(path_or_url)`` — the follower's entry point (``tools/fs4_closure.py``): a
  DuckDB file (migration applied on open) or an ``http(s)://`` service URL (token from
  ``WAREHOUSE_TOKEN``, never an argument). Both return an object with
  ``submit_terminals(plan_sha256, terminals, host_role=None)``,
  ``read_terminals(plan_sha256, task_id=None, ...)``, ``readback_summary(plan_sha256)`` and
  ``reconcile(plan_sha256, expected=None, receipts=None)``.
* ``render-migration`` / ``migrate`` (with ``--dry-run``) / ``readback`` / ``reconcile`` CLI.

This module never opens the configured production cube. Host references are roles only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_SQL = REPO_ROOT / "olap" / "migrations" / "fs4" / "0001_fs4_extractibility_additive.sql"


def _import_core():
    """The packaged store when it carries fs4_store; otherwise this checkout's source tree. An
    installed older predictor_olap_store (without fs4_store) must not shadow the source tree."""
    try:
        from predictor_olap_store import fs4_store as core
        return core
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "olap" / "store" / "src"))
        for name in [n for n in sys.modules if n == "predictor_olap_store" or n.startswith("predictor_olap_store.")]:
            del sys.modules[name]
        from predictor_olap_store import fs4_store as core
        return core


core = _import_core()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_phase23_warehouse as _p23  # noqa: E402  (the transport with bounded retries)

Refusal = core.Refusal
MIGRATION_ID = core.MIGRATION_ID
TABLE = core.TABLE
ALL_RELATIONS = core.ALL_RELATIONS
ARMS = core.ARMS
canonical_bytes = core.canonical_bytes
digest = core.digest
task_id_of = core.task_id_of
terminals_digest = core.terminals_digest
prepare_terminal = core.prepare_terminal

SCHEMA_READBACK = "fs4_readback_report.v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return core.now()


# --------------------------------------------------------------------------- migration file
def load_ddl(path: Path = MIGRATION_SQL) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def write_migration_file(path: Path = MIGRATION_SQL) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(core.render_sql(), encoding="utf-8")
    return file_sha256(path)


def migration_file_matches_code(path: Path = MIGRATION_SQL) -> bool:
    norm = lambda s: " ".join(s.split())  # noqa: E731
    return [norm(s) for s in load_ddl(path)] == [norm(s) for s in core.ddl()]


def apply_migration(conn, path: Path = MIGRATION_SQL) -> dict:
    if path.exists() and not migration_file_matches_code(path):
        raise Refusal(f"{path.name} does not match fs4_store.ddl(); regenerate it with "
                      "`fs4_warehouse.py render-migration`")
    result = core.apply_migration(conn)
    result["sql_sha256"] = file_sha256(path) if path.exists() else None
    return result


def migration_plan(conn) -> dict:
    present = set(core.list_relations(conn))
    return {"migration_id": MIGRATION_ID,
            "sql_sha256": file_sha256(MIGRATION_SQL) if MIGRATION_SQL.exists() else None,
            "would_create": [t for t in ALL_RELATIONS if t not in present],
            "already_present": [t for t in ALL_RELATIONS if t in present],
            "statements": len(core.ddl()), "destructive_statements": 0}


# --------------------------------------------------------------------------- warehouses
class Warehouse:
    """A DuckDB file. The follower's interface plus readback helpers."""

    backend = "duckdb"

    def __init__(self, path: Path, *, read_only: bool = False, memory_limit: str = "256MB"):
        import duckdb
        self.path = Path(path).expanduser()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path), read_only=read_only,
                                   config={"memory_limit": memory_limit, "threads": 1})
        if not read_only:
            self.migration = apply_migration(self.conn)

    def submit_terminals(self, plan_sha256: str, terminals: list[dict], *, host_role: str | None = None) -> dict:
        return core.submit_terminals(self.conn, plan_sha256, terminals, host_role=host_role, backend=self.backend)

    def read_terminals(self, plan_sha256: str, *, task_id: str | None = None, population_id: str | None = None,
                       feature_id: str | None = None, fold_id: str | None = None, arm: str | None = None) -> list[dict]:
        return core.read_terminals(self.conn, plan_sha256, task_id=task_id, population_id=population_id,
                                   feature_id=feature_id, fold_id=fold_id, arm=arm)

    def readback_summary(self, plan_sha256: str) -> dict:
        n, d = self.conn.execute(core.stored_summary_sql(plan_sha256)).fetchone()
        return {"count": int(n), "terminals_sha256": d}

    def reconcile(self, plan_sha256: str, expected: dict | None = None, receipts: list[dict] | None = None) -> dict:
        return core.reconcile(self.conn, plan_sha256, expected, receipts, backend=self.backend)

    def verify_receipt(self, receipt: dict) -> dict:
        return core.verify_receipt(self.conn, receipt)

    def query(self, sql: str) -> list[dict]:
        cursor = self.conn.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()


class ServiceWarehouse(_p23.ServiceWarehouse):
    """The running warehouse through its ``/api/v2/fs4/*`` routes. Token from the environment;
    transport (bounded timeouts, retries after /healthz, 400/422 never retried) is the phase-2/3 one."""

    def __init__(self, url: str, token_env: str = "WAREHOUSE_TOKEN", **transport):
        try:
            super().__init__(url, token_env, **transport)
        except _p23.Refusal as exc:
            raise Refusal(str(exc)) from None

    def _call(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        try:
            return super()._call(method, path, body, params)
        except _p23.Refusal as exc:  # the host's 400/422 verdict, in this module's refusal class
            raise Refusal(str(exc)) from None

    def submit_terminals(self, plan_sha256: str, terminals: list[dict], *, host_role: str | None = None) -> dict:
        document: dict[str, Any] = {"plan_sha256": plan_sha256, "terminals": terminals}
        if host_role:
            document["host_role"] = host_role
        return self._call("POST", "/api/v2/fs4/terminals", document)

    def read_terminals(self, plan_sha256: str, *, task_id: str | None = None, population_id: str | None = None,
                       feature_id: str | None = None, fold_id: str | None = None, arm: str | None = None) -> list[dict]:
        out: list[dict] = []
        after = None
        while True:
            page = self._call("GET", "/api/v2/fs4/terminals",
                              params={"plan_sha256": plan_sha256, "task_id": task_id, "population_id": population_id,
                                      "feature_id": feature_id, "fold_id": fold_id, "arm": arm,
                                      "after": after, "limit": self.page})
            out.extend(page.get("terminals") or [])
            after = page.get("next_after")
            if not after:
                return out

    def readback_summary(self, plan_sha256: str) -> dict:
        rows = self.query(core.stored_summary_sql(plan_sha256))
        n = int(rows[0]["n"]) if rows else 0
        return {"count": n, "terminals_sha256": (rows[0]["terminals_sha256"] if n else core.EMPTY_DIGEST)}

    def reconcile(self, plan_sha256: str, expected: dict | None = None, receipts: list[dict] | None = None) -> dict:
        document: dict[str, Any] = {"plan_sha256": plan_sha256}
        if expected:
            document["expected"] = expected
        if receipts:
            document["receipts"] = receipts
        return self._call("POST", "/api/v2/fs4/reconcile", document)

    def verify_receipt(self, receipt: dict) -> dict:
        report = self.reconcile(receipt["plan_sha256"], receipts=[receipt])
        if report.get("receipts_verified") != 1:
            raise Refusal("the service did not verify the receipt")
        return {"accepted": True, "receipt_sha256": receipt["receipt_sha256"]}


def open_warehouse(path_or_url, *, token_env: str = "WAREHOUSE_TOKEN", page: int = 1000):
    """The follower's entry point: a DuckDB file path or an http(s) service URL."""
    target = str(path_or_url)
    if target.startswith(("http://", "https://")):
        return ServiceWarehouse(target, token_env, page=page)
    return Warehouse(Path(target))


# --------------------------------------------------------------------------- readback report
def readback_report(query, plan_sha256: str, *, source: str = "local") -> dict:
    """Counts and digests per population x arm x host role, plus triple agreement (closure input)."""
    pid = plan_sha256.replace("'", "''")
    groups = []
    for row in query(f"SELECT population_id, arm, coalesce(host_role, 'UNDECLARED') AS host_role, count(*) AS n,"
                     f" sha256(string_agg(terminal_sha256, '' ORDER BY terminal_sha256)) AS terminals_sha256"
                     f" FROM {TABLE} WHERE plan_sha256 = '{pid}' GROUP BY 1, 2, 3 ORDER BY 1, 2, 3 LIMIT 100000"):
        groups.append({"population_id": row["population_id"], "arm": row["arm"], "host_role": row["host_role"],
                       "n": int(row["n"]), "terminals_sha256": row["terminals_sha256"]})
    triples = query("SELECT count(*) AS triples, sum(CASE WHEN arms = 3 THEN 1 ELSE 0 END) AS complete,"
                    " sum(CASE WHEN rows_digests > 1 OR mask_digests > 1 OR input_digests > 1 OR population_ns > 1"
                    " OR naive_maes > 1 THEN 1 ELSE 0 END) AS inconsistent"
                    f" FROM fs4_extractibility_triples WHERE plan_sha256 = '{pid}' LIMIT 1")
    t = triples[0] if triples else {"triples": 0, "complete": 0, "inconsistent": 0}
    report = {"schema": SCHEMA_READBACK, "source": source, "plan_sha256": plan_sha256, "groups": groups,
              "total": sum(g["n"] for g in groups),
              "table_terminals_sha256": terminals_digest(g["terminals_sha256"] for g in groups),
              "by_population": _sum_by(groups, "population_id"), "by_arm": _sum_by(groups, "arm"),
              "by_host_role": _sum_by(groups, "host_role"),
              "triples": {"total": int(t["triples"] or 0), "complete": int(t["complete"] or 0),
                          "inconsistent": int(t["inconsistent"] or 0)},
              "generated_at": now()}
    report["report_sha256"] = digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def _sum_by(groups: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for g in groups:
        out[g[key]] = out.get(g[key], 0) + g["n"]
    return out


def compare_readback(local: dict, remote: dict) -> dict:
    left = {(g["population_id"], g["arm"], g["host_role"]): g for g in local.get("groups", [])}
    right = {(g["population_id"], g["arm"], g["host_role"]): g for g in remote.get("groups", [])}
    differences = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None or a["n"] != b["n"] or a["terminals_sha256"] != b["terminals_sha256"]:
            differences.append({"group": list(key),
                                "local": None if a is None else {"n": a["n"], "terminals_sha256": a["terminals_sha256"]},
                                "remote": None if b is None else {"n": b["n"], "terminals_sha256": b["terminals_sha256"]}})
    return {"agree": not differences, "differences": differences}


# --------------------------------------------------------------------------- CLI
def _emit(value: Any, path: Path | None = None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render-migration", help="write the migration .sql from the code (the single DDL source)")
    p = sub.add_parser("migrate", help="apply the additive migration to a DuckDB file (never the production cube)")
    p.add_argument("--duckdb", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("readback", help="counts and digests per population x arm x host role")
    p.add_argument("--warehouse", required=True, help="DuckDB file or service URL (token from WAREHOUSE_TOKEN)")
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("reconcile", help="stored counts against the controller's expected counts")
    p.add_argument("--warehouse", required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--expected", type=Path, help="JSON {total, by_population} (e.g. from fs4_closure.py expected)")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("compare-readback")
    p.add_argument("--local", required=True, type=Path)
    p.add_argument("--remote", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "render-migration":
            _emit({"path": str(MIGRATION_SQL), "sql_sha256": write_migration_file(), "migration_id": MIGRATION_ID})
        elif args.command == "migrate":
            import duckdb
            if args.dry_run:
                conn = duckdb.connect(str(args.duckdb), read_only=args.duckdb.exists())
                _emit({"dry_run": True, **migration_plan(conn)})
                conn.close()
            else:
                _emit(Warehouse(args.duckdb).migration)
        elif args.command == "readback":
            store = open_warehouse(args.warehouse)
            _emit(readback_report(store.query, args.plan_sha256,
                                  source="service" if store.backend == "service" else "local"), args.out)
            store.close()
        elif args.command == "reconcile":
            expected = json.loads(args.expected.read_text()) if args.expected else None
            store = open_warehouse(args.warehouse)
            _emit(store.reconcile(args.plan_sha256, expected), args.out)
            store.close()
        elif args.command == "compare-readback":
            _emit(compare_readback(json.loads(args.local.read_text()), json.loads(args.remote.read_text())))
    except Refusal as exc:
        print(json.dumps({"refused": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
