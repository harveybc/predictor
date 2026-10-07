#!/usr/bin/env python3
"""Rebuild the primary-key (ART) index of fs_phase23 fact tables in a local DuckDB warehouse file.

Incident 2026-10-06: after the follower had been SIGKILLed mid-transaction, the EURUSD store
refused every further submission with ``Duplicate key "row_identity_sha256: ..."`` although no
row with that identity existed (``SELECT`` returned 0): the ART index kept a key whose row had
been rolled back (DuckDB 1.5.6).  ``CHECKPOINT`` does not clear it.  Recreating the table from
its own rows rebuilds the index; rows, payloads and digests are unchanged, which this tool
verifies before and after (per-table row count and the sorted-row_sha256 digest).

Run only with the store's single writer stopped (one writer per DuckDB file).

    rebuild_duckdb_pk_index.py --warehouse FILE [--table NAME ...] [--memory-limit 512MB]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _core():
    try:
        from predictor_olap_store import fs_phase23_store as core
    except ImportError:
        sys.path.insert(0, str(REPO / "olap" / "store" / "src"))
        from predictor_olap_store import fs_phase23_store as core
    return core


def table_digest(con, table: str) -> dict:
    shas = sorted(r[0] for r in con.execute(f"SELECT row_sha256 FROM {table}").fetchall())
    return {"count": len(shas), "rows_sha256": hashlib.sha256("".join(shas).encode()).hexdigest()}


def rebuild(con, table: str, core) -> dict:
    cols = core.columns(table)
    before = table_digest(con, table)
    t0 = time.time()
    con.execute(f"CREATE TABLE _rebuild_{table} AS SELECT * FROM {table}")
    con.execute(f"DROP TABLE {table}")
    ddl = [stmt for stmt in core.ddl() if stmt.split("(")[0].rstrip().endswith(table)]
    if len(ddl) != 1:
        raise RuntimeError(f"cannot find exactly one DDL statement for {table}")
    con.execute(ddl[0])
    con.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _rebuild_{table}")
    con.execute(f"DROP TABLE _rebuild_{table}")
    after = table_digest(con, table)
    if before != after:
        raise RuntimeError(f"{table}: rows changed during rebuild {before} -> {after}")
    return {"table": table, **after, "seconds": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warehouse", required=True, type=Path)
    p.add_argument("--table", action="append", default=None)
    p.add_argument("--memory-limit", default="512MB")
    args = p.parse_args(argv)
    import duckdb
    core = _core()
    tables = args.table or list(core.FACT_TABLES)
    con = duckdb.connect(str(args.warehouse))
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    receipt = {"schema": "fs_phase23.pk_index_rebuild.v1", "warehouse": args.warehouse.name, "duckdb": duckdb.__version__, "tables": []}
    try:
        for table in tables:
            con.execute("BEGIN TRANSACTION")
            try:
                receipt["tables"].append(rebuild(con, table, core))
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        con.execute("CHECKPOINT")
    finally:
        con.close()
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
