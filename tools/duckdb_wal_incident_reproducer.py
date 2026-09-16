#!/usr/bin/env python3
"""Can the incident be reproduced from the explanation I gave for it?

F3 of `docs/handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md`:

    "Create a fresh, minimal disposable reproducer for the claimed schema/WAL failure. If it
     cannot reproduce, label the root cause a hypothesis and the checkpoint change a mitigation.
     Verify abrupt interruption between schema creation, envelope commit, checkpoint and
     response; reopen and reconcile accepted outcomes."

I said the cause was creating a schema DURING a write, leaving DDL in the write-ahead log that
DuckDB then could not replay. That is an explanation, and an explanation is not evidence. This
tries to produce the failure on disposable databases, in the four arrangements that differ in
WHERE the schema is created and WHETHER the process is interrupted before it can checkpoint.

The production write-ahead log is never opened, replayed or read here. Nothing in this file
touches a path under the deployed cube.

Each case reports whether the database REOPENS, and if not, the exact error.

usage:
  duckdb_wal_incident_reproducer.py --work DIR --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

#: A child that writes and is then killed without a chance to close or checkpoint. Abrupt
#: termination is the point: a clean close checkpoints, and a clean close is not what happened.
WRITER = '''
import duckdb, os, sys
path, mode = sys.argv[1], sys.argv[2]
con = duckdb.connect(path)
if mode in ("schema_at_write", "both"):
    con.execute("CREATE SCHEMA IF NOT EXISTS public")
con.execute("CREATE TABLE IF NOT EXISTS public.thing (id INTEGER, v VARCHAR)")
con.execute("INSERT INTO public.thing VALUES (1, 'a')")
if mode == "checkpointed":
    con.execute("CHECKPOINT")
sys.stdout.write("written\\n"); sys.stdout.flush()
os._exit(9)          # abrupt: no close, no checkpoint, no flush of anything DuckDB defers
'''


def prepare(path: Path, *, schema_at_startup: bool) -> None:
    import duckdb

    con = duckdb.connect(str(path))
    if schema_at_startup:
        con.execute("CREATE SCHEMA IF NOT EXISTS public")
        con.execute("CHECKPOINT")
    con.close()


def reopen(path: Path) -> dict:
    import duckdb

    try:
        con = duckdb.connect(str(path), read_only=True)
        rows = con.execute("SELECT count(*) FROM public.thing").fetchone()[0]
        con.close()
        return {"reopens": True, "rows_visible": rows}
    except Exception as exc:
        return {"reopens": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def case(work: Path, name: str, *, schema_at_startup: bool, mode: str) -> dict:
    path = work / f"{name}.duckdb"
    for suffix in ("", ".wal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    prepare(path, schema_at_startup=schema_at_startup)
    script = work / "writer.py"
    script.write_text(WRITER, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(path), mode],
                            capture_output=True, text=True, timeout=120)
    wal = Path(str(path) + ".wal")
    return {"case": name, "schema_at_startup": schema_at_startup, "writer_mode": mode,
            "writer_exit": result.returncode,
            "wal_present_after_kill": wal.is_file(),
            "wal_bytes": wal.stat().st_size if wal.is_file() else 0,
            **reopen(path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True,
                        help="a DISPOSABLE directory; nothing under the deployed cube")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if "crispdm-duckdb/prod" in str(work):
        raise SystemExit("refusing to run inside the production cube directory")

    cases = [
        # the arrangement I blamed: the schema is created inside the write, then the process
        # dies before it can checkpoint
        case(work, "schema_at_write_then_killed", schema_at_startup=False,
             mode="schema_at_write"),
        # the mitigation: the schema already exists and is checkpointed; the write still dies
        case(work, "schema_at_startup_then_killed", schema_at_startup=True, mode="plain"),
        # a control: no schema creation anywhere, write then killed
        case(work, "no_schema_creation_then_killed", schema_at_startup=True, mode="plain"),
        # a control: the write checkpoints itself before dying
        case(work, "checkpointed_then_killed", schema_at_startup=True, mode="checkpointed"),
    ]
    blamed = cases[0]
    mitigated = cases[1]
    verdict = {
        "reproduced_the_failure": not blamed["reopens"],
        "mitigation_survives": mitigated["reopens"],
        "conclusion": (
            "ROOT_CAUSE_CONFIRMED" if not blamed["reopens"] and mitigated["reopens"]
            else "NOT_REPRODUCED_ROOT_CAUSE_REMAINS_A_HYPOTHESIS"),
    }
    body = {"schema": "duckdb_wal_incident_reproducer.v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "duckdb_version": __import__("duckdb").__version__,
            "work": str(work), "cases": cases, "verdict": verdict,
            "note": ("The production write-ahead log was never opened, replayed or read. These "
                     "are fresh disposable databases.")}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict,
                      "cases": [{k: c[k] for k in ("case", "reopens")} for c in cases]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
