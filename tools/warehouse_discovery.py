#!/usr/bin/env python3
"""Who actually writes and reads the OLAP warehouse, and what is in it.

D0 of `docs/handoffs/MUSASHI_DUCKDB_WAREHOUSE_MIGRATION_ORDER_2026_09_16.md`:

    "Inventory actual warehouse writers, readers, loaders, outboxes, ETLs, Metabase
     connections and DOIN result routes. Distinguish deployed calls from unused code. Record
     concrete storage ownership before making changes. Existing warehouse tables include more
     than gov_*: enumerate df_*, dimensions, performance facts and receipts as applicable."

Two things this refuses to conflate:

* **a table that exists** and **a table anything writes**. Both are recorded, separately;
* **code that mentions a table** and **code a running unit executes**. A grep finds the first;
  only the systemd units, their `ExecStart` and the service configurations say the second.

Read-only throughout: it runs `SELECT`s and reads files. It never writes to the cube.

usage:
  warehouse_discovery.py --out INVENTORY.json [--repo DIR ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
UNITS = HOME / ".config" / "systemd" / "user"


def psql(query: str) -> list[str]:
    out = subprocess.run(["psql", "-At", "-F", "|", "-c", query],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.strip().split("\n") if line]


def tables() -> list[dict]:
    rows = psql(
        "SELECT c.relname, c.relkind, c.reltuples::bigint, "
        "pg_total_relation_size(c.oid) FROM pg_class c JOIN pg_namespace n "
        "ON n.oid = c.relnamespace WHERE n.nspname = 'public' "
        "AND c.relkind IN ('r','v','m') ORDER BY c.relname")
    out = []
    for row in rows:
        name, kind, estimate, size = row.split("|")
        entry = {"name": name, "kind": {"r": "table", "v": "view", "m": "matview"}[kind],
                 "estimated_rows": int(estimate), "bytes": int(size)}
        if entry["kind"] == "table":
            exact = psql(f'SELECT count(*) FROM public."{name}"')
            entry["rows"] = int(exact[0]) if exact else None
        out.append(entry)
    return out


def family(name: str) -> str:
    for prefix, label in (("gov_", "governance"), ("df_", "data_foundation"),
                          ("dim_", "dimension"), ("fact_", "fact"),
                          ("bridge_", "bridge"), ("v_", "view")):
        if name.startswith(prefix):
            return label
    return "other"


def deployed_units() -> list[dict]:
    """The units that exist, whether they are enabled, and what they execute."""
    out = []
    for path in sorted(UNITS.glob("crispdm-*.service")):
        text = path.read_text(encoding="utf-8", errors="replace")
        active = subprocess.run(["systemctl", "--user", "is-active", path.name],
                                capture_output=True, text=True).stdout.strip()
        enabled = subprocess.run(["systemctl", "--user", "is-enabled", path.name],
                                 capture_output=True, text=True).stdout.strip()
        exec_start = " ".join(
            line.split("=", 1)[1].strip().rstrip("\\").strip()
            for line in text.splitlines() if line.startswith("ExecStart"))
        env_file = next((line.split("=", 1)[1].strip()
                         for line in text.splitlines()
                         if line.startswith("EnvironmentFile")), None)
        out.append({"unit": path.name, "active": active, "enabled": enabled,
                    "exec_start": exec_start, "environment_file": env_file,
                    "touches_postgres_env": bool(env_file and "PG" in
                                                 Path(os.path.expandvars(
                                                     env_file.replace("%h", str(HOME)))
                                                 ).read_text(errors="replace")
                                                 if env_file and Path(
                                                     env_file.replace("%h", str(HOME))
                                                 ).is_file() else False)})
    return out


def code_references(repos: list[Path], names: list[str]) -> dict:
    """Which files mention each relation. A mention is not a call: both are reported."""
    references = {}
    for name in names:
        hits = []
        for repo in repos:
            if not repo.is_dir():
                continue
            out = subprocess.run(
                ["git", "-C", str(repo), "grep", "-l", "-F", name, "--", "*.py", "*.sql"],
                capture_output=True, text=True)
            hits.extend(f"{repo.name}/{line}" for line in out.stdout.strip().split("\n")
                        if line)
        if hits:
            references[name] = sorted(hits)
    return references


def outboxes() -> list[dict]:
    out = []
    for root in (HOME / ".local/share/predictor/olap_outbox",
                 HOME / ".local/state/data-gov/terminal-outbox",
                 HOME / ".cache/data-gov-outbox"):
        entry = {"root": str(root), "exists": root.is_dir(), "states": {}}
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    entry["states"][child.name] = sum(1 for _ in child.iterdir())
        out.append(entry)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", action="append", type=Path, default=[])
    args = parser.parse_args(argv)

    relations = tables()
    for entry in relations:
        entry["family"] = family(entry["name"])
    names = [entry["name"] for entry in relations]

    body = {
        "schema": "warehouse_discovery.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"),
        "database": (psql("SELECT current_database()") or [None])[0],
        "size_bytes": int((psql("SELECT pg_database_size(current_database())") or [0])[0]),
        "relations": relations,
        "by_family": {label: sum(1 for entry in relations if entry["family"] == label)
                      for label in sorted({entry["family"] for entry in relations})},
        "rows_by_family": {
            label: sum(entry.get("rows") or 0 for entry in relations
                       if entry["family"] == label and entry["kind"] == "table")
            for label in sorted({entry["family"] for entry in relations})},
        "units": deployed_units(),
        "outboxes": outboxes(),
        "code_references": code_references(args.repo, names),
    }
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"database": body["database"], "relations": len(relations),
                      "by_family": body["by_family"],
                      "rows_by_family": body["rows_by_family"],
                      "size_gb": round(body["size_bytes"] / 1e9, 2)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
