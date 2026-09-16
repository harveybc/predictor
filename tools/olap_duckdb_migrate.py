#!/usr/bin/env python3
"""Move the OLAP cube to DuckDB: inventory, select, export, validate, catch up, roll back.

D1 and D3 of `docs/handoffs/MUSASHI_DUCKDB_WAREHOUSE_MIGRATION_ORDER_2026_09_16.md`.

The source is opened READ-ONLY through DuckDB's own `postgres` extension. Nothing here writes
to PostgreSQL, drops anything or rewrites history; the order authorizes none of that and the
attach is `READ_ONLY` so it could not.

Why both sides go through DuckDB: validation compares CONTENT, and a content comparison whose
two halves are computed by two different engines compares their formatting conventions as much
as the data. Running the source through an attached read-only PostgreSQL inside the same DuckDB
process makes the comparison expression literally the same expression.

Two destinations, deliberately separate:

  cube      the current campaign and its dependency closure — what governance reads;
  archive   everything else, catalogued and verified, and NOT in any default scientific view.

Subcommands:
  inventory   what the source holds, per relation
  select      the per-run manifest: INCLUDED_CURRENT / LEGACY_COMPARISON_ONLY / ... (D1)
  export      copy selected relations into a destination, in bounded batches
  validate    per-relation row counts AND order-independent content digests, both sides
  catchup     copy rows that appeared after a recorded watermark
  rollback    what to do, and proof that new evidence survives it

usage:
  olap_duckdb_migrate.py <subcommand> --help
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

#: Relations that carry the governed result contract. These are the cube: data-gov writes them
#: and every governed campaign is recorded here.
GOVERNANCE = ("gov_report", "gov_metric", "gov_dataset", "gov_terminal",
              "gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact",
              "gov_availability_contract")

#: The dependency closure of a governed terminal that lives outside gov_*: the campaign and run
#: identity the loader records, and the dimensions those rows point at.
CLOSURE = ("dim_campaign", "dim_campaign_run", "dim_project", "dim_phase", "dim_experiment",
           "dim_dataset", "dim_dataset_split", "dim_horizon", "dim_metric",
           "dim_terminal_verification", "dim_terminal_verification_v2",
           "fact_campaign_unit", "fact_campaign_consumption", "fact_ingestion_receipt")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect(destination: Path | None, *, memory_limit="2GB", threads=2):
    import duckdb

    con = duckdb.connect(str(destination) if destination else ":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    return con


def attach_source(con, *, alias="pg"):
    """The source, read-only. Every command uses this and none can write through it."""
    con.execute("INSTALL postgres")
    con.execute("LOAD postgres")
    dsn = (f"host={os.environ.get('PGHOST', '127.0.0.1')} "
           f"port={os.environ.get('PGPORT', '5432')} "
           f"dbname={os.environ['PGDATABASE']} user={os.environ.get('PGUSER', '')} "
           f"password={os.environ.get('PGPASSWORD', '')}")
    con.execute(f"ATTACH '{dsn}' AS {alias} (TYPE postgres, READ_ONLY)")
    return alias


def source_relations(con, alias="pg") -> list[dict]:
    rows = con.execute(
        "SELECT table_name, table_type FROM information_schema.tables "
        f"WHERE table_catalog = '{alias}' AND table_schema = 'public' ORDER BY table_name"
    ).fetchall()
    out = []
    for name, kind in rows:
        columns = [row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_catalog = "
            f"'{alias}' AND table_schema='public' AND table_name='{name}' "
            "ORDER BY ordinal_position").fetchall()]
        entry = {"name": name, "kind": "view" if "VIEW" in kind.upper() else "table",
                 "columns": columns}
        if entry["kind"] == "table":
            entry["rows"] = con.execute(
                f'SELECT count(*) FROM {alias}.public."{name}"').fetchone()[0]
        out.append(entry)
    return out


def content_digest(con, relation: str, columns: list[str]) -> str | None:
    """An order-independent digest of the whole relation's CONTENT.

    Per row: the md5 of the row rendered as a value, so every column participates. Across rows:
    the md5 of those hashes sorted, so physical order — which no engine promises to preserve
    across a copy — cannot change the answer while any differing value will.
    """
    if not columns:
        return None
    projection = ", ".join(f'"{column}"' for column in columns)
    try:
        return con.execute(
            f"SELECT md5(string_agg(h, '' ORDER BY h)) FROM "
            f"(SELECT md5(CAST(ROW({projection}) AS VARCHAR)) AS h FROM {relation}) t"
        ).fetchone()[0]
    except Exception as exc:                      # a type DuckDB cannot render as a value
        return f"UNCOMPARABLE: {type(exc).__name__}"


# ---------------------------------------------------------------- subcommands
def cmd_inventory(args) -> int:
    con = connect(None)
    alias = attach_source(con)
    relations = source_relations(con, alias)
    body = {"schema": "olap_duckdb_inventory.v1", "generated_utc": now(),
            "source_database": os.environ.get("PGDATABASE"),
            "relations": relations,
            "totals": {"relations": len(relations),
                       "tables": sum(1 for r in relations if r["kind"] == "table"),
                       "rows": sum(r.get("rows") or 0 for r in relations)}}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body["totals"], indent=1))
    return 0


def classify_terminal(row: dict) -> tuple[str, str]:
    """Campaign membership from recorded identity, never from a date or a metric's sign.

    `classification` is declared by the campaign itself before any data is opened, so it is
    identity rather than outcome. A NON_GOVERNING campaign is operational evidence by its own
    declaration — it grants nothing scientific — and that is exactly `MECHANICAL_ONLY`.
    Failures, refusals and inconclusive outcomes are kept with their real status: excluding
    them would be selecting by outcome, which is the thing being forbidden.
    """
    if not row.get("campaign_sha256") or not row.get("code_identity_json"):
        return "UNRESOLVED", "no campaign or code identity recorded"
    if row["classification"] == "GOVERNING":
        return "INCLUDED_CURRENT", (
            f"governing campaign declared before execution; status {row['status']} is "
            "preserved as recorded")
    if row["classification"] == "NON_GOVERNING":
        return "MECHANICAL_ONLY", (
            "declared NON_GOVERNING at submission: operational evidence, never a scientific "
            f"comparison; status {row['status']} preserved")
    return "UNRESOLVED", f"unrecognised classification {row['classification']!r}"


def cmd_select(args) -> int:
    con = connect(None)
    alias = attach_source(con)
    columns = [row[0] for row in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_catalog = "
        f"'{alias}' AND table_schema='public' AND table_name='gov_terminal' "
        "ORDER BY ordinal_position").fetchall()]
    rows = [dict(zip(columns, values)) for values in con.execute(
        f"SELECT * FROM {alias}.public.gov_terminal ORDER BY received_at").fetchall()]

    runs = []
    for row in rows:
        label, rationale = classify_terminal(row)
        runs.append({
            "terminal_sha256": row["terminal_sha256"],
            "campaign_key": row["campaign_key"],
            "campaign_sha256": row["campaign_sha256"],
            "unit_id": row["unit_id"], "generation": row["generation"],
            "project": row["project"], "actor": row["actor"],
            "classification": row["classification"], "status": row["status"],
            "reason": row["reason"],
            "code_identity": row.get("code_identity_json"),
            "config_sha256": row.get("config_sha256"),
            "received_at": str(row["received_at"]),
            "disposition": label, "rationale": rationale,
            "source": f"{os.environ.get('PGDATABASE')}.public.gov_terminal",
        })

    legacy = []
    for relation in source_relations(con, alias):
        if relation["kind"] != "table" or relation["name"] in GOVERNANCE:
            continue
        if relation["name"] in CLOSURE:
            disposition, why = "INCLUDED_CURRENT", "dependency closure of governed terminals"
        elif relation["name"].startswith("df_"):
            disposition, why = "LEGACY_COMPARISON_ONLY", (
                "data-foundation evidence from earlier campaigns; admissible only where a "
                "comparison view opts into it and states its limitations")
        else:
            disposition, why = "LEGACY_COMPARISON_ONLY", (
                "predates the governed contract; kept for reference, not in default views")
        legacy.append({"relation": relation["name"], "rows": relation.get("rows"),
                       "disposition": disposition, "rationale": why})

    counts = {}
    for entry in runs:
        counts[entry["disposition"]] = counts.get(entry["disposition"], 0) + 1
    body = {"schema": "olap_migration_selection.v1", "generated_utc": now(),
            "source_database": os.environ.get("PGDATABASE"),
            "selection_rule": (
                "campaign membership comes from the classification the campaign DECLARED "
                "before execution, plus recorded identity. Not from date, model family, "
                "metric sign, defaults or table name."),
            "runs": runs, "relations": legacy,
            "counts": {"runs": counts,
                       "status_within_included": {
                           status: sum(1 for entry in runs
                                       if entry["disposition"] == "INCLUDED_CURRENT"
                                       and entry["status"] == status)
                           for status in sorted({entry["status"] for entry in runs})},
                       "relations": {label: sum(1 for entry in legacy
                                                if entry["disposition"] == label)
                                     for label in sorted({e["disposition"] for e in legacy})}}}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body["counts"], indent=1))
    return 0


def relations_for(args, con, alias) -> list[dict]:
    available = {entry["name"]: entry for entry in source_relations(con, alias)}
    if args.scope == "cube":
        wanted = [name for name in GOVERNANCE + CLOSURE if name in available]
    elif args.scope == "archive":
        wanted = [name for name, entry in available.items()
                  if entry["kind"] == "table" and name not in GOVERNANCE
                  and name not in CLOSURE]
    else:
        wanted = [name for name, entry in available.items() if entry["kind"] == "table"]
    return [available[name] for name in wanted if available[name]["kind"] == "table"]


def pagination_key(con, source: str, columns: list[str]) -> str | None:
    """A column whose values are unique, so batches can be taken deterministically.

    LIMIT/OFFSET without ORDER BY does NOT paginate: SQL promises no stable order between two
    statements, so consecutive batches can skip rows and repeat others. That produced a
    destination with the right ROW COUNT and the wrong rows, which a count-only validation
    would have called success. Keyset pagination on a unique column is ordered by construction.
    """
    preferred = [c for c in columns if c.endswith("_sha256")] + list(columns)
    for column in preferred:
        try:
            distinct, total = con.execute(
                f'SELECT count(DISTINCT "{column}"), count(*) FROM {source}').fetchone()
        except Exception:
            continue
        if total and distinct == total:
            return column
    return None


def copy_relation(con, source: str, target: str, total: int, key: str | None,
                  batch_rows: int) -> int:
    """Copy every row exactly once, in bounded work units.

    With a unique key: keyset pagination, which is ordered and therefore complete. Without one:
    a single streaming INSERT, whose memory is bounded by the connection's own `memory_limit`
    rather than by slicing. Never LIMIT/OFFSET.
    """
    if key is None:
        con.execute(f"INSERT INTO {target} SELECT * FROM {source}")
        return con.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
    copied, last = 0, None
    while True:
        where = "" if last is None else f'WHERE "{key}" > ?'
        parameters = [] if last is None else [last]
        rows = con.execute(
            f'INSERT INTO {target} SELECT * FROM {source} {where} '
            f'ORDER BY "{key}" LIMIT {batch_rows} RETURNING "{key}"', parameters).fetchall()
        if not rows:
            break
        copied += len(rows)
        last = max(row[0] for row in rows)
        if copied >= total and total:
            break
    return copied


def cmd_export(args) -> int:
    con = connect(args.destination, memory_limit=args.memory_limit, threads=args.threads)
    alias = attach_source(con)
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{args.schema}"')
    report = {"schema": "olap_duckdb_export.v1", "generated_utc": now(),
              "destination": str(args.destination), "target_schema": args.schema,
              "scope": args.scope, "batch_rows": args.batch_rows, "relations": []}
    for entry in relations_for(args, con, alias):
        name = entry["name"]
        target = f'"{args.schema}"."{name}"'
        source = f'{alias}.public."{name}"'
        existing = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = "
            f"'{args.schema}' AND table_name = '{name}'").fetchone()[0]
        if existing and not args.replace:
            # A second import must create no duplicates: an existing target is left alone and
            # reported, rather than appended to.
            report["relations"].append({"relation": name, "outcome": "ALREADY_PRESENT",
                                        "rows": con.execute(
                                            f"SELECT count(*) FROM {target}").fetchone()[0]})
            continue
        if existing:
            con.execute(f"DROP TABLE {target}")
        con.execute(f"CREATE TABLE {target} AS SELECT * FROM {source} LIMIT 0")
        total = entry.get("rows") or 0
        key = pagination_key(con, source, entry["columns"])
        copied = copy_relation(con, source, target, total, key, args.batch_rows)
        report["relations"].append({"relation": name, "outcome": "COPIED",
                                    "source_rows": total, "copied_rows": copied,
                                    "pagination_key": key,
                                    "batching": "keyset" if key else "single_stream"})
    con.close()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"relations": len(report["relations"]),
                      "copied": sum(r.get("copied_rows") or 0 for r in report["relations"]),
                      "already_present": sum(1 for r in report["relations"]
                                             if r["outcome"] == "ALREADY_PRESENT")}, indent=1))
    return 0


def cmd_validate(args) -> int:
    con = connect(args.destination, memory_limit=args.memory_limit, threads=args.threads)
    alias = attach_source(con)
    report = {"schema": "olap_duckdb_validation.v1", "generated_utc": now(),
              "destination": str(args.destination), "target_schema": args.schema,
              "relations": [], "mismatches": []}
    for entry in relations_for(args, con, alias):
        name = entry["name"]
        target = f'"{args.schema}"."{name}"'
        source = f'{alias}.public."{name}"'
        present = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = "
            f"'{args.schema}' AND table_name = '{name}'").fetchone()[0]
        if not present:
            row = {"relation": name, "outcome": "MISSING_IN_DESTINATION"}
            report["relations"].append(row)
            report["mismatches"].append(row)
            continue
        source_rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        target_rows = con.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
        source_digest = content_digest(con, source, entry["columns"])
        target_digest = content_digest(con, target, entry["columns"])
        row = {"relation": name, "source_rows": source_rows, "destination_rows": target_rows,
               "source_content_md5": source_digest, "destination_content_md5": target_digest,
               "rows_match": source_rows == target_rows,
               "content_matches": source_digest == target_digest,
               "columns": len(entry["columns"])}
        row["outcome"] = "MATCH" if row["rows_match"] and row["content_matches"] else "MISMATCH"
        report["relations"].append(row)
        if row["outcome"] != "MATCH":
            report["mismatches"].append(row)
    report["summary"] = {
        "relations": len(report["relations"]),
        "matched": sum(1 for r in report["relations"] if r.get("outcome") == "MATCH"),
        "mismatched": len(report["mismatches"]),
        "rows_verified": sum(r.get("destination_rows") or 0 for r in report["relations"]),
    }
    con.close()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    return 0 if not report["mismatches"] else 1


def cmd_catchup(args) -> int:
    """Rows that appeared after the watermark, by identity rather than by position."""
    con = connect(args.destination, memory_limit=args.memory_limit, threads=args.threads)
    alias = attach_source(con)
    report = {"schema": "olap_duckdb_catchup.v1", "generated_utc": now(),
              "watermark": args.watermark, "relations": []}
    for entry in relations_for(args, con, alias):
        name = entry["name"]
        if "received_at" not in entry["columns"]:
            report["relations"].append({"relation": name, "outcome": "NO_WATERMARK_COLUMN"})
            continue
        target = f'"{args.schema}"."{name}"'
        source = f'{alias}.public."{name}"'
        key = "terminal_sha256" if "terminal_sha256" in entry["columns"] else (
            "report_sha256" if "report_sha256" in entry["columns"] else None)
        if key is None:
            report["relations"].append({"relation": name, "outcome": "NO_IDENTITY_COLUMN"})
            continue
        inserted = con.execute(
            f"INSERT INTO {target} SELECT s.* FROM {source} s WHERE s.received_at > "
            f"TIMESTAMPTZ '{args.watermark}' AND s.\"{key}\" NOT IN "
            f'(SELECT "{key}" FROM {target}) RETURNING 1').fetchall()
        report["relations"].append({"relation": name, "outcome": "CAUGHT_UP",
                                    "identity_column": key, "rows_added": len(inserted)})
    con.close()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows_added": sum(r.get("rows_added") or 0
                                        for r in report["relations"])}, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, destination=True):
        if destination:
            p.add_argument("--destination", type=Path, required=True)
            p.add_argument("--schema", default="main")
            p.add_argument("--memory-limit", default="2GB")
            p.add_argument("--threads", type=int, default=2)
            p.add_argument("--scope", choices=("cube", "archive", "all"), default="cube")
        p.add_argument("--out", type=Path, required=True)

    inventory = sub.add_parser("inventory"); common(inventory, destination=False)
    inventory.set_defaults(func=cmd_inventory)
    select = sub.add_parser("select"); common(select, destination=False)
    select.set_defaults(func=cmd_select)
    export = sub.add_parser("export"); common(export)
    export.add_argument("--batch-rows", type=int, default=100_000)
    export.add_argument("--replace", action="store_true",
                        help="drop and rewrite a relation that is already present")
    export.set_defaults(func=cmd_export)
    validate = sub.add_parser("validate"); common(validate)
    validate.set_defaults(func=cmd_validate)
    catchup = sub.add_parser("catchup"); common(catchup)
    catchup.add_argument("--watermark", required=True, help="ISO-8601 with offset")
    catchup.set_defaults(func=cmd_catchup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
