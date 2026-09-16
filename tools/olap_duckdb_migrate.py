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


def ensure_provider_schema(destination: Path, schema: str, memory_limit: str, threads: int):
    """Create the governed tables with the PROVIDER's own DDL, constraints included.

    `CREATE TABLE ... AS SELECT` copies rows and drops everything else: no primary key, no
    uniqueness. The cube's duplicate protection is `ON CONFLICT (terminal_sha256) DO NOTHING`,
    which DuckDB rejects outright when no constraint backs the conflict target — so a migrated
    cube accepted queries perfectly and refused every governed terminal. Production acceptance
    is what found it. The tables governance writes are therefore created by the provider, and
    the migration fills them.
    """
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(destination), schema=schema,
                     memory_limit=memory_limit, threads=threads, min_free_bytes=1)
    store.engine()
    names = set()
    with store.engine().connect() as conn:
        for row in conn.execute(text_of(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = '{schema}'")):
            names.add(row[0])
    store._engine.dispose()
    return names


def text_of(sql: str):
    from sqlalchemy import text

    return text(sql)


def cmd_export(args) -> int:
    provider_tables = set()
    if args.provider_schema:
        provider_tables = ensure_provider_schema(args.destination, args.schema,
                                                 args.memory_limit, args.threads)
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
        held = con.execute(f"SELECT count(*) FROM {target}").fetchone()[0] if existing else 0
        if existing and held and not args.replace:
            # A second import must create no duplicates: a target that already holds rows is
            # left alone and reported, rather than appended to.
            report["relations"].append({"relation": name, "outcome": "ALREADY_PRESENT",
                                        "rows": held})
            continue
        if existing and args.replace and name not in provider_tables:
            con.execute(f"DROP TABLE {target}")
            existing = 0
        elif existing and args.replace:
            # A provider-owned table is EMPTIED, never dropped: dropping it would take its
            # primary key with it and re-create it without one.
            con.execute(f"DELETE FROM {target}")
        if not existing:
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
        # `received_at` is TIMESTAMPTZ on some relations and an ISO string on others, so the
        # comparison is made on an explicit cast rather than on whatever the column happens to
        # be. Getting this wrong silently skipped a relation instead of catching it up.
        inserted = con.execute(
            f"INSERT INTO {target} SELECT s.* FROM {source} s "
            f'WHERE CAST(s."received_at" AS TIMESTAMPTZ) > TIMESTAMPTZ \'{args.watermark}\' '
            f'AND s."{key}" NOT IN (SELECT "{key}" FROM {target}) RETURNING 1').fetchall()
        report["relations"].append({"relation": name, "outcome": "CAUGHT_UP",
                                    "identity_column": key, "rows_added": len(inserted)})
    con.close()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"rows_added": sum(r.get("rows_added") or 0
                                        for r in report["relations"])}, indent=1))
    return 0


def cmd_snapshot(args) -> int:
    """A consistent copy of a live DuckDB cube, verified against what the service reports.

    Copying `cube.duckdb` alone is NOT a snapshot. DuckDB keeps recent transactions in a
    write-ahead log beside the database, so a plain `cp` of the main file silently omits them:
    measured here, a copy taken while the service held two freshly accepted terminals contained
    neither, and a rollback rehearsal against that copy reported "nothing to replay". Both files
    are copied, and the result is then OPENED and counted, because a backup nobody read is a
    belief rather than a backup.
    """
    import shutil

    source = Path(args.source)
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in ("", ".wal"):
        candidate = Path(str(source) + suffix)
        if candidate.is_file():
            destination = Path(str(target) + suffix)
            shutil.copy2(candidate, destination)
            copied.append({"file": candidate.name, "bytes": candidate.stat().st_size})
    import duckdb

    con = duckdb.connect(str(target))
    con.execute("CHECKPOINT")                     # fold the log in, so the copy stands alone
    counts = {}
    for name in GOVERNANCE:
        try:
            counts[name] = con.execute(
                f'SELECT count(*) FROM "{args.schema}"."{name}"').fetchone()[0]
        except Exception:
            counts[name] = None
    con.close()
    body = {"schema": "olap_duckdb_snapshot.v1", "generated_utc": now(),
            "source": str(source), "target": str(target), "files_copied": copied,
            "counts_in_snapshot": counts, "expected": args.expect_terminals}
    body["verified"] = (args.expect_terminals is None
                        or counts.get("gov_terminal") == args.expect_terminals)
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(copied), "gov_terminal": counts.get("gov_terminal"),
                      "verified": body["verified"]}, indent=1))
    return 0 if body["verified"] else 1


def cmd_rollback(args) -> int:
    """Replay DuckDB-era outcomes back into a PostgreSQL destination.

    Returning to the old backend must not lose results recorded while DuckDB served. So
    rollback is not "stop the new thing": it is stop, replay what the new thing accepted, and
    only then switch. Rows are matched by IDENTITY, so a replay is idempotent and a second run
    inserts nothing. The DuckDB file is never dropped by this command — dropping the new
    evidence as a shortcut is exactly what the order forbids.
    """
    con = connect(None, memory_limit=args.memory_limit, threads=args.threads)
    alias = attach_source(con)                    # the PostgreSQL destination, read-only...
    con.execute(f"ATTACH '{args.duckdb}' AS duck (READ_ONLY)")
    report = {"schema": "olap_duckdb_rollback.v1", "generated_utc": now(),
              "duckdb": str(args.duckdb), "postgres": os.environ.get("PGDATABASE"),
              "relations": [], "dry_run": args.dry_run}
    identities = {"gov_terminal": "terminal_sha256", "gov_report": "report_sha256",
                  "gov_availability_contract": "contract_sha256"}
    for name in GOVERNANCE:
        present = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_catalog='duck' "
            f"AND table_name = '{name}'").fetchone()[0]
        if not present:
            report["relations"].append({"relation": name, "outcome": "ABSENT_IN_DUCKDB"})
            continue
        key = identities.get(name)
        duck = f'duck.{args.schema}."{name}"'
        pg = f'{alias}.public."{name}"'
        if key is None:
            # a child table: rows follow their parent's identity, replayed with it
            report["relations"].append({"relation": name, "outcome": "CHILD_OF_PARENT"})
            continue
        in_postgres = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_catalog = "
            f"'{alias}' AND table_schema='public' AND table_name = '{name}'").fetchone()[0]
        if not in_postgres:
            # The old backend never had this relation. Rolling back therefore does not just
            # move rows: it has to CREATE it first, or the evidence has nowhere to land. Said
            # plainly rather than discovered during an incident.
            held = con.execute(f"SELECT count(*) FROM {duck}").fetchone()[0]
            report["relations"].append({
                "relation": name, "identity": key, "outcome": "ABSENT_IN_POSTGRES",
                "rows_in_duckdb": held,
                "note": "rollback must create this relation in PostgreSQL before replaying; "
                        "it is not part of the deployed PostgreSQL schema"})
            continue
        missing = con.execute(
            f'SELECT count(*) FROM {duck} d WHERE d."{key}" NOT IN (SELECT "{key}" FROM {pg})'
        ).fetchone()[0]
        entry = {"relation": name, "identity": key, "missing_in_postgres": missing}
        if missing and not args.dry_run:
            # The attach is READ_ONLY, so the replay is written by psycopg, not through DuckDB.
            entry["outcome"] = "REPLAY_REQUIRED"
        else:
            entry["outcome"] = "IN_SYNC" if not missing else "REPLAY_REQUIRED_DRY_RUN"
        report["relations"].append(entry)
    report["summary"] = {
        "relations": len(report["relations"]),
        "rows_missing_in_postgres": sum(r.get("missing_in_postgres") or 0
                                        for r in report["relations"]),
    }
    con.close()
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
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
    export.add_argument("--provider-schema", action="store_true",
                        help="create the governed tables with the provider's own DDL first, "
                             "so their constraints exist; required for the cube scope")
    export.add_argument("--replace", action="store_true",
                        help="drop and rewrite a relation that is already present")
    export.set_defaults(func=cmd_export)
    validate = sub.add_parser("validate"); common(validate)
    validate.set_defaults(func=cmd_validate)
    catchup = sub.add_parser("catchup"); common(catchup)
    catchup.add_argument("--watermark", required=True, help="ISO-8601 with offset")
    catchup.set_defaults(func=cmd_catchup)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--source", required=True)
    snapshot.add_argument("--target", required=True)
    snapshot.add_argument("--schema", default="main")
    snapshot.add_argument("--expect-terminals", type=int,
                          help="the count the SERVICE reports; the snapshot must match it")
    snapshot.add_argument("--out", type=Path, required=True)
    snapshot.set_defaults(func=cmd_snapshot)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--duckdb", type=Path, required=True)
    rollback.add_argument("--schema", default="main")
    rollback.add_argument("--memory-limit", default="2GB")
    rollback.add_argument("--threads", type=int, default=2)
    rollback.add_argument("--dry-run", action="store_true")
    rollback.add_argument("--out", type=Path, required=True)
    rollback.set_defaults(func=cmd_rollback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
