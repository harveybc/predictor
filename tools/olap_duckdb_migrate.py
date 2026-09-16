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
import re
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


# ---------------------------------------------------------------- correctness primitives
#: The governed parent relations and the children whose rows belong to a parent's identity.
#: A terminal without its metrics, datasets and artifacts is not a restored outcome, so a
#: replay that moved only parents would be another report dressed as an action.
PARENTS = {"gov_terminal": "terminal_sha256", "gov_report": "report_sha256",
           "gov_availability_contract": "contract_sha256"}
CHILDREN = {"gov_terminal": ("gov_terminal_metric", "gov_terminal_dataset",
                             "gov_terminal_artifact"),
            "gov_report": ("gov_metric", "gov_dataset")}


def digests_agree(left, right) -> bool:
    """Two content digests agree only when both were COMPUTED and are equal.

    `content_digest` returns `UNCOMPARABLE: <error>` when a column cannot be rendered. Two
    relations that failed the same way produced identical strings, so `==` called them a match
    and validation passed on a comparison that never happened.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if left.startswith("UNCOMPARABLE") or right.startswith("UNCOMPARABLE"):
        return False
    return left == right


def import_state(source_rows: int, destination_rows: int) -> str:
    """Whether a destination relation is empty, partial, complete or wrong.

    An interrupted multi-batch import leaves rows behind. Treating any non-empty target as
    ALREADY_PRESENT stranded the remainder and reported success; a partial relation must be
    resumed, and one holding MORE rows than the source is a defect, not a completion.
    """
    if destination_rows == 0:
        return "EMPTY"
    if destination_rows > source_rows:
        return "OVERFILLED"
    return "COMPLETE" if destination_rows == source_rows else "PARTIAL"


#: A data-foundation run records its order item in `module` ("C140 coverage", "D0 contracts",
#: "D2 fresh confirmation …"). That token IS its campaign identity: `dim_campaign.run_id` does
#: not join to these rows at all, which is a fact measured rather than assumed, so using it
#: would have produced a NULL campaign for every one of them and archived the lot.
ORDER_ITEM = re.compile(r"^(?P<item>[A-Z]+[0-9]+(?:[-/][A-Z]*[0-9]+)*|[A-Z_]{6,})")


def campaign_of_module(module: str) -> str | None:
    """The order item a data-foundation run belongs to, from what the run itself recorded."""
    if not module:
        return None
    match = ORDER_ITEM.match(module.strip())
    return match.group("item") if match else None


def classify_run(row: dict, current_campaigns: set) -> tuple:
    """Membership from the run's recorded CAMPAIGN, not from the table it happens to sit in.

    Two corrections the review required. `df_` is a naming convention, so classifying by it
    archived current data-foundation work for predating `gov_terminal`. And `GOVERNING` alone
    does not establish membership: a NON_GOVERNING campaign can be current history worth
    keeping, while a governing one can be superseded.

    Status never decides membership. A FAILED or inconclusive run of a current campaign is
    included WITH its status, because excluding it is selecting by outcome.
    """
    campaign = row.get("campaign_key")
    status = row.get("status") or "UNKNOWN"
    if not campaign:
        return "UNRESOLVED", "no campaign identity recorded for this run"
    if str(row.get("result_class") or "").upper().startswith("SUPERSEDED"):
        return "INVALIDATED", f"campaign {campaign} is recorded as superseded"
    if campaign in current_campaigns:
        return "INCLUDED_CURRENT", (
            f"campaign {campaign} is named by the committed orders; status {status} is "
            "preserved as recorded and does not decide membership")
    return "LEGACY_COMPARISON_ONLY", (
        f"campaign {campaign} is not in the current campaign set; admissible only where a "
        "comparison view opts into it and states its limitations")


def rows_for_runs(database: str, schema: str, relation: str, run_ids: set) -> int:
    """How many rows of a relation belong to these runs. Row level, not table level."""
    import duckdb

    con = duckdb.connect(database, read_only=True)
    try:
        columns = {row[1] for row in con.execute(
            f'PRAGMA table_info("{schema}"."{relation}")').fetchall()}
        if "run_id" not in columns:
            return 0
        marks = ", ".join("?" for _ in run_ids)
        return con.execute(
            f'SELECT count(*) FROM "{schema}"."{relation}" WHERE run_id IN ({marks})',
            list(run_ids)).fetchone()[0]
    finally:
        con.close()


def build_fixture_cube(path: str, *, terminals: int = 0, with_children: bool = False,
                       with_contract: bool = False, drop_contract_table: bool = False):
    """A disposable cube with the governed schema, for tests and rehearsals.

    Built through the PROVIDER, so the tables it creates carry their real constraints — the
    absence of which is what made a migrated cube answer every query and refuse every governed
    terminal.
    """
    import hashlib

    from predictor_duckdb_store.provider import PredictorDuckdbStore
    from sqlalchemy import text as sql

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=path, schema="main", memory_limit="1GB", threads=2,
                     min_free_bytes=1)
    store.engine()
    if drop_contract_table:
        with store.write_engine().begin() as conn:
            conn.execute(sql('DROP TABLE IF EXISTS "main"."gov_availability_contract"'))
    contract_digest = None
    if with_contract:
        body = json.dumps({"resource_id": "r", "availability": {
            "label": "WINDOW_START", "completion_lag_max": "UNKNOWN",
            "timezone_evidence": "UNKNOWN", "use_class": "ARCHIVE_RETROSPECTIVE"}},
            sort_keys=True, separators=(",", ":"))
        contract_digest = hashlib.sha256(body.encode("ascii")).hexdigest()
        store.write_availability_contracts([{"contract_sha256": contract_digest,
                                             "canonical_bytes": body}])
    with store.write_engine().begin() as conn:
        held = {row[0] for row in conn.execute(
            sql('SELECT terminal_sha256 FROM "main"."gov_terminal"'))}
        for index in range(terminals):
            digest = hashlib.sha256(f"terminal-{index}".encode()).hexdigest()
            if digest in held:
                continue          # idempotent: growing a fixture must not re-insert its past
            conn.execute(sql(
                'INSERT INTO "main"."gov_terminal" (terminal_sha256, campaign_sha256,'
                " campaign_key, unit_id, generation, actor, project, classification, status,"
                " reason, started_at, finished_at, terminal_lake, config_sha256,"
                " code_identity_json, costs_json, tags_json, synthetic_spec_sha256,"
                " received_at) VALUES (:d, :c, :k, :u, 1, 'a', 'p', 'NON_GOVERNING',"
                " 'COMPLETED', NULL, '2026-01-01', '2026-01-01', 'olap_cube', :cfg, '{}',"
                " '{}', '{}', NULL, '2026-01-01T00:00:00+00:00')"),
                {"d": digest, "c": "c" * 64, "k": f"fixture-{index}", "u": f"unit-{index}",
                 "cfg": "e" * 64})
            if with_children:
                conn.execute(sql(
                    'INSERT INTO "main"."gov_terminal_metric" (terminal_sha256, metric, split,'
                    " horizon, unit, value, std_dev, min_value, max_value)"
                    " VALUES (:d, 'wall_seconds', 'test', 0, 's', 1.0, NULL, NULL, NULL)"),
                    {"d": digest})
                conn.execute(sql(
                    'INSERT INTO "main"."gov_terminal_dataset" (terminal_sha256, delivery_id,'
                    " lake_id, resource_id, role, sha256, bytes,"
                    " availability_contract_sha256, verification_state)"
                    " VALUES (:d, :dl, 'l', 'r', 'input', :s, 1, :c, 'VERIFIED_TRANSFER')"),
                    {"d": digest, "dl": f"delivery-{index}", "s": "b" * 64,
                     "c": contract_digest or "f" * 64})
                conn.execute(sql(
                    'INSERT INTO "main"."gov_terminal_artifact" (terminal_sha256, role,'
                    " sha256, bytes) VALUES (:d, 'log', :s, 10)"),
                    {"d": digest, "s": "a" * 64})
    store._engine.dispose()
    return path


def replay_between(source: str, target: str, *, schema: str = "main",
                   dry_run: bool = False) -> dict:
    """Actually move accepted outcomes from one cube to another, parents and children.

    This is what `rollback` claimed and did not do. Rules:

    * a relation the destination lacks is CREATED, through the provider's own DDL, so it
      arrives with its constraints rather than as a bare copy;
    * children travel with their parent's identity, in the same transaction;
    * a row already present is skipped, so a second run replays nothing and still succeeds;
    * an interrupted replay resumes, because identity - not position - decides what is missing.
    """
    import duckdb

    # What the destination LACKS is measured before the schema step, because that step is what
    # creates it: asking afterwards would always answer "present" and report nothing created.
    absent_before = set()
    if Path(target).is_file():
        probe = duckdb.connect(target, read_only=True)
        try:
            held = {row[0] for row in probe.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = "
                f"'{schema}'").fetchall()}
        finally:
            probe.close()
        absent_before = set(PARENTS) - held
    else:
        absent_before = set(PARENTS)
    if not dry_run:
        # The destination must have the governed schema WITH constraints before anything lands.
        build_fixture_cube(target)
    con = duckdb.connect(target)
    con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
    created, replayed, children_replayed, contracts = [], 0, 0, 0
    report = {"schema": "olap_duckdb_replay.v1", "generated_utc": now(),
              "source": source, "target": target, "dry_run": dry_run, "relations": []}
    try:
        for parent, key in PARENTS.items():
            present = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = "
                f"'{schema}' AND table_name = '{parent}'").fetchone()[0]
            if parent in absent_before:
                created.append(parent)
            if not present:
                if dry_run:
                    report["relations"].append({"relation": parent,
                                                "outcome": "WOULD_CREATE"})
                    continue
            missing = con.execute(
                f'SELECT count(*) FROM src."{schema}"."{parent}" s WHERE s."{key}" NOT IN '
                f'(SELECT "{key}" FROM "{schema}"."{parent}")').fetchone()[0]
            if dry_run:
                report["relations"].append({"relation": parent, "outcome": "WOULD_REPLAY",
                                            "missing": missing})
                continue
            con.execute("BEGIN TRANSACTION")
            try:
                inserted = con.execute(
                    f'INSERT INTO "{schema}"."{parent}" SELECT s.* FROM '
                    f'src."{schema}"."{parent}" s WHERE s."{key}" NOT IN '
                    f'(SELECT "{key}" FROM "{schema}"."{parent}") RETURNING "{key}"'
                ).fetchall()
                moved = [row[0] for row in inserted]
                child_rows = 0
                if moved:
                    marks = ", ".join("?" for _ in moved)
                    for child in CHILDREN.get(parent, ()):
                        # children travel with their parent, inside the SAME transaction: a
                        # terminal whose metrics arrived separately could be observed without
                        # them, which is a half-restored outcome pretending to be whole
                        added = con.execute(
                            f'INSERT INTO "{schema}"."{child}" SELECT c.* FROM '
                            f'src."{schema}"."{child}" c WHERE c."{key}" IN ({marks}) '
                            'RETURNING 1', list(moved)).fetchall()
                        child_rows += len(added)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            if parent == "gov_availability_contract":
                contracts += len(moved)
            else:
                replayed += len(moved)
            children_replayed += child_rows
            report["relations"].append({"relation": parent, "outcome": "REPLAYED",
                                        "rows": len(moved), "child_rows": child_rows})
    finally:
        con.close()
    report["summary"] = {"terminals_replayed": replayed, "child_rows_replayed":
                         children_replayed, "contracts_replayed": contracts,
                         "relations_created": len(created)}
    return report


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


#: The campaigns the committed orders name as current. Read from a file so the set is a
#: reviewable input rather than a constant buried in a tool: the order requires membership to
#: come from committed orders, and a list nobody can see is not that.
DEFAULT_CURRENT_CAMPAIGNS = "docs/audits/work_plan/CURRENT_CAMPAIGNS.json"


def load_current_campaigns(path: Path) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict) or "campaigns" not in body:
        raise SystemExit(f"{path}: expected an object with a 'campaigns' list")
    return body


def cmd_select(args) -> int:
    """The per-run manifest, from lineage.

    Two corrections. Membership now comes from each run's recorded CAMPAIGN, checked against
    the campaign set the committed orders name — not from `GOVERNING` alone and not from a
    table's name prefix. And data-foundation runs are inventoried alongside governed terminals,
    so current `df_*` work is not archived merely for predating `gov_terminal`.
    """
    declared = load_current_campaigns(args.current_campaigns)
    current = set(declared["campaigns"])
    con = connect(None)
    alias = attach_source(con)

    def rows_of(relation, order=""):
        columns = [row[0] for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_catalog = "
            f"'{alias}' AND table_schema='public' AND table_name='{relation}' "
            "ORDER BY ordinal_position").fetchall()]
        if not columns:
            return [], []
        values = con.execute(f'SELECT * FROM {alias}.public."{relation}" {order}').fetchall()
        return columns, [dict(zip(columns, value)) for value in values]

    runs = []
    _cols, terminals = rows_of("gov_terminal", "ORDER BY received_at")
    for row in terminals:
        record = {"kind": "governed_terminal", "run_id": row["terminal_sha256"],
                  "campaign_key": row["campaign_key"], "status": row["status"],
                  "result_class": row["classification"],
                  "project": row["project"], "actor": row["actor"],
                  "unit_id": row["unit_id"], "generation": row["generation"],
                  "code_identity": row.get("code_identity_json"),
                  "config_sha256": row.get("config_sha256"),
                  "received_at": str(row["received_at"])}
        record["disposition"], record["rationale"] = classify_run(record, current)
        runs.append(record)

    # the data-foundation runs, linked to their campaign through dim_campaign.run_id
    _cols, campaigns = rows_of("dim_campaign")
    by_run = {}
    for row in campaigns:
        by_run.setdefault(str(row.get("run_id") or ""), row)
    _cols, foundation = rows_of("df_dim_run")
    for row in foundation:
        run_id = str(row.get("run_id") or "")
        campaign = next((entry for prefix, entry in by_run.items()
                         if prefix and run_id.startswith(prefix)), None)
        module = str(row.get("module") or "")
        record = {"kind": "data_foundation_run", "run_id": run_id,
                  "campaign_key": (campaign or {}).get("campaign_key")
                  or campaign_of_module(module),
                  "result_class": (campaign or {}).get("result_class"),
                  "status": row.get("status"), "module": row.get("module"),
                  "code_sha256": row.get("code_sha256"),
                  "inputs_sha256": row.get("inputs_sha256")}
        record["disposition"], record["rationale"] = classify_run(record, current)
        runs.append(record)

    included_runs = {entry["run_id"] for entry in runs
                     if entry["disposition"] == "INCLUDED_CURRENT"
                     and entry["kind"] == "data_foundation_run"}
    relations = []
    for relation in source_relations(con, alias):
        if relation["kind"] != "table":
            continue
        name = relation["name"]
        if name in GOVERNANCE or name in CLOSURE:
            relations.append({"relation": name, "rows": relation.get("rows"),
                              "disposition": "INCLUDED_CURRENT",
                              "selection": "WHOLE_RELATION",
                              "rationale": "governed contract or its dependency closure"})
            continue
        if "run_id" not in relation["columns"]:
            relations.append({"relation": name, "rows": relation.get("rows"),
                              "disposition": "LEGACY_COMPARISON_ONLY",
                              "selection": "WHOLE_RELATION",
                              "rationale": "carries no run identity, so no row in it can be "
                                           "tied to a current campaign"})
            continue
        # ROW level: one relation can hold both current and legacy runs, and a whole-table
        # decision cannot express that. This is the closure the previous version never computed.
        current_rows = con.execute(
            f'SELECT count(*) FROM {alias}.public."{name}" WHERE run_id IN '
            f'({", ".join("?" for _ in included_runs) or "NULL"})',
            list(included_runs)).fetchone()[0] if included_runs else 0
        relations.append({
            "relation": name, "rows": relation.get("rows"),
            "current_rows": current_rows,
            "legacy_rows": (relation.get("rows") or 0) - current_rows,
            "disposition": ("MIXED" if current_rows and current_rows != relation.get("rows")
                            else "INCLUDED_CURRENT" if current_rows
                            else "LEGACY_COMPARISON_ONLY"),
            "selection": "BY_RUN_ID",
            "rationale": "rows are selected by the run identities of current campaigns"})

    counts = {}
    for entry in runs:
        counts.setdefault(entry["kind"], {})
        counts[entry["kind"]][entry["disposition"]] = counts[entry["kind"]].get(
            entry["disposition"], 0) + 1
    body = {"schema": "olap_migration_selection.v2", "generated_utc": now(),
            "source_database": os.environ.get("PGDATABASE"),
            "current_campaigns": sorted(current),
            "current_campaigns_source": str(args.current_campaigns),
            "selection_rule": (
                "membership comes from each run's recorded campaign, checked against the "
                "campaigns the committed orders name. Not from GOVERNING alone, not from a "
                "table-name prefix, not from a date, a metric's sign or an outcome."),
            "runs": runs, "relations": relations,
            "counts": {"runs": counts,
                       "relations": {label: sum(1 for entry in relations
                                                if entry["disposition"] == label)
                                     for label in sorted({e["disposition"]
                                                          for e in relations})},
                       "current_rows_outside_governance": sum(
                           entry.get("current_rows") or 0 for entry in relations)}}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body["counts"], indent=1))
    return 0


def relations_from_manifest(args, available: dict) -> list[dict]:
    """What to copy, taken from the REVIEWED manifest rather than from a constant list.

    The previous export ignored the manifest entirely and copied whole tables named in two
    module-level tuples, so the selection was a document nobody's code read. Here each relation
    carries its own selection mode, and `BY_RUN_ID` means row level: a relation holding both
    current and legacy runs contributes only the current ones.
    """
    manifest = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    current_runs = sorted({entry["run_id"] for entry in manifest["runs"]
                           if entry["disposition"] == "INCLUDED_CURRENT"
                           and entry["kind"] == "data_foundation_run"})
    wanted = []
    for entry in manifest["relations"]:
        name = entry["relation"]
        if name not in available or available[name]["kind"] != "table":
            continue
        if args.scope == "cube" and entry["disposition"] == "LEGACY_COMPARISON_ONLY":
            continue
        if args.scope == "archive" and entry["disposition"] == "INCLUDED_CURRENT" \
                and entry.get("selection") == "WHOLE_RELATION":
            continue
        wanted.append({**available[name], "selection": entry.get("selection", "WHOLE_RELATION"),
                       "disposition": entry["disposition"]})
    return wanted, current_runs


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
                  batch_rows: int, *, predicate: str = "", parameters=None) -> int:
    """Copy every row exactly once, in bounded work units.

    With a unique key: keyset pagination, which is ordered and therefore complete. Without one:
    a single streaming INSERT, whose memory is bounded by the connection's own `memory_limit`
    rather than by slicing. Never LIMIT/OFFSET.
    """
    parameters = list(parameters or [])
    if key is None:
        con.execute(f"INSERT INTO {target} SELECT * FROM {source} {predicate}", parameters)
        return con.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
    copied, last = 0, None
    while True:
        if last is None:
            where, values = predicate, list(parameters)
        elif predicate:
            where = f'{predicate} AND "{key}" > ?'
            values = list(parameters) + [last]
        else:
            where, values = f'WHERE "{key}" > ?', [last]
        rows = con.execute(
            f'INSERT INTO {target} SELECT * FROM {source} {where} '
            f'ORDER BY "{key}" LIMIT {batch_rows} RETURNING "{key}"', values).fetchall()
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
    report = {"schema": "olap_duckdb_export.v2", "generated_utc": now(),
              "destination": str(args.destination), "target_schema": args.schema,
              "scope": args.scope, "batch_rows": args.batch_rows,
              "selection": str(args.selection) if args.selection else None, "relations": []}
    if args.selection:
        available = {entry["name"]: entry for entry in source_relations(con, alias)}
        chosen, current_runs = relations_from_manifest(args, available)
    else:
        chosen, current_runs = relations_for(args, con, alias), []
    for entry in chosen:
        name = entry["name"]
        target = f'"{args.schema}"."{name}"'
        source = f'{alias}.public."{name}"'
        existing = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = "
            f"'{args.schema}' AND table_name = '{name}'").fetchone()[0]
        held = con.execute(f"SELECT count(*) FROM {target}").fetchone()[0] if existing else 0
        state = import_state(entry.get("rows") or 0, held) if existing else "EMPTY"
        if existing and state == "COMPLETE" and not args.replace:
            # Complete is complete: a second import copies nothing and says so.
            report["relations"].append({"relation": name, "outcome": "ALREADY_PRESENT",
                                        "rows": held})
            continue
        if existing and state == "PARTIAL" and not args.replace:
            # An interrupted batch left rows behind. Calling that ALREADY_PRESENT stranded the
            # remainder and reported success; it is RESUMED instead, from the identity the
            # destination already holds.
            report["relations"].append({"relation": name, "outcome": "RESUMING",
                                        "rows_held": held,
                                        "source_rows": entry.get("rows")})
        if existing and state == "OVERFILLED":
            report["relations"].append({"relation": name, "outcome": "OVERFILLED_REFUSED",
                                        "rows_held": held, "source_rows": entry.get("rows"),
                                        "note": "the destination holds MORE rows than the "
                                                "source; that is a defect, not a completion, "
                                                "and it is not silently accepted"})
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
        # Row-level closure: only the current runs' rows, when the manifest says so.
        predicate = ""
        parameters = []
        if entry.get("selection") == "BY_RUN_ID" and current_runs and "run_id" in entry["columns"]:
            marks = ", ".join("?" for _ in current_runs)
            predicate = f'WHERE run_id IN ({marks})'
            parameters = list(current_runs)
            total = con.execute(f"SELECT count(*) FROM {source} {predicate}",
                                parameters).fetchone()[0]
        key = pagination_key(con, source, entry["columns"])
        copied = copy_relation(con, source, target, total, key, args.batch_rows,
                               predicate=predicate, parameters=parameters)
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
    report = {"schema": "olap_duckdb_validation.v2", "generated_utc": now(),
              "destination": str(args.destination), "target_schema": args.schema,
              "selection": str(args.selection) if args.selection else None,
              "relations": [], "mismatches": []}
    if args.selection:
        available = {entry["name"]: entry for entry in source_relations(con, alias)}
        chosen, current_runs = relations_from_manifest(args, available)
    else:
        chosen, current_runs = relations_for(args, con, alias), []
    for entry in chosen:
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
        # A selected relation is compared against the SELECTION, not against the whole source:
        # comparing a row-level subset with its full table would report a mismatch by design
        # and teach everyone to ignore the report.
        scope_predicate = ""
        if entry.get("selection") == "BY_RUN_ID" and current_runs \
                and "run_id" in entry["columns"]:
            values = ", ".join("'" + run.replace("'", "''") + "'" for run in current_runs)
            scope_predicate = f" WHERE run_id IN ({values})"
        source_rows = con.execute(
            f"SELECT count(*) FROM {source}{scope_predicate}").fetchone()[0]
        target_rows = con.execute(f"SELECT count(*) FROM {target}").fetchone()[0]
        source_digest = content_digest(con, f"(SELECT * FROM {source}{scope_predicate})",
                                       entry["columns"]) if scope_predicate else \
            content_digest(con, source, entry["columns"])
        target_digest = content_digest(con, target, entry["columns"])
        row = {"relation": name, "source_rows": source_rows, "destination_rows": target_rows,
               "source_content_md5": source_digest, "destination_content_md5": target_digest,
               "rows_match": source_rows == target_rows,
               "content_matches": source_digest == target_digest,
               "columns": len(entry["columns"])}
        # An EMPTY relation has no digest on either side, and that is a real agreement: the
        # aggregate over no rows is NULL. It is spelled out rather than folded into
        # `digests_agree`, whose job is to refuse two failures that look alike.
        both_empty = (source_rows == 0 and target_rows == 0
                      and source_digest is None and target_digest is None)
        row["content_matches"] = both_empty or digests_agree(source_digest, target_digest)
        row["compared_as"] = "EMPTY_ON_BOTH_SIDES" if both_empty else "CONTENT_DIGEST"
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


def snapshot_database(source: str, target: str, *, schema: str = "main",
                      expect_terminals=None, owner_stopped: bool = False) -> dict:
    """A copy of a DuckDB cube, and an honest statement of whether it is a snapshot.

    Copying `cube.duckdb` and `cube.duckdb.wal` one after another while a writer is running
    copies two moments, not one: a transaction can land between the two reads. Checkpointing
    the copy afterwards makes it self-consistent as a FILE and proves nothing about whether it
    matches any instant of the source.

    So verification requires BOTH: the owner is stopped, so there is a closed boundary to copy
    at, AND the resulting count matches what the service reported before it stopped. Without an
    expected count there is nothing to compare, and `verified` is false - it used to be
    unconditionally true, which is the weakest possible kind of check.
    """
    import shutil

    import duckdb

    source_path, target_path = Path(source), Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in ("", ".wal"):
        candidate = Path(str(source_path) + suffix)
        if candidate.is_file():
            shutil.copy2(candidate, Path(str(target_path) + suffix))
            copied.append({"file": candidate.name, "bytes": candidate.stat().st_size})
    con = duckdb.connect(str(target_path))
    con.execute("CHECKPOINT")
    counts = {}
    for name in GOVERNANCE:
        try:
            counts[name] = con.execute(
                f'SELECT count(*) FROM "{schema}"."{name}"').fetchone()[0]
        except Exception:
            counts[name] = None
    con.close()

    reasons = []
    if not owner_stopped:
        reasons.append("no coordinated writer boundary: the owning service was running, so "
                       "the main file and the write-ahead log were read at two different "
                       "moments and the pair need not correspond to any single instant")
    if expect_terminals is None:
        reasons.append("no expected terminal count to compare against, so nothing was verified")
    elif counts.get("gov_terminal") != expect_terminals:
        reasons.append(f"snapshot holds {counts.get('gov_terminal')} terminals, the service "
                       f"reported {expect_terminals}")
    return {"schema": "olap_duckdb_snapshot.v2", "generated_utc": now(),
            "source": source, "target": target, "files_copied": copied,
            "counts_in_snapshot": counts, "expected_terminals": expect_terminals,
            "owner_stopped": owner_stopped, "verified": not reasons,
            "unverified_because": reasons}


def cmd_snapshot(args) -> int:
    """A consistent copy of a live DuckDB cube, verified against what the service reports.

    Copying `cube.duckdb` alone is NOT a snapshot. DuckDB keeps recent transactions in a
    write-ahead log beside the database, so a plain `cp` of the main file silently omits them:
    measured here, a copy taken while the service held two freshly accepted terminals contained
    neither, and a rollback rehearsal against that copy reported "nothing to replay". Both files
    are copied, and the result is then OPENED and counted, because a backup nobody read is a
    belief rather than a backup.
    """
    source = Path(args.source)
    target = Path(args.target)
    body = snapshot_database(str(source), str(target), schema=args.schema,
                             expect_terminals=args.expect_terminals,
                             owner_stopped=args.owner_stopped)
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(body["files_copied"]),
                      "gov_terminal": body["counts_in_snapshot"].get("gov_terminal"),
                      "verified": body["verified"],
                      "unverified_because": body["unverified_because"]}, indent=1))
    return 0 if body["verified"] else 1


def cmd_rollback(args) -> int:
    """Replay DuckDB-era outcomes into another cube, and do it.

    The previous version emitted `REPLAY_REQUIRED` and returned zero having written nothing. A
    plain invocation now performs the replay; `--dry-run` is the only mode that reports without
    acting, and it says `WOULD_REPLAY` so the two can never be read as the same thing.

    The target is a DuckDB cube, which is what makes this testable against an explicitly
    disposable destination rather than against production.
    """
    report = replay_between(str(args.duckdb), str(args.target), schema=args.schema,
                            dry_run=args.dry_run)
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
    select.add_argument("--current-campaigns", type=Path,
                        default=Path(DEFAULT_CURRENT_CAMPAIGNS),
                        help="the reviewable list of campaigns the committed orders name")
    select.set_defaults(func=cmd_select)
    export = sub.add_parser("export"); common(export)
    export.add_argument("--batch-rows", type=int, default=100_000)
    export.add_argument("--selection", type=Path,
                        help="the reviewed selection manifest; the export copies what it says")
    export.add_argument("--provider-schema", action="store_true",
                        help="create the governed tables with the provider's own DDL first, "
                             "so their constraints exist; required for the cube scope")
    export.add_argument("--replace", action="store_true",
                        help="drop and rewrite a relation that is already present")
    export.set_defaults(func=cmd_export)
    validate = sub.add_parser("validate"); common(validate)
    validate.add_argument("--selection", type=Path,
                          help="compare against the reviewed selection, not the whole source")
    validate.set_defaults(func=cmd_validate)
    catchup = sub.add_parser("catchup"); common(catchup)
    catchup.add_argument("--watermark", required=True, help="ISO-8601 with offset")
    catchup.set_defaults(func=cmd_catchup)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--source", required=True)
    snapshot.add_argument("--target", required=True)
    snapshot.add_argument("--schema", default="main")
    snapshot.add_argument("--owner-stopped", action="store_true",
                          help="assert that the owning service is stopped, so the copy has a "
                               "closed boundary; without it the snapshot is not verified")
    snapshot.add_argument("--expect-terminals", type=int,
                          help="the count the SERVICE reports; the snapshot must match it")
    snapshot.add_argument("--out", type=Path, required=True)
    snapshot.set_defaults(func=cmd_snapshot)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--duckdb", type=Path, required=True, help="the DuckDB-era source")
    rollback.add_argument("--target", type=Path, required=True,
                          help="the destination cube; use a DISPOSABLE one to rehearse")
    rollback.add_argument("--schema", default="main")
    rollback.add_argument("--dry-run", action="store_true",
                          help="report what WOULD be replayed and write nothing")
    rollback.add_argument("--out", type=Path, required=True)
    rollback.set_defaults(func=cmd_rollback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
