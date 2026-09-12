#!/usr/bin/env python3
"""C25: migrate the inventory tables to versioned history.

The first snapshot was stored with a logical id as PRIMARY KEY and
`ON CONFLICT DO NOTHING`, so a later observation of the same id was
silently dropped and the cube kept the older one. This migration
adds an observation identity, backfills it for the rows already
loaded, and moves the primary key to (logical id, observation) so
a new observation becomes a NEW VERSION beside the old.

No row is deleted and no value is rewritten. A backup is required
and is opened and re-hashed, exactly as the campaign backfill is.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import inventory_rows as ir  # noqa: E402
from tools.backfill_campaign_envelopes import (  # noqa: E402
    _engine, _sha_file)

TABLES = {
    "dim_lake_appearance": ("appearance_id", (
        "entity", "source_class", "frequency", "period_start",
        "period_end", "physical_sha256", "digest_state",
        "authority_class", "census_sha256")),
    "dim_lake_variable": ("variable_id", (
        "entity", "concept_name", "source_class", "unit",
        "event_time", "available_time", "semantics_declared",
        "appearance_count", "authority_class", "census_sha256")),
    "dim_public_series": ("series_id", (
        "family", "dataset_id", "frequency", "digest",
        "authority_class")),
    "dim_synthetic_generator": ("generator_id", (
        "producer", "family", "role", "authority_class")),
}


def main(argv=None) -> int:
    from sqlalchemy import text
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup-file", required=True, type=Path)
    ap.add_argument("--backup-sha256", required=True)
    ap.add_argument("--receipt", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args(argv)

    if not re.fullmatch(r"[0-9a-f]{64}", a.backup_sha256 or ""):
        raise SystemExit("REFUSED: --backup-sha256 is not "
                         "canonical")
    if not a.backup_file.is_file():
        raise SystemExit("REFUSED: the backup dump does not "
                         "exist — a digest is not a backup")
    if _sha_file(a.backup_file) != a.backup_sha256:
        raise SystemExit("REFUSED: the backup dump does not "
                         "match its declared digest")

    engine = _engine()
    before, after = {}, {}
    with engine.connect() as c:
        for t in list(TABLES) + list(ir.HISTORICAL_TABLES):
            try:
                before[t] = c.execute(text(
                    f"select count(*) from {t}")).scalar()
            except Exception:                   # noqa: BLE001
                before[t] = "ABSENT"

    migrated = {}
    with engine.begin() as c:
        for table, (key, fields) in TABLES.items():
            if before.get(table) in ("ABSENT", None):
                continue
            c.execute(text(
                f"ALTER TABLE {ir.SCHEMA}.{table} "
                "ADD COLUMN IF NOT EXISTS observation_sha256 "
                "TEXT"))
            rows = c.execute(text(
                f"SELECT {key}, {', '.join(fields)} FROM "
                f"{ir.SCHEMA}.{table} "
                "WHERE observation_sha256 IS NULL")
            ).mappings().all()
            for row in rows:
                payload = {f: (str(row[f])
                               if not isinstance(
                                   row[f], (int, bool))
                               else row[f])
                           for f in fields}
                c.execute(text(
                    f"UPDATE {ir.SCHEMA}.{table} SET "
                    "observation_sha256 = :o WHERE "
                    f"{key} = :k AND observation_sha256 IS NULL"),
                    {"o": ir.observation_sha256(payload),
                     "k": row[key]})
            migrated[table] = len(rows)
            c.execute(text(
                f"ALTER TABLE {ir.SCHEMA}.{table} "
                "ALTER COLUMN observation_sha256 SET NOT NULL"))
            # move the primary key: the old one froze history
            c.execute(text(
                f"ALTER TABLE {ir.SCHEMA}.{table} "
                f"DROP CONSTRAINT IF EXISTS {table}_pkey"))
            c.execute(text(
                f"ALTER TABLE {ir.SCHEMA}.{table} "
                f"ADD PRIMARY KEY ({key}, observation_sha256)"))
        c.exec_driver_sql(ir.INVENTORY_DDL)

    with engine.connect() as c:
        for t in list(TABLES) + list(ir.HISTORICAL_TABLES):
            try:
                after[t] = c.execute(text(
                    f"select count(*) from {t}")).scalar()
            except Exception:                   # noqa: BLE001
                after[t] = "ABSENT"

    lost = {t: [before[t], after[t]] for t in before
            if before[t] != after[t]}
    if lost:
        raise SystemExit(
            f"REFUSED: row counts changed during a migration "
            f"that must not delete anything: {lost}")

    receipt = {
        "schema": "crispdm.inventory_versioning_receipt.v1",
        "as_of": a.as_of,
        "backup": {"name": a.backup_file.name,
                   "sha256": a.backup_sha256,
                   "verification": "OPENED_AND_REHASHED"},
        "rows_given_an_observation_identity": migrated,
        "counts_before": before,
        "counts_after": after,
        "row_counts_unchanged": True,
        "deletions": "NONE — the primary key moved so history can "
                     "accumulate; no row was deleted and no value "
                     "was rewritten",
    }
    receipt["receipt_sha256"] = ir.observation_sha256(receipt)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1,
                                    sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in
                      ("rows_given_an_observation_identity",
                       "row_counts_unchanged",
                       "receipt_sha256")}, indent=1,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
