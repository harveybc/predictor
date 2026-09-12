#!/usr/bin/env python3
"""C35 (order 2026-09-11): migrate the cube to campaign -> run -> unit.

Strictly additive. Nothing is dropped, deleted or rewritten in place
beyond two things that are pure labelling:

  * `dim_campaign.identity_authority` is stamped with what the frozen
    columns always were — the FIRST observation of the campaign, never
    its identity;
  * `fact_campaign_unit.run_id`, added as NULL by an earlier migration,
    is backfilled from the campaign's first observation for the rows
    that predate it. A NULL run is not a fact, it is a missing one, and
    filling it in from the only run those rows can belong to loses
    nothing.

The historical identity values stay exactly as loaded. Every row count
is asserted before and after; the migration refuses if any of them
moves.

    python olap/migrate_campaign_run.py [--dsn ...] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.campaign_envelope import (ENVELOPE_DDL, FIRST_OBSERVATION,  # noqa: E402
                                    SCHEMA)

#: counts that must be identical before and after. A migration that
#: changes any of them is not additive, whatever it calls itself.
CONSERVED = {
    "dim_campaign": f"SELECT count(*) FROM {SCHEMA}.dim_campaign",
    "fact_campaign_unit":
        f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit",
    "fact_campaign_consumption":
        f"SELECT count(*) FROM {SCHEMA}.fact_campaign_consumption",
    "dim_experiment": f"SELECT count(*) FROM {SCHEMA}.dim_experiment",
    "fact_performance": f"SELECT count(*) FROM {SCHEMA}.fact_performance",
    "units_producer_verified":
        f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit "
        "WHERE authority_state = 'PRODUCER_VERIFIED'",
    "units_translated":
        f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit "
        "WHERE authority_state = 'TRANSLATED_SUMMARY_NON_AUTHORITATIVE'",
    "units_born_at_terminal":
        f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit "
        "WHERE authority_state = 'PRODUCER_EMITTED_AT_TERMINAL'",
}

BACKFILLED = "BACKFILLED_FROM_CAMPAIGN_FIRST_OBSERVATION"
FROM_ENVELOPE = "ENVELOPE"


class MigrationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def dsn_from_env() -> str:
    need = ("PGUSER", "PGPASSWORD", "PGDATABASE")
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        raise MigrationRefusal(
            f"no DSN given and {missing} are not in the environment")
    return (f"postgresql+psycopg2://{os.environ['PGUSER']}:"
            f"{os.environ['PGPASSWORD']}@"
            f"{os.environ.get('PGHOST', 'localhost')}:"
            f"{os.environ.get('PGPORT', '5432')}/"
            f"{os.environ['PGDATABASE']}")


def counts(conn) -> dict:
    """Count what exists; absence is a fact, not a crash.

    A failed statement aborts the whole PostgreSQL transaction, so a
    missing table would poison every count after it. Each probe is
    rolled back explicitly, which is the only way "this table is not
    here" stays an observation instead of ending the connection.
    """
    from sqlalchemy import text
    out = {}
    for name, q in CONSERVED.items():
        try:
            out[name] = int(conn.execute(text(q)).scalar())
        except Exception:                                  # noqa: BLE001
            conn.rollback()
            out[name] = "ABSENT"
    return out


def migrate(engine, *, apply: bool = False) -> dict:
    from sqlalchemy import text

    with engine.connect() as probe:
        before = counts(probe)

    report = {"before": before, "applied": bool(apply)}
    if not apply:
        report["plan"] = [
            "create dim_campaign_run and the two run views",
            "stamp dim_campaign.identity_authority",
            "backfill fact_campaign_unit.run_id where NULL",
            "seed dim_campaign_run from the observed runs",
        ]
        return report

    with engine.begin() as conn:
        conn.exec_driver_sql(ENVELOPE_DDL)

        # 1. Fill the NULL run ids from the campaign's only known run.
        filled = conn.execute(text(f"""
            UPDATE {SCHEMA}.fact_campaign_unit f
               SET run_id = d.run_id
              FROM {SCHEMA}.dim_campaign d
             WHERE f.campaign_key = d.campaign_key
               AND f.run_id IS NULL
        """)).rowcount
        report["run_ids_backfilled"] = filled

        still_null = conn.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit "
            "WHERE run_id IS NULL")).scalar()
        if still_null:
            raise MigrationRefusal(
                f"{still_null} unit rows still carry no run_id — a fact "
                "that belongs to no run cannot be migrated by guessing")

        # 2. Seed one run row per observed (campaign, run). The
        #    identity comes from the campaign's first observation,
        #    which is the only identity these rows ever had; the
        #    source is recorded so nobody mistakes it for a fact the
        #    producer asserted per run.
        conn.exec_driver_sql(f"""
            ALTER TABLE {SCHEMA}.dim_campaign_run
              ADD COLUMN IF NOT EXISTS identity_source TEXT
        """)
        seeded = conn.execute(text(f"""
            INSERT INTO {SCHEMA}.dim_campaign_run
              (campaign_key, run_id, producer, result_class,
               design_sha256, code_identity, terminal_state,
               adjudication, identity_source)
            SELECT f.campaign_key,
                   f.run_id,
                   d.producer,
                   d.result_class,
                   d.design_sha256,
                   d.code_identity,
                   CASE WHEN count(DISTINCT f.terminal_state) = 1
                        THEN min(f.terminal_state)
                        ELSE 'MIXED' END,
                   CASE WHEN count(DISTINCT f.adjudication) = 1
                        THEN min(f.adjudication)
                        ELSE 'MIXED' END,
                   :src
              FROM {SCHEMA}.fact_campaign_unit f
              JOIN {SCHEMA}.dim_campaign d
                ON d.campaign_key = f.campaign_key
             GROUP BY f.campaign_key, f.run_id, d.producer,
                      d.result_class, d.design_sha256,
                      d.code_identity
            ON CONFLICT (campaign_key, run_id) DO NOTHING
        """), {"src": BACKFILLED}).rowcount
        report["runs_seeded"] = seeded

        conn.execute(text(f"""
            UPDATE {SCHEMA}.dim_campaign
               SET identity_authority = :ia
             WHERE identity_authority IS NULL
        """), {"ia": FIRST_OBSERVATION})

        # 3. No fact may reference a run that has no dimension row.
        #    This is not hypothetical: a loader process still holding
        #    the pre-C35 code in memory writes the fact and knows
        #    nothing about dim_campaign_run, so it leaves exactly this
        #    orphan. Re-running the migration adopts such rows; leaving
        #    the check out would let them stay invisible.
        orphans = [tuple(r) for r in conn.execute(text(f"""
            SELECT DISTINCT f.campaign_key, f.run_id
              FROM {SCHEMA}.fact_campaign_unit f
              LEFT JOIN {SCHEMA}.dim_campaign_run d
                     ON d.campaign_key = f.campaign_key
                    AND d.run_id       = f.run_id
             WHERE d.run_id IS NULL
        """))]
        if orphans:
            raise MigrationRefusal(
                f"{len(orphans)} fact run(s) have no dimension row "
                f"after seeding: {orphans[:3]}")
        report["orphan_runs"] = 0

    with engine.connect() as probe:
        after = counts(probe)
        report["after"] = after
        from sqlalchemy import text as _t
        report["dim_campaign_run"] = int(probe.execute(_t(
            f"SELECT count(*) FROM {SCHEMA}.dim_campaign_run")).scalar())
        report["attempts_by_campaign"] = [
            dict(r) for r in probe.execute(_t(
                f"SELECT * FROM {SCHEMA}.v_campaign_attempts "
                "ORDER BY campaign_key")).mappings()]

    moved = {k: (before[k], after[k]) for k in before
             if before[k] != after[k]}
    if moved:
        raise MigrationRefusal(
            f"the migration moved conserved counts: {moved}")
    report["conserved"] = True
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    from sqlalchemy import create_engine
    engine = create_engine(args.dsn or dsn_from_env())
    try:
        report = migrate(engine, apply=args.apply)
    finally:
        engine.dispose()
    print(json.dumps(report, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
