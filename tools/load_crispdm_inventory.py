#!/usr/bin/env python3
"""Load a CRISP-DM inventory document into the additive OLAP tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.crispdm_inventory import load_json
from olap.init_db import DDL, SEED, build_engine_from_pg_env
from olap.information_schema import INFORMATION_DDL
from olap.information_schema import load_inventory_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory = load_json(args.inventory)
    engine = build_engine_from_pg_env()
    # Use only additive, idempotent DDL here. The legacy ETL migration also
    # deduplicates old performance facts and must not be a side effect of an
    # inventory load.
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
        conn.exec_driver_sql(INFORMATION_DDL)
        conn.exec_driver_sql(SEED)
    counts = load_inventory_document(engine, inventory)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
