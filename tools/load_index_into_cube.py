#!/usr/bin/env python3
"""C15: load the census and common index into the cube.

Additive only, idempotent, and kept apart from the historical
variable profiles. Requires a real backup, exactly as the
campaign backfill does.
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
    _engine, _sha_file, counts)

WATCHED = ir.HISTORICAL_TABLES + ir.NEW_TABLES


def _counts(engine):
    from sqlalchemy import text
    out = {}
    with engine.connect() as c:
        for t in WATCHED:
            try:
                out[t] = c.execute(
                    text(f"select count(*) from {t}")).scalar()
            except Exception:                   # noqa: BLE001
                out[t] = "ABSENT"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, type=Path)
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

    index = json.loads(a.index.read_text())
    engine = _engine()
    before = _counts(engine)
    loaded = ir.load_index(engine, index)
    after = _counts(engine)
    changed = {k: [before[k], after[k]] for k in WATCHED
               if before[k] != after[k]}
    historical_untouched = {
        k: before[k] for k in ir.HISTORICAL_TABLES
        if before[k] == after[k]}
    if any(before[k] != after[k]
           for k in ir.HISTORICAL_TABLES):
        raise SystemExit(
            "REFUSED: a historical table changed — this loader "
            "is additive only")
    receipt = {
        "schema": "crispdm.index_ingestion_receipt.v1",
        "as_of": a.as_of,
        "index_sha256": index["index_sha256"],
        "backup": {"name": a.backup_file.name,
                   "sha256": a.backup_sha256,
                   "verification": "OPENED_AND_REHASHED"},
        "loaded": loaded,
        "counts_before": before,
        "counts_after": after,
        "changed_tables": changed,
        "historical_tables_untouched": historical_untouched,
        "separation": "these tables are distinct from the 99 "
                      "historical variable profiles; the two "
                      "are never mixed",
        "deletions": "NONE",
    }
    receipt["receipt_sha256"] = ir.json.dumps and __import__(
        "hashlib").sha256(json.dumps(
            {k: v for k, v in receipt.items()},
            sort_keys=True).encode()).hexdigest()
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1,
                                    sort_keys=True) + "\n")
    print(json.dumps({"loaded": loaded, "changed": changed,
                      "historical_untouched":
                          historical_untouched,
                      "receipt_sha256":
                          receipt["receipt_sha256"]},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
