#!/usr/bin/env python3
"""Backfill campaign envelopes into the populated cube.

Additive only. The tool refuses to run unless a backup digest is
supplied, records counts before and after, loads each envelope
idempotently, and writes a receipt naming exactly what changed.
It has no delete, truncate or reset path — `reset_olap.py` is not
part of this plan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import campaign_envelope as ce  # noqa: E402


def _sha_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

WATCHED = (
    "dim_experiment", "fact_performance", "fact_results_summary",
    "dim_dataset", "dim_series", "dim_variable",
    "fact_dataset_inventory", "fact_variable_profile",
    "fact_ingestion_receipt",
    "dim_campaign", "fact_campaign_unit",
    "fact_campaign_consumption",
)


def _engine():
    from sqlalchemy import create_engine
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    pw = os.getenv("PGPASSWORD", "metabase_pass")
    return create_engine(
        f"postgresql://{user}:{pw}@{host}:{port}/{db}",
        future=True)


def counts(engine) -> dict:
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
    ap.add_argument("--envelope-dir", required=True, type=Path)
    ap.add_argument("--backup-file", required=True, type=Path,
                    help="the pre-backfill dump itself; the "
                         "backfill OPENS it and recomputes its "
                         "digest")
    ap.add_argument("--backup-sha256", required=True,
                    help="expected digest of that dump")
    ap.add_argument("--backup-logical-name", required=True)
    ap.add_argument("--receipt", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args(argv)

    # C13: the backup gate is no longer a length check. The dump
    # is opened and re-hashed here, because a well-formed string
    # authorised a migration with no file on disk.
    import re
    if not re.fullmatch(r"[0-9a-f]{64}", a.backup_sha256 or ""):
        raise SystemExit(
            "REFUSED: --backup-sha256 is not a canonical "
            "SHA-256")
    if not a.backup_file.is_file():
        raise SystemExit(
            f"REFUSED: the backup dump does not exist at "
            f"{a.backup_file.name} — a digest is not a backup")
    recomputed = _sha_file(a.backup_file)
    if recomputed != a.backup_sha256:
        raise SystemExit(
            "REFUSED: the backup dump on disk does not match "
            f"the declared digest (recomputed {recomputed[:16]}, "
            f"declared {a.backup_sha256[:16]}) — the migration "
            "is not reversible from that file")
    backup_bytes = a.backup_file.stat().st_size

    engine = _engine()
    before = counts(engine)
    ce.ensure_envelope_tables(engine)
    loaded, receipts = {}, []
    for p in sorted(a.envelope_dir.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        ce.validate_envelope(doc)
        c = ce.load_envelope(engine, doc)
        loaded[doc["campaign_key"]] = c
        receipts.append(ce.ingestion_receipt(
            doc, loaded_counts=c, as_of=a.as_of))
    after = counts(engine)

    changed = {k: [before[k], after[k]]
               for k in WATCHED if before[k] != after[k]}
    untouched = {k: before[k] for k in WATCHED
                 if before[k] == after[k]}
    receipt = {
        "schema": "crispdm.campaign_backfill_receipt.v1",
        "as_of": a.as_of,
        "backup": {"logical_name": a.backup_logical_name,
                   "sha256": a.backup_sha256,
                   "bytes": backup_bytes,
                   "verification": "OPENED_AND_REHASHED"},
        "counts_before": before,
        "counts_after": after,
        "changed_tables": changed,
        "untouched_tables": untouched,
        "loaded": loaded,
        "ingestion_receipts": receipts,
        "deletions": "NONE — this tool has no delete, truncate "
                     "or reset path",
    }
    receipt["receipt_sha256"] = ce._sha(receipt,
                                        "receipt_sha256")
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1,
                                    sort_keys=True) + "\n")
    print(json.dumps({"changed": changed,
                      "loaded": loaded,
                      "receipt_sha256":
                          receipt["receipt_sha256"]},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
