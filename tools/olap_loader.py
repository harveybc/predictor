#!/usr/bin/env python3
"""Drain the OLAP outbox into the cube. CPU only, idempotent.

The loader is the only component that needs PostgreSQL. It reads
pending entries, loads each exactly once, records a receipt, and
leaves anything it could not load in a retryable state with the
reason attached. A database that is down costs a retry, never a
scientific result.

  --once      drain what is pending and exit (the default)
  --watch N   drain every N seconds until interrupted
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import campaign_envelope as ce  # noqa: E402
from olap import outbox as ob  # noqa: E402


def _engine():
    import os
    from sqlalchemy import create_engine
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    pw = os.getenv("PGPASSWORD", "metabase_pass")
    return create_engine(
        f"postgresql://{user}:{pw}@{host}:{port}/{db}",
        future=True, pool_pre_ping=True)


def drain_once(root=None, *, as_of: str | None = None) -> dict:
    entries = ob.pending_entries(root)
    result = {"attempted": len(entries), "loaded": 0,
              "already_present": 0, "failed": 0,
              "database_unavailable": False, "receipts": []}
    if not entries:
        result["heartbeat"] = ob.heartbeat(root)
        return result
    engine = None
    try:
        engine = _engine()
        ce.ensure_envelope_tables(engine)
    except Exception as exc:                    # noqa: BLE001
        # The database is unavailable. Nothing is lost and
        # nothing is marked failed: the entries stay pending and
        # the next drain retries them.
        result["database_unavailable"] = True
        result["reason"] = f"{exc.__class__.__name__}"
        result["heartbeat"] = ob.heartbeat(root)
        return result

    try:
        return _drain_entries(engine, entries, root, as_of,
                              result)
    finally:
        # a short-lived drain must not hold a connection open:
        # it blocks a throwaway database from being dropped and
        # keeps a pooled socket for no reason
        engine.dispose()


def _drain_entries(engine, entries, root, as_of, result):
    for path in entries:
        try:
            body = ob.read_entry(path)
        except Exception as exc:                # noqa: BLE001
            ob.mark(path, ob.FAILED, root=root,
                    reason=f"unreadable entry: "
                           f"{exc.__class__.__name__}")
            result["failed"] += 1
            continue
        doc = body.get("document", {})
        if body.get("outbox_kind") != "envelope":
            ob.mark(path, ob.FAILED, root=root,
                    reason="only envelopes are loadable today")
            result["failed"] += 1
            continue
        try:
            counts = ce.load_envelope(engine, doc)
        except SystemExit as exc:
            # a refusal is a PERMANENT verdict about these bytes
            ob.mark(path, ob.FAILED, root=root,
                    reason=str(exc))
            result["failed"] += 1
            continue
        except Exception as exc:                # noqa: BLE001
            # transient: leave it pending for the next drain
            result["database_unavailable"] = True
            result["reason"] = f"{exc.__class__.__name__}"
            break
        if counts["units"] == 0 and counts["skipped_existing"]:
            result["already_present"] += 1
        else:
            result["loaded"] += 1
        receipt = ce.ingestion_receipt(doc, loaded_counts=counts,
                                       as_of=as_of)
        r = ob.ensure_outbox(root)
        (r / "receipts" /
         f"receipt-{receipt['receipt_sha256']}.json").write_text(
            json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        result["receipts"].append(receipt["receipt_sha256"])
        ob.mark(path, ob.LOADED, root=root)
    result["heartbeat"] = ob.heartbeat(root)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outbox", type=Path)
    ap.add_argument("--as-of")
    ap.add_argument("--watch", type=float)
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="stop after N watch cycles (0 = "
                         "forever); keeps the loader testable")
    a = ap.parse_args(argv)
    if a.watch is None:
        print(json.dumps(drain_once(a.outbox, as_of=a.as_of),
                         indent=1, sort_keys=True))
        return 0
    cycles = 0
    while True:
        out = drain_once(a.outbox, as_of=a.as_of)
        print(json.dumps(out, sort_keys=True))
        cycles += 1
        if a.max_cycles and cycles >= a.max_cycles:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    raise SystemExit(main())
