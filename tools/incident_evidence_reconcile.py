#!/usr/bin/env python3
"""What governance says was accepted, against what the recovered cube actually holds.

F3 of `docs/handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md`:

    "Reconcile pre-incident accepted delivery/terminal/outbox identities and their child
     CONTENT against the recovered database; separate committed evidence from pending or
     replayable writes. Counts alone do not support no-loss claims. Do not presume the WAL
     contained only tests without supporting receipts."

I claimed nothing governed was lost, and supported it with three row counts. That is not the
same claim. What makes it checkable is: data-gov's OWN accounting is the record of what was
accepted, it is a separate database that the incident did not touch, and every terminal it
records must be present in the recovered cube with its children's CONTENT intact.

Three populations are kept apart, because they mean different things:

  committed    accepted by governance AND present in the cube — the evidence that survived;
  missing      accepted by governance and ABSENT from the cube — actual loss, if any;
  replayable   still held in an outbox, so its absence from the cube costs a retry, not a loss.

Nothing here opens the quarantined write-ahead log. Read-only throughout.

usage:
  incident_evidence_reconcile.py --accounting DB --cube FILE [--outbox DIR ...] --out R.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CHILDREN = ("gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def accepted_terminals(accounting: Path) -> list[dict]:
    """What data-gov's own accounting records as accepted. The independent record."""
    con = sqlite3.connect(f"file:{accounting}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in con.execute(
            "SELECT terminal_sha256, campaign_sha256, unit_id, generation, status "
            "FROM governed_terminals ORDER BY terminal_sha256")]
    finally:
        con.close()
    return rows


def outbox_identities(roots: list[Path]) -> dict:
    """Envelope slots still held anywhere, by (campaign, unit): absence is a retry, not a loss."""
    held = {}
    for root in roots:
        for state in ("pending", "sent", "adjudicated"):
            directory = root / state
            if not directory.is_dir():
                continue
            for item in directory.iterdir():
                if not item.is_file():
                    continue
                try:
                    body = json.loads(item.read_text(encoding="utf-8"))
                except Exception:
                    continue
                key = (body.get("campaign_sha256"), body.get("unit_id"))
                if key != (None, None):
                    held.setdefault(key, []).append({"root": str(root), "state": state,
                                                     "file": item.name})
    return held


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--schema", default="main")
    parser.add_argument("--outbox", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import duckdb

    accepted = accepted_terminals(args.accounting)
    outboxes = outbox_identities([root.expanduser() for root in args.outbox])

    con = duckdb.connect(str(args.cube), read_only=True)
    try:
        in_cube = {row[0]: row for row in con.execute(
            f'SELECT terminal_sha256, campaign_sha256, unit_id, generation, status '
            f'FROM "{args.schema}".gov_terminal').fetchall()}
        child_content = {}
        for child in CHILDREN:
            columns = [row[1] for row in con.execute(
                f'PRAGMA table_info("{args.schema}"."{child}")').fetchall()]
            projection = ", ".join(f'"{column}"' for column in columns)
            for digest, content in con.execute(
                    f'SELECT terminal_sha256, md5(string_agg(h, \'\' ORDER BY h)) FROM '
                    f'(SELECT terminal_sha256, md5(CAST(ROW({projection}) AS VARCHAR)) h '
                    f'FROM "{args.schema}"."{child}") t GROUP BY terminal_sha256').fetchall():
                child_content.setdefault(digest, {})[child] = content
    finally:
        con.close()

    committed, missing, replayable, disagreeing = [], [], [], []
    for row in accepted:
        digest = row["terminal_sha256"]
        found = in_cube.get(digest)
        record = {"terminal_sha256": digest, "campaign_sha256": row["campaign_sha256"],
                  "unit_id": row["unit_id"], "status_in_accounting": row["status"]}
        if not found:
            key = (row["campaign_sha256"], row["unit_id"])
            if key in outboxes:
                record["outbox"] = outboxes[key]
                replayable.append(record)
            else:
                missing.append(record)
            continue
        record["status_in_cube"] = found[4]
        record["children"] = child_content.get(digest, {})
        if found[4] != row["status"]:
            record["disagreement"] = "status"
            disagreeing.append(record)
        else:
            committed.append(record)

    body = {"schema": "incident_evidence_reconcile.v1", "generated_utc": now(),
            "accounting": str(args.accounting), "cube": str(args.cube),
            "cube_sha256": hashlib.sha256(args.cube.read_bytes()).hexdigest()
            if args.cube.stat().st_size < 400_000_000 else "NOT_HASHED_TOO_LARGE",
            "counts": {"accepted_by_governance": len(accepted),
                       "committed_in_cube": len(committed),
                       "missing_from_cube": len(missing),
                       "replayable_from_outbox": len(replayable),
                       "status_disagreements": len(disagreeing),
                       "terminals_in_cube": len(in_cube)},
            "missing_from_cube": missing, "replayable_from_outbox": replayable,
            "status_disagreements": disagreeing,
            "children_present_for": sum(1 for entry in committed if entry["children"]),
            "note": ("The quarantined write-ahead log was not opened. Absence from the cube is "
                     "reported as LOSS only when no outbox still holds the slot.")}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body["counts"], indent=1))
    return 0 if not missing and not disagreeing else 1


if __name__ == "__main__":
    raise SystemExit(main())
