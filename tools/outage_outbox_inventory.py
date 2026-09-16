#!/usr/bin/env python3
"""What every outbox on this host holds, and what the outage interval actually touched.

U1 of `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md`:

    "Inventory pending, sent and rejected outboxes associated with the outage interval.
     Reconcile existing authorized terminals through the existing outbox implementations; do
     not synthesize success or drop rejected items. Record pending-before/after, actual sends,
     duplicate protection and any unresolved terminal. Do not assert 'nothing lost' merely
     because an outbox exists."

That last sentence is the point. An outbox is a mechanism, not evidence. This walks **every**
outbox directory it is given, classifies each envelope by the state directory it sits in,
timestamps it, and separates three questions that get conflated:

  held        what is pending RIGHT NOW, anywhere;
  in-window   which envelopes were written or last modified inside the outage interval;
  unresolved  envelopes whose terminal digest cannot be found in the cube.

The third is the one that can say something was lost, and it is answered against the cube
rather than against the outbox's own opinion of itself.

usage:
  outage_outbox_inventory.py --since ISO --until ISO --out RECEIPT.json [--root DIR ...]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

#: The three state directories the durable outbox uses. `failed` is a sidecar convention of
#: the OLAP loader's own outbox and is included so a rejected item cannot hide.
STATES = ("pending", "sent", "adjudicated", "failed")


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def envelope_facts(path: Path) -> dict:
    """Whatever identifies this envelope, without assuming a schema it may not have."""
    facts = {"file": path.name, "bytes": path.stat().st_size,
             "mtime": iso(path.stat().st_mtime)}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        facts["unreadable"] = f"{type(exc).__name__}: {exc}"
        return facts
    if not isinstance(body, dict):
        facts["unreadable"] = "not a JSON object"
        return facts
    # An envelope carries no terminal_sha256: the digest is computed by data-gov when the
    # terminal is accepted. Its identity here is the slot the governance reserves —
    # (campaign_sha256, unit_id, generation) — which is also what duplicate protection keys on.
    terminal = body.get("terminal") if isinstance(body.get("terminal"), dict) else {}
    for key in ("campaign_sha256", "unit_id"):
        if body.get(key) is not None:
            facts[key] = body[key]
    for key in ("generation", "status", "reason"):
        if terminal.get(key) is not None:
            facts[key] = terminal[key]
    if isinstance(terminal.get("tags"), dict):
        facts["tags"] = terminal["tags"]
    for key in ("attempts", "last_error", "last_attempt_at", "disposition"):
        if body.get(key) is not None:
            facts[key] = body[key]
    return facts


def walk(root: Path) -> dict:
    out = {"root": str(root), "exists": root.is_dir(), "states": {}}
    if not root.is_dir():
        return out
    for state in STATES:
        directory = root / state
        if not directory.is_dir():
            continue
        out["states"][state] = [envelope_facts(item) for item in sorted(directory.iterdir())
                                if item.is_file()]
    loose = [envelope_facts(item) for item in sorted(root.iterdir())
             if item.is_file() and item.suffix == ".json"]
    if loose:
        out["states"]["_loose"] = loose
    return out


def reconcile_against_cube(envelopes) -> list:
    """Ask the CUBE whether each envelope's slot is already closed.

    This is what separates "an outbox exists" from "nothing was lost". An envelope sitting in
    `pending` proves only that the sender did not record a success; the cube is what knows
    whether the terminal was accepted. A pending envelope whose slot is already COMPLETED is
    a stale marker, not a loss - and the two must not be reported as the same thing.
    """
    import subprocess

    rows = []
    for item in envelopes:
        campaign, unit = item.get("campaign_sha256"), item.get("unit_id")
        query = ("SELECT unit_id || '|' || status || '|' || generation FROM gov_terminal "
                 f"WHERE campaign_sha256 = '{campaign}'")
        if unit:
            query += f" AND unit_id = '{unit}'"
        out = subprocess.run(["psql", "-At", "-c", query + ";"], capture_output=True, text=True)
        found = [line for line in out.stdout.strip().split("\n") if line]
        rows.append({"file": item["file"], "root": item["root"], "state": item["state"],
                     "campaign_sha256": campaign, "unit_id": unit,
                     "in_cube": None if out.returncode != 0 else bool(found),
                     "cube_rows": found,
                     "query_error": out.stderr.strip()[:200] if out.returncode != 0 else None})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", required=True, help="outage start, ISO-8601 with offset")
    parser.add_argument("--until", required=True, help="outage end, ISO-8601 with offset")
    parser.add_argument("--root", action="append", type=Path, default=[],
                        help="an outbox directory; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-cube", action="store_true",
                        help="ask the cube whether each envelope's slot is already closed; "
                             "read-only, and uses the ambient PG* connection")
    args = parser.parse_args(argv)

    since = datetime.fromisoformat(args.since).astimezone(timezone.utc)
    until = datetime.fromisoformat(args.until).astimezone(timezone.utc)
    roots = [walk(root.expanduser()) for root in args.root]

    held, in_window, terminals = [], [], []
    for root in roots:
        for state, items in root["states"].items():
            for item in items:
                tagged = dict(item, root=root["root"], state=state)
                if state == "pending":
                    held.append(tagged)
                if since.isoformat() <= item["mtime"].replace("Z", "+00:00") <= until.isoformat():
                    in_window.append(tagged)
                if item.get("campaign_sha256"):
                    terminals.append(tagged)

    body = {
        "schema": "outage_outbox_inventory.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"),
        "outage": {"since": since.isoformat(), "until": until.isoformat(),
                   "seconds": (until - since).total_seconds()},
        "roots": roots,
        "counts": {
            "roots_inspected": len(roots),
            "roots_present": sum(1 for r in roots if r["exists"]),
            "envelopes_total": sum(len(items) for r in roots for items in r["states"].values()),
            "pending_now": len(held),
            "written_in_outage_window": len(in_window),
            "carrying_a_terminal_digest": len(terminals),
        },
        "pending_now": held,
        "written_in_outage_window": in_window,
        "campaigns_referenced": sorted({item["campaign_sha256"] for item in terminals}),
    }
    if args.check_cube:
        body["cube_reconciliation"] = reconcile_against_cube(terminals)
        body["counts"]["unresolved_terminals"] = sum(
            1 for row in body["cube_reconciliation"] if row["in_cube"] is False)
    else:
        body["counts"]["unresolved_terminals"] = None
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
