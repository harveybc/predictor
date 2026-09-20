#!/usr/bin/env python3
"""RP51: the receipts a governed run leaves behind, written by the SAME client that produced them.

The dictum's F2: the runner registered prepare, the pilots and every cell before the first child ran,
and when that child failed the report still said `all_units_governed=true` — eight campaigns
registered, one closed, seven open — while an empty report passed the same summary. And nothing ever
persisted the accepted terminals, so the closure could only call the units HISTORICAL_UNGOVERNED.

This module holds the two receipts and the rules over them:

  DELIVERIES.json        written by tools/df_e1_governed.acquire, one entry per unit
  TERMINAL_RECEIPTS.json written HERE, from the client's own answer when a terminal is accepted:
                         the campaign it belongs to, the terminal's digest as the service returned
                         it, the instant it was accepted, its status, and the reconciliation that
                         followed. Nothing is synthesised from a summary.

The population is the design's enumeration, and every unit has a state at all times:

  NOT_STARTED   registered or not, it has not run
  RUNNING       its attempt exists without a verdict
  CLOSED        its terminal was accepted and its campaign reconciles
  BLOCKED       its dependency failed or the budget ran out before it could run
  PENDING       its campaign is registered and it is NOT closed: this is the state the previous
                summary hid, and it is named per unit until a later run recovers it
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "df_e1_terminal_receipts.v1"
NOT_STARTED, RUNNING, CLOSED, BLOCKED, PENDING = "NOT_STARTED", "RUNNING", "CLOSED", "BLOCKED", "PENDING"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def record_accepted(root: Path, unit_id: str, *, campaign_sha256: str, campaign_key: str, terminal: dict,
                    receipt: dict, reconciliation: dict, design_sha256: str, started_at: str | None = None) -> dict:
    """Persist ONE accepted terminal, from the client's own receipt, under a lock: children run in
    parallel and a whole-file rewrite would erase the units that closed a moment earlier."""
    import fcntl
    path = Path(root) / "TERMINAL_RECEIPTS.json"
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock, "w")
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        return _record_locked(path, unit_id, campaign_sha256=campaign_sha256, campaign_key=campaign_key,
                              terminal=terminal, receipt=receipt, reconciliation=reconciliation,
                              design_sha256=design_sha256, started_at=started_at)
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _record_locked(path: Path, unit_id: str, *, campaign_sha256: str, campaign_key: str, terminal: dict,
                   receipt: dict, reconciliation: dict, design_sha256: str, started_at: str | None) -> dict:
    import os
    doc = json.loads(path.read_text()) if path.is_file() else {"schema": SCHEMA, "design_sha256": design_sha256,
                                                               "units": {}}
    if doc.get("design_sha256") != design_sha256:
        raise SystemExit("REFUSED: the terminal receipts in this root belong to another design")
    doc["units"][unit_id] = {
        "campaign_key": campaign_key, "campaign_sha256": campaign_sha256,
        "terminal_sha256": (receipt or {}).get("terminal_sha256"),
        "generation": terminal.get("generation"), "status": terminal.get("status"),
        "accepted_at": now_iso(), "work_started_at": started_at,
        "started_at": terminal.get("started_at"), "finished_at": terminal.get("finished_at"),
        "metrics": len(terminal.get("metrics") or []), "deliveries": list(terminal.get("deliveries") or []),
        "reconciliation": reconciliation,
        "source": "the governance client's own answer when it accepted this terminal",
    }
    doc["updated_at"] = now_iso()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, path)
    return doc["units"][unit_id]


def population_states(design: dict, root: Path, *, registered: dict | None = None) -> dict:
    """Every unit of the sealed design with the state it is actually in."""
    root = Path(root)
    receipts = {}
    path = root / "TERMINAL_RECEIPTS.json"
    if path.is_file():
        try:
            receipts = (json.loads(path.read_text()).get("units") or {})
        except ValueError:
            receipts = {}
    deliveries = {}
    dpath = root / "DELIVERIES.json"
    if dpath.is_file():
        try:
            deliveries = (json.loads(dpath.read_text()).get("units") or {})
        except ValueError:
            deliveries = {}
    registered = registered or {u: bool(v.get("campaign_sha256")) for u, v in deliveries.items()}
    units = ["prepare"] + [c["cell_id"] for c in design["pilots"] + design["cells"]]
    states, reasons = {}, {}
    for unit in units:
        attempt = root / "attempts" / unit
        outcome = attempt / "outcome.json"
        closed = receipts.get(unit)
        recon = (closed or {}).get("reconciliation") or {}
        reconciled = bool(recon.get("http") == 200 and not recon.get("missing_units")
                          and not recon.get("accounting_only") and not recon.get("lake_only"))
        if closed and reconciled:
            states[unit] = CLOSED
        elif registered.get(unit):
            states[unit] = PENDING
            reasons[unit] = ("its campaign is registered and it has no accepted terminal with a clean reconciliation"
                             if closed else "its campaign is registered and no terminal was accepted for it")
        elif outcome.is_file():
            states[unit] = RUNNING
        else:
            states[unit] = NOT_STARTED
    return {"units": states, "reasons": reasons, "receipts": receipts,
            "counts": {state: sum(1 for s in states.values() if s == state)
                       for state in (NOT_STARTED, RUNNING, CLOSED, BLOCKED, PENDING)},
            "complete": all(s == CLOSED for s in states.values()),
            "rule": "the population is the sealed design's enumeration; a registered unit that is not closed is "
                    "PENDING and named, never absorbed into a total"}


def mark_blocked(states: dict, unit_id: str, why: str) -> dict:
    states["units"][unit_id] = BLOCKED
    states["reasons"][unit_id] = why
    states["counts"] = {state: sum(1 for s in states["units"].values() if s == state)
                        for state in (NOT_STARTED, RUNNING, CLOSED, BLOCKED, PENDING)}
    states["complete"] = all(s == CLOSED for s in states["units"].values())
    return states
