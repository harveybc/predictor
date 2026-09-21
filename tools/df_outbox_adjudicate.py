#!/usr/bin/env python3
"""Adjudicate a terminal envelope that can never be accepted, using the existing additive procedure.

Nothing is deleted or rewritten. `TerminalOutbox.dispose` MOVES the envelope, byte for byte, from
`pending/` to `adjudicated/` and writes a write-once disposition beside it that records the decision,
its reason, the envelope's digest, the campaign, the unit, the generation and the failure sidecar as
it stood. The audit trail grows; it never shrinks.

Two decisions exist, and both are additive:

    INVALID_ENVELOPE  this envelope can never be accepted as it is (for example, a duplicate of a
                      fact the service already holds at that generation, or a malformed payload)
    SUPERSEDED        an accepted successor terminal replaces it; its digest is recorded in the link

Use `list` first: it shows every pending envelope with its unit, generation, failure class and last
error, so a disposition is chosen against what is actually there.

    python tools/df_outbox_adjudicate.py list [--outbox-dir DIR]
    python tools/df_outbox_adjudicate.py dispose --file NAME.json --decision INVALID_ENVELOPE \\
        --reason "..." [--successor-terminal-sha256 SHA] [--outbox-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def spools(root: Path):
    """The outbox itself and every per-unit spool under it."""
    root = Path(root)
    found = []
    if (root / "pending").is_dir():
        found.append(root)
    units = root / "units"
    if units.is_dir():
        found += [p for p in sorted(units.iterdir()) if (p / "pending").is_dir()]
    return found


def listing(root: Path) -> dict:
    GR = _module("governed_run")
    out = {}
    for spool in spools(root):
        status = GR.TerminalOutbox(spool).status()
        if status["pending"] or status["adjudicated"]:
            out[str(spool)] = {"pending": status["pending"], "adjudicated": len(status["adjudicated"]),
                               "sent": status["sent"], "recoverable": status["recoverable"],
                               "awaiting_adjudication": status["awaiting_adjudication"],
                               "unresolved": status["unresolved"]}
    return {"schema": "outbox_adjudication_listing.v1", "outbox": str(root), "spools": out}


def dispose(root: Path, name: str, decision: str, reason: str, *, successor: str | None = None) -> dict:
    GR = _module("governed_run")
    for spool in spools(root):
        if (Path(spool) / "pending" / name).is_file():
            outbox = GR.TerminalOutbox(spool)
            record = outbox.dispose(name, decision, reason, successor_terminal_sha256=successor)
            return {"spool": str(spool), "disposition": record,
                    "still_pending": len(outbox.status()["pending"])}
    raise SystemExit(f"REFUSED: {name} is not a pending envelope of any spool under {root}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "dispose"])
    ap.add_argument("--outbox-dir", default="~/.local/state/data-gov/terminal-outbox")
    ap.add_argument("--file")
    ap.add_argument("--decision", choices=["INVALID_ENVELOPE", "SUPERSEDED"])
    ap.add_argument("--reason")
    ap.add_argument("--successor-terminal-sha256")
    a = ap.parse_args(argv)
    root = Path(a.outbox_dir).expanduser()
    if a.command == "list":
        print(json.dumps(listing(root), indent=1, default=str))
        return 0
    if not (a.file and a.decision and a.reason):
        raise SystemExit("REFUSED: a disposition names the envelope, the decision and its reason")
    print(json.dumps(dispose(root, a.file, a.decision, a.reason, successor=a.successor_terminal_sha256),
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
