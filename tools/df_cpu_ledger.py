#!/usr/bin/env python3
"""CPU ledger of an order from the systemd user journal: every transient unit whose name matches the
given prefixes, with the CPU time systemd accounted at its end ("Consumed X CPU time"), summed
against a declared ceiling. Host-local: run it on every host that executed units and add the parts.

    python tools/df_cpu_ledger.py --since "2026-09-18 22:00" --match rp9 rp10 rp11 rp12 rp13 rp14 rp15 rp16 rp1011 --ceiling 14400 --out LEDGER.json
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
from pathlib import Path

LINE = re.compile(r"^(?P<stamp>\S+ \d+ [\d:]+) \S+ systemd\[\d+\]: (?P<unit>\S+): Consumed (?P<cpu>.+?) CPU time over (?P<wall>.+?) wall clock time(?:, (?P<mem>\S+) memory peak)?\.")
DUR = re.compile(r"(?:(?P<h>\d+)h )?(?:(?P<m>\d+)min )?(?:(?P<s>[\d.]+)s)?(?:(?P<ms>\d+)ms)?")


def seconds(text: str) -> float:
    m = DUR.fullmatch(text.strip())
    if not m:
        return float("nan")
    return (int(m.group("h") or 0) * 3600 + int(m.group("m") or 0) * 60 + float(m.group("s") or 0) + int(m.group("ms") or 0) / 1000.0)


def ledger(since: str, match: list, ceiling: float, journal_text: str | None = None) -> dict:
    if journal_text is None:
        journal_text = subprocess.run(["journalctl", "--user", "--since", since, "--no-pager"], capture_output=True, text=True, timeout=120).stdout
    rows = []
    for line in journal_text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        unit = m.group("unit")
        if not any(tag in unit for tag in match):
            continue
        rows.append({"unit": unit, "at": m.group("stamp"), "cpu_seconds": seconds(m.group("cpu")), "wall_seconds": seconds(m.group("wall")),
                     "memory_peak": m.group("mem")})
    total = float(sum(r["cpu_seconds"] for r in rows))
    return {"schema": "cpu_ledger.v1", "host": socket.gethostname(), "since": since, "match": match, "units": rows, "cpu_seconds_total": round(total, 3),
            "ceiling_seconds": ceiling, "remaining_seconds": round(ceiling - total, 3), "within_ceiling": total <= ceiling,
            "note": "systemd accounting of the transient unit (all its children); units still running are not listed"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True)
    parser.add_argument("--match", nargs="+", required=True)
    parser.add_argument("--ceiling", type=float, default=14400.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    doc = ledger(args.since, args.match, args.ceiling)
    if args.out:
        args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "units"} | {"units": len(doc["units"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
