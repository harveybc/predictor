#!/usr/bin/env python3
"""End a running ``while read … done < PLAN`` driver loop at its next cell boundary.

bash's ``read`` builtin reads a seekable file at the position after the last consumed
line, so truncating the plan IN PLACE (same inode; ``os.replace`` would NOT work) to the
bytes already consumed makes the next ``read`` hit EOF: the loop exits after the cell
it is running right now, logs ``DONE`` and the service becomes inactive. The running cell
is a child process of a different launcher and is never touched. The consumed prefix is
kept byte-identical; the file is written under a lock-free single ``O_TRUNC`` + ``write``,
which is safe because the only reader is blocked on its current cell.

The number of consumed lines is derived from the driver's log: the last ``BEGIN`` line
names the current cell, which must be line ``k`` of the plan; the prefix is lines 1..k.
Refuses when the current cell is not found, when it appears more than once, or when the
log shows the loop already finished (``DONE`` after the last BEGIN).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

BEGIN = re.compile(r"^\S+\s+BEGIN\s+(?P<rest>.+)$")


def current_cell_from_log(lines: List[str], key_index: int = -1) -> Optional[str]:
    """Return the feature named by the last BEGIN line, or None if the loop is DONE.

    ``key_index`` selects which whitespace token of the BEGIN remainder is the feature
    (default: the last one, as every driver logs ``… <stage> <batch> <feature>``).
    """

    current = None
    for line in lines:
        text = line.strip()
        match = BEGIN.match(text)
        if match:
            current = match.group("rest").split()[key_index]
        elif re.search(r"\s(DONE|STOP file)\b", text):
            current = None
    return current


def consumed_prefix(plan_bytes: bytes, feature: str, key_index: int = -1) -> Tuple[bytes, int]:
    lines = plan_bytes.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.split() and line.split()[key_index].decode() == feature]
    if not hits:
        raise ValueError(f"current cell {feature!r} not found in plan")
    if len(hits) > 1:
        raise ValueError(f"current cell {feature!r} appears {len(hits)} times in plan")
    k = hits[0] + 1
    return b"".join(lines[:k]), k


def truncate_in_place(path: Path, prefix: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC)
    try:
        view = memoryview(prefix)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--backup", type=Path, help="write the full original plan here first")
    parser.add_argument("--apply", action="store_true", help="without it, only report what would happen")
    args = parser.parse_args(argv)

    data = args.plan.read_bytes()
    feature = current_cell_from_log(args.log.read_text(encoding="utf-8", errors="replace").splitlines())
    if feature is None:
        print("REFUSED: the log shows no running cell (loop DONE or never started)", file=sys.stderr)
        return 3
    try:
        prefix, k = consumed_prefix(data, feature)
    except ValueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    total = len(data.splitlines())
    print(f"current cell {feature} is plan line {k} of {total}; keeping {k} lines ({len(prefix)} bytes), dropping {total - k}")
    if not args.apply:
        return 0
    if args.backup:
        args.backup.write_bytes(data)
    truncate_in_place(args.plan, prefix)
    assert args.plan.read_bytes() == prefix
    print(f"APPLIED: {args.plan} truncated in place; the loop ends after {feature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
