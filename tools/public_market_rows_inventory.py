#!/usr/bin/env python3
"""What is actually published, measured from the published revision's own bytes.

U3 of `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md`:

    "The reported 47233 rows are not independently verified by this review. Produce a
     file-level inventory of the six public CSVs: repository, published revision, path, digest,
     header, row count, inferred/declared producer and actual evidence supporting that producer.
     Distinguish rows from unique observations and examples from lake resources. Do not assume
     a heading establishes provenance."

Two distinctions this makes and the earlier count did not:

* **rows are not observations.** A file with one row per bar per instrument, or with a derived
  features table over the same bars, counts the same observation more than once. Unique
  (timestamp) and (timestamp, symbol) pairs are counted separately from lines;
* **a published example is not a lake resource.** These files are committed into a public
  repository; the governed lake resource is a different artefact, and conflating them is how
  "no market rows are public" became both wrong and unfalsifiable.

Everything is read from `git show <revision>:<path>` — the bytes as published, not the working
tree, which may differ.

usage:
  public_market_rows_inventory.py --repo DIR --revision REV --out INVENTORY.json PATH [PATH ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def published_bytes(repo: Path, revision: str, path: str) -> bytes | None:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                         capture_output=True)
    return out.stdout if out.returncode == 0 else None


def time_column(header: list[str]) -> str | None:
    for candidate in ("DateTime", "Date", "DATE_TIME", "open_time", "timestamp", "time"):
        for name in header:
            if name.strip().lower() == candidate.lower():
                return name
    return None


def symbol_column(header: list[str]) -> str | None:
    for name in header:
        if name.strip().lower() in ("symbol", "ticker", "pair", "instrument"):
            return name
    return None


def measure(raw: bytes) -> dict:
    """Lines, rows, and how many DISTINCT observations those rows actually carry."""
    text = raw.decode("utf-8", "replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return {"header": None, "rows": 0}
    column = time_column(header)
    symbol = symbol_column(header)
    index = header.index(column) if column else None
    symbol_index = header.index(symbol) if symbol else None
    rows = 0
    stamps, pairs = set(), set()
    first = last = None
    for record in reader:
        if not record:
            continue
        rows += 1
        if index is not None and index < len(record):
            value = record[index]
            stamps.add(value)
            pairs.add((value, record[symbol_index] if symbol_index is not None
                       and symbol_index < len(record) else ""))
            if first is None:
                first = value
            last = value
    return {
        "header": header,
        "columns": len(header),
        "rows": rows,
        "time_column": column,
        "symbol_column": symbol,
        "distinct_timestamps": len(stamps) if column else None,
        "distinct_timestamp_symbol_pairs": len(pairs) if column else None,
        "first_timestamp": first,
        "last_timestamp": last,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="the PUBLISHED revision, e.g. origin/master")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    resolved = subprocess.run(["git", "-C", str(args.repo), "rev-parse", args.revision],
                              capture_output=True, text=True).stdout.strip()
    url = subprocess.run(["git", "-C", str(args.repo), "remote", "get-url", args.remote],
                         capture_output=True, text=True).stdout.strip()

    files, totals = [], {"rows": 0, "distinct_timestamp_symbol_pairs": 0}
    for path in args.paths:
        raw = published_bytes(args.repo, args.revision, path)
        entry = {"path": path, "present_at_revision": raw is not None}
        if raw is None:
            files.append(entry)
            continue
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), **measure(raw))
        files.append(entry)
        totals["rows"] += entry["rows"]
        if entry.get("distinct_timestamp_symbol_pairs"):
            totals["distinct_timestamp_symbol_pairs"] += entry[
                "distinct_timestamp_symbol_pairs"]

    body = {
        "schema": "public_market_rows_inventory.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"),
        "repository": {"path": str(args.repo), "remote": url,
                       "revision_requested": args.revision, "revision_sha": resolved},
        "files": files,
        "totals": totals,
        "note": ("Rows are lines of CSV. Distinct pairs are (timestamp, symbol) and are what a "
                 "count of observations means. Neither establishes a producer: provenance is "
                 "recorded separately, from evidence, not from a heading."),
    }
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"revision": resolved, "files": len(files), **totals}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
