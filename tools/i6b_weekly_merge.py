#!/usr/bin/env python3
"""Merge disjoint I6-B worker shards and emit the verified I7 donor index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from tools import i6b_weekly_campaign as C


def merge_campaign(destination, sources, design):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = [Path(source) for source in sources]
    for ordinal in range(len(design["weeks"])):
        relative = Path(f"week_{ordinal:03d}")
        candidates = [source / relative for source in sources
                      if (source / relative / "WEEK_RECEIPT.json").is_file()]
        if not candidates:
            continue
        receipts = [C._load_receipt(path / "WEEK_RECEIPT.json", design, ordinal)
                    for path in candidates]
        if len({row["receipt_sha256"] for row in receipts}) != 1:
            raise ValueError(f"conflicting worker receipts for week {ordinal}")
        target = destination / relative
        if (target / "WEEK_RECEIPT.json").is_file():
            retained = C._load_receipt(target / "WEEK_RECEIPT.json", design, ordinal)
            if retained["receipt_sha256"] != receipts[0]["receipt_sha256"]:
                raise ValueError(f"destination conflicts at week {ordinal}")
            continue
        if target.exists():
            raise ValueError(f"uncommitted destination directory at week {ordinal}")
        shutil.copytree(candidates[0], target)
    status = C.campaign_status(destination, design)
    if status["status"] == "COMPLETE":
        index = C.build_donor_index(destination, design)
        status = {**status, "donor_index_sha256": index["index_sha256"]}
        C._atomic_json(destination / "CAMPAIGN_STATUS.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source", action="append", required=True)
    args = parser.parse_args(argv)
    result = merge_campaign(args.destination, args.source, C.load_design(args.design))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
