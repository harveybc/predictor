#!/usr/bin/env python3
"""Read-only population pairing checks; no models, market data or credentials."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


FIELDS = ("support_hash", "val_hash", "rows_train", "rows_val", "seed", "k_folds")


def signature(row):
    return tuple(row.get(k) for k in FIELDS)


def audit(payload):
    latest, attempts = {}, 0
    for line in payload.splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["key"]] = row
            attempts += 1
    rows = list(latest.values())
    report = {"schema": "selection_support_audit.v1", "ledger_sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "attempt_rows": attempts, "unique_units": len(rows), "failed_units": sum("error" in r for r in rows), "stages": {}}
    for stage in ("contrast", "encoding"):
        groups = defaultdict(list)
        for row in rows:
            if row.get("stage") == stage and row.get("family") != "NONE" and "error" not in row:
                groups[(row.get("task"), row.get("model"), row.get("family"), row.get("ctx"), row.get("fold"))].append(row)
        bad, incomplete = [], []
        for key, group in groups.items():
            if len(group) < 2:
                incomplete.append(key)
                continue
            missing = sorted({f for r in group for f in FIELDS if r.get(f) is None})
            if missing or len({signature(r) for r in group}) > 1:
                bad.append({"group": key, "missing_fields": missing,
                            "arms": [{k: r.get(k) for k in ("key", "arm", *FIELDS)} for r in group]})
        report["stages"][stage] = {"groups_observed": len(groups), "groups_with_multiple_arms": len(groups) - len(incomplete),
                                    "mismatched_or_unverifiable_groups": len(bad), "witnesses": bad}
    controls = {(r.get("task"), r.get("fold")): r for r in rows
                if r.get("stage") == "contrast" and r.get("family") == "NONE" and r.get("ctx") == "core15" and "error" not in r}
    rescue = [r for r in rows if r.get("stage") == "rescue" and "error" not in r]
    bad = []
    for row in rescue:
        ref = controls.get((row.get("task"), row.get("fold")))
        if ref is None or signature(ref) != signature(row) or any(row.get(f) is None for f in FIELDS):
            bad.append({"key": row["key"], "reference_key": ref.get("key") if ref else None})
    report["stages"]["rescue_vs_context"] = {"comparisons": len(rescue), "mismatched_or_missing_controls": len(bad), "witnesses": bad}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(Path(args.ledger).read_text())
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {k: {f: v for f, v in d.items() if f != "witnesses"} for k, d in report["stages"].items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
