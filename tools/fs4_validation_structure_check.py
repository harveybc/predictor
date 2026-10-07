#!/usr/bin/env python3
"""No-score structural check of the prepared VALIDATION-2024 parquets (receipt: satoshi/fs4-data-20261007 @46bca39b).

1. sha256 of every receipt file present on this host (EURUSD 40 files under validation_2024/ps1, ETH 2 files under validation_2024/eth)
   equals the receipt.
2. The weekly wrapper's own loader (``DataStore.from_paths``) opens TRAIN + VALIDATION and the check reports ONLY structure: rows,
   row-id namespace offset, time bounds, row spacing, sealed-calendar week coverage (rows per week, never a value), plan-member column
   coverage, typed censoring counts (non-finite targets), and that nothing at or after 2025-01-01 is present.
No prediction, no metric, no target value is computed or printed, and TEST is never read. Run it under crispdm-run on one host at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs4_weekly_wrapper as WW  # noqa: E402
from tools.fs4_candidates import canonical  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_receipt_files(receipt: dict, root: Path, population: str) -> dict:
    if population == "EURUSD":
        expected = {f"validation_2024/{k}": v for k, v in receipt["B_eurusd_rsync"]["sha256"].items()}
    else:
        expected = {f"validation_2024/eth/{k}": v["sha256"] for k, v in receipt["A_eth"]["files"].items()}
    missing, mismatched, ok = [], [], 0
    for rel, want in sorted(expected.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != want:
            mismatched.append(rel)
        else:
            ok += 1
    return {"expected": len(expected), "verified": ok, "missing": missing, "mismatched": mismatched, "all_equal_receipt": ok == len(expected)}


def structure(population: str, store: "WW.DataStore", weeks: list[dict], members: set[str]) -> dict:
    ts = store.ts
    validation_start = int(WW._parse(weeks[0]["start"]).timestamp())
    in_val = ts >= validation_start
    after = int(WW._parse("2025-01-01T00:00:00Z").timestamp())
    per_week = []
    for w in weeks:
        idx = store.range_idx(w["start"], w["end"])
        cens = {t: int(np.sum(~np.isfinite(store.targets[t][idx]))) for t in sorted(store.targets)}
        per_week.append({"week_start": w["start"], "rows": int(idx.size), "max_target_censored_rows": max(cens.values()) if cens else 0})
    tail = store.range_idx(weeks[-1]["end"], "2025-01-01T00:00:00Z")
    return {"population": population, "rows_total": int(len(ts)), "rows_validation": int(in_val.sum()), "row_id_offset": store.row_id_offset,
            "row_ids_unique": int(len(set(store.row_ids.tolist()))) == int(len(store.row_ids)),
            "first_validation_ts": WW._iso(__import__("datetime").datetime.fromtimestamp(int(ts[in_val][0]), WW.UTC)),
            "last_ts": WW._iso(__import__("datetime").datetime.fromtimestamp(int(ts[-1]), WW.UTC)),
            "rows_at_or_after_2025": int(np.sum(ts >= after)), "rows_after_last_week_before_2025": int(tail.size),
            "weeks_expected": len(weeks), "weeks_with_zero_rows": [p["week_start"] for p in per_week if p["rows"] == 0],
            "rows_per_week_min_max": [min(p["rows"] for p in per_week), max(p["rows"] for p in per_week)],
            "weeks_with_censored_target_rows": {p["week_start"]: p["max_target_censored_rows"] for p in per_week if p["max_target_censored_rows"]},
            "feature_columns": len(store.names), "plan_members_checked": len(members), "plan_members_missing": sorted(members - set(store.names)),
            "targets": sorted(store.targets), "scored": False, "metrics_computed": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--state-root", required=True, help="$HOME/.local/state/canonical_20261003")
    ap.add_argument("--population", choices=("EURUSD", "ETH"), required=True)
    ap.add_argument("--consolidated", required=True)
    ap.add_argument("--train-features", nargs="+", required=True)
    ap.add_argument("--train-targets", required=True)
    ap.add_argument("--bar-hours", type=int, required=True)
    ap.add_argument("--skip-structure", action="store_true", help="hash check only (hosts that only hold the files)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    receipt = json.loads(Path(a.receipt).read_text())
    root = Path(a.state_root)
    out = {"schema": "fs4.validation_structure_check.v1", "population": a.population, "receipt_schema": receipt.get("schema"),
           "sha256": verify_receipt_files(receipt, root, a.population), "test_read": False}
    if not out["sha256"]["all_equal_receipt"]:
        out["status"] = "SHA256_MISMATCH_OR_MISSING"
    elif a.skip_structure:
        out["status"] = "SHA256_VERIFIED"
    else:
        if a.population == "EURUSD":
            vf = [str(root / f"validation_2024/ps1/batch_00{i}/features_train.parquet") for i in (1, 2, 3)]
            vt = str(root / "validation_2024/ps1/batch_001/targets_train.parquet")
        else:
            vf, vt = [str(root / "validation_2024/eth/features_train.parquet")], str(root / "validation_2024/eth/targets_train.parquet")
        store = WW.DataStore.from_paths(a.population, a.train_features, a.train_targets, vf, vt, bar_hours=a.bar_hours)
        cons = json.loads(Path(a.consolidated).read_text())
        weeks = [{"start": w["start"], "end": w["end"]} for w in WW.build_plan([cons], [_dummy_seal(cons)], validation_year=2024,
                                                                                bar_hours={cons["population_id"]: a.bar_hours})["weeks"]]
        members = {m for s in cons["sets"] for m in s["members"]}
        out["structure"] = structure(a.population, store, weeks, members)
        s = out["structure"]
        out["status"] = ("STRUCTURE_OK" if s["row_ids_unique"] and not s["weeks_with_zero_rows"] and not s["plan_members_missing"]
                         and s["rows_at_or_after_2025"] == 0 else "STRUCTURE_FINDINGS")
    text = canonical(out)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(text)
    return 0 if out["status"] in ("STRUCTURE_OK", "SHA256_VERIFIED") else 3


def _dummy_seal(cons: dict) -> dict:
    """A seal-shaped object only to enumerate the calendar (weeks do not depend on it); never written or used for selection."""
    return {"population_id": cons["population_id"], "frontier_set_ids": [cons["sets"][0]["set_id"]], "denominator": {"sets": 1, "in_frontier": 1, "deferred": 0},
            "seal_sha256": "0" * 64, "rule_sha256": "0" * 64, "inputs": {"consolidated_sha256": cons["consolidated_sha256"],
                                                                         "extractibility": {"closure_sha256": None}}}


if __name__ == "__main__":
    sys.exit(main())
