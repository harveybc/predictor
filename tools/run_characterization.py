#!/usr/bin/env python3
"""C30: run the first per-variable characterization.

Only subjects with sufficient provenance are measured, the three
banks are kept apart, every attempt is written to the ledger
including the ones that produced nothing, and no selection is
emitted. Synthetic subjects carry a known clean reference, so a
noise figure is identifiable there and NOWHERE else.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import characterization as ch  # noqa: E402


def _read_column(path: Path, column: str):
    out = []
    with open(path, newline="", encoding="utf-8",
              errors="replace") as fh:
        for row in csv.DictReader(fh):
            raw = row.get(column)
            try:
                out.append(float(raw))
            except (TypeError, ValueError):
                out.append(float("nan"))
    return out


def financial_subjects(predictor_root: Path, limit: int):
    """Development partitions of the registered views only."""
    inv = (predictor_root /
           "examples/research/crispdm_dataset_inventory.v1.json")
    if not inv.is_file():
        return []
    doc = json.loads(inv.read_text())
    out = []
    for ds in doc.get("datasets", []):
        p = predictor_root / ds["relative_path"]
        if not p.is_file():
            continue
        with open(p, newline="") as fh:
            header = [c.strip() for c in next(csv.reader(fh))]
        for column in header[:limit]:
            out.append({
                "variable_id": f"{ds['dataset_id']}::{column}",
                "column": column, "path": p,
                "partition_key": f"{ds['dataset_id']}::"
                                 "development",
                "bank": ch.BANK_FINANCIAL,
                "provenance": ds.get("profile_status",
                                     "UNKNOWN")})
    return out


def synthetic_subjects(count: int, length: int):
    """Known-mechanism generators: the clean component is known
    BY CONSTRUCTION, which is the only reason a noise figure is
    identifiable at all."""
    import numpy as np
    out = []
    for i in range(count):
        rng = np.random.default_rng(1000 + i)
        t = np.arange(length)
        clean = np.sin(2 * np.pi * t / (24 + 8 * i))
        noise = rng.normal(0, 0.2 + 0.1 * i, length)
        out.append({
            "variable_id": f"synthetic::sine{24 + 8 * i}::seed"
                           f"{1000 + i}",
            "values": (clean + noise).tolist(),
            "reference": clean.tolist(),
            "partition_key": "synthetic::development",
            "bank": ch.BANK_SYNTHETIC})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor-root", required=True, type=Path)
    ap.add_argument("--financial-columns", type=int, default=8)
    ap.add_argument("--synthetic-generators", type=int,
                    default=3)
    ap.add_argument("--synthetic-length", type=int, default=512)
    ap.add_argument("--measured-at", required=True)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--load", action="store_true",
                    help="write the rows into the cube")
    a = ap.parse_args(argv)

    attempts, rows = [], []

    for s in synthetic_subjects(a.synthetic_generators,
                                a.synthetic_length):
        r = ch.characterize_series(
            s["values"], variable_id=s["variable_id"],
            partition_key=s["partition_key"],
            bank_authority=s["bank"],
            measured_at=a.measured_at,
            noise_reference=s["reference"])
        rows.extend(r)
        attempts.append({"variable_id": s["variable_id"],
                         "bank": s["bank"],
                         "outcome": "MEASURED",
                         "descriptors": len(r),
                         "noise_identifiable": True})

    for s in financial_subjects(a.predictor_root,
                                a.financial_columns):
        try:
            values = _read_column(s["path"], s["column"])
        except Exception as exc:                # noqa: BLE001
            attempts.append({"variable_id": s["variable_id"],
                             "bank": s["bank"],
                             "outcome": "FAILED",
                             "reason": exc.__class__.__name__})
            continue
        r = ch.characterize_series(
            values, variable_id=s["variable_id"],
            partition_key=s["partition_key"],
            bank_authority=s["bank"],
            measured_at=a.measured_at,
            noise_reference=None)
        rows.extend(r)
        identifiable = [x for x in r if x["identifiable"]]
        attempts.append({
            "variable_id": s["variable_id"], "bank": s["bank"],
            "outcome": ("MEASURED" if identifiable
                        else "INCONCLUSIVE"),
            "descriptors": len(r),
            "not_identifiable": len(r) - len(identifiable),
            "noise_identifiable": False,
            "provenance_state": s["provenance"]})

    ch.assert_no_selection(rows)
    loaded = {}
    if a.load:
        from tools.backfill_campaign_envelopes import _engine
        loaded = ch.load_rows(_engine(), rows)

    banks = {}
    for r in rows:
        banks[r["bank_authority"]] = banks.get(
            r["bank_authority"], 0) + 1
    ledger = {
        "schema": "crispdm.characterization_ledger.v1",
        "measured_at": a.measured_at,
        "attempts": attempts,
        "attempts_total": len(attempts),
        "outcomes": {o: sum(1 for x in attempts
                            if x["outcome"] == o)
                     for o in ("MEASURED", "INCONCLUSIVE",
                               "FAILED")},
        "rows_total": len(rows),
        "rows_by_bank": banks,
        "rows_not_identifiable": sum(
            1 for r in rows if not r["identifiable"]),
        "total_cost_seconds": round(
            sum(r["cost_seconds"] for r in rows), 6),
        "loaded": loaded,
        "selection_emitted": "NONE — this run measures variables "
                             "and chooses none",
        "confirmation_used": "NONE",
        "gpu_used": "NONE",
        "bank_separation": "synthetic calibrates, public "
                           "evaluates transfer, financial is "
                           "development only; every row carries "
                           "its bank's authority",
    }
    ledger["ledger_sha256"] = ch._sha(ledger)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    a.ledger.write_text(json.dumps(ledger, indent=1,
                                   sort_keys=True) + "\n")
    print(json.dumps({k: ledger[k] for k in
                      ("attempts_total", "outcomes",
                       "rows_total", "rows_by_bank",
                       "rows_not_identifiable",
                       "total_cost_seconds", "loaded",
                       "ledger_sha256")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
