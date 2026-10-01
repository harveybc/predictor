#!/usr/bin/env python3
"""Artefacts for every VERIFIED cell of a lane F2 campaign (idempotent; receipts only).

For each candidate whose status is ``verified`` and whose verification is an exact match:
``EVIDENCE_<cid8>.json`` (predictor.forecast_naive_evidence.v1, strict naive = per-horizon
minimum, zero-return mandatory bar; ``tools.eth_forecast_naives.strict_record``),
``PREDICTIONS_<cid8>.csv`` (one row per validation origin, built from the float32 bytes whose
sha256 the independent scorer reproduced), and an ``ARTIFACTS_INDEX.json`` with every path,
sha256, cell label, seed and objective. Then the closure table for the whole queue set
(``tools.eth_forecast_closure``). Cells that already have both files are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from tools import eth_forecast_closure as cl
from tools import eth_forecast_naives as nv
from tools import modular_forecast_evidence as mfe


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_cells(queue):
    db = sqlite3.connect(f"file:{queue}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    out = []
    for row in db.execute("SELECT * FROM candidates WHERE status='verified' ORDER BY position"):
        train = db.execute("SELECT receipt_path FROM attempts WHERE cid=? AND kind='train' AND status='completed' "
                           "ORDER BY attempt DESC LIMIT 1", (row["cid"],)).fetchone()
        verify = db.execute("SELECT receipt_path FROM attempts WHERE cid=? AND kind='verify' AND verdict='VERIFIED' "
                            "ORDER BY attempt DESC LIMIT 1", (row["cid"],)).fetchone()
        if not train or not verify:
            continue
        verification = json.loads(Path(verify["receipt_path"]).read_text())
        if verification.get("exact_match") is not True:
            continue
        out.append({"cid": row["cid"], "label": row["label"], "seed": row["seed"], "config_id": row["config_id"],
                    "receipt_path": train["receipt_path"], "verification": verification, "queue": str(queue)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seasonal-period", type=int, default=6)
    parser.add_argument("--asset", default="ETHUSDT 4h spot (trading asset; DEVELOPMENT validation 2024)")
    parser.add_argument("--family", default=None, help="prediction family label (hourly | daily) for FX records")
    parser.add_argument("--clock", default=None, help="source clock declared in the dataset manifest")
    parser.add_argument("--seeds", default="2021,2022")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, _ = nv._load_validation(args.validation)
    mu, sigma = nv._manifest_scaler(args.manifest, data)
    table = nv.naive_table(data, mu, sigma, args.seasonal_period)
    index_path = out / "ARTIFACTS_INDEX.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema": "f2.artifacts_index.v1", "cells": {}}
    made = []
    for queue in args.queue:
        for cell in verified_cells(queue):
            cid8 = cell["cid"][:8]
            ev_path, csv_path = out / f"EVIDENCE_{cid8}.json", out / f"PREDICTIONS_{cid8}.csv"
            if ev_path.exists() and csv_path.exists() and cid8 in index["cells"]:
                continue
            receipt = json.loads(Path(cell["receipt_path"]).read_text())
            record = mfe.build(receipt, args.validation, campaign_id=args.campaign_id,
                               asset=args.asset)
            if args.family:
                record["population"]["family"] = args.family
            if args.clock:
                record["population"]["clock"] = args.clock
            strict = nv.strict_record(record, table, seasonal_period=args.seasonal_period)
            strict["cell"] = {"label": cell["label"], "seed": cell["seed"], "config_id": cell["config_id"],
                              "verification_exact_match": True,
                              "verification_predictions_sha256": cell["verification"]["digests"]["predictions_sha256"],
                              "label_status": "DEVELOPMENT"}
            strict["evidence_sha256"] = hashlib.sha256(mfe.canonical(
                {k: v for k, v in strict.items() if k != "evidence_sha256"}).encode()).hexdigest()
            ev_path.write_text(json.dumps(strict, indent=1, sort_keys=True) + "\n")
            pred = np.load(receipt["artifacts"]["predictions_validation"])
            info = nv.predictions_csv(pred, data, mu, sigma, args.view, csv_path,
                                      expected_sha256=cell["verification"]["digests"]["predictions_sha256"])
            index["cells"][cid8] = {"cid": cell["cid"], "label": cell["label"], "seed": cell["seed"],
                                    "objective_MAE_z": receipt["objective"]["value"],
                                    "evidence": {"path": ev_path.name, "sha256": sha_file(ev_path),
                                                 "evidence_sha256": strict["evidence_sha256"]},
                                    "predictions": {"path": csv_path.name, "sha256": info["sha256"],
                                                    "predictions_sha256": info["predictions_sha256"], "rows": info["rows"],
                                                    "row_ids_sha256": info["row_ids_sha256"]},
                                    "beats_zero_return_by_horizon": {str(e["horizon"]): e["beats_zero_return"]
                                                                     for e in strict["per_horizon"]},
                                    "queue": cell["queue"], "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            made.append(cid8)
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    closure = cl.closure(args.queue, sigma=sigma, seeds=tuple(int(s) for s in args.seeds.split(",")))
    (out / "CLOSURE.json").write_text(json.dumps(closure, indent=1, sort_keys=True) + "\n")
    cl.write_csv(closure, out / "CLOSURE.csv")
    print(json.dumps({"made": made, "cells_indexed": len(index["cells"]),
                      "configurations": {k: {"mean_MAE_z": v["mean_MAE_z"], "spread": v["spread"],
                                             "beats_zero_return_all": v["beats_zero_return_all_horizons_all_seeds"]}
                                         for k, v in closure["configurations"].items()}}, indent=1))


if __name__ == "__main__":
    main()
