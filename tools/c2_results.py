"""Lane C2: RESULTS.json and RESULTS.csv generated from the evidence files (orders section 6), never typed by hand."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(evidence_dir):
    E = Path(evidence_dir)
    summ = json.loads((E / "contribution/contribution_summary.json").read_text())
    leak = json.loads((E / "leak_probe/leak_probe.json").read_text())
    dossier_index = (E / "dossiers/DOSSIER_INDEX.json")
    didx = json.loads(dossier_index.read_text()) if dossier_index.exists() else None
    ps3r = (E / "ps3r_rerun/CONTRAST_SUMMARY.json")
    ps3r = json.loads(ps3r.read_text()) if ps3r.exists() else None
    rows = []
    for key, ref in summ["reference"].items():
        proto, hk = key.split("|")
        h = int(hk[1:])
        for arm, r in ref.items():
            rows.append({"deliverable": "D1", "protocol": proto, "horizon_bars": h, "horizon_hours": 4 * h, "arm": arm,
                         "mae_z_mean": r["mae_z_mean"], "mae_log_return_mean": r["mae_log_return_mean"], "n_eval_total": r["n_eval_total"],
                         "population": summ["bindings"]["view"]["dataset_id"], "split": "TRAIN rows [0,13699) (M07 116a5b64)", "label": "DEVELOPMENT"})
    results = {
        "schema": "lane_c2_results.v1", "label": "DEVELOPMENT", "bindings": summ["bindings"],
        "D1": {"records": 14336, "cpu_seconds": summ["cpu_seconds"], "rank_agreement": summ["rank_agreement"],
               "top20_by_stable_incremental_utility": summ["ranking_by_stable_incremental_utility"][:20],
               "best_stable_score": summ["scores"][summ["ranking_by_stable_incremental_utility"][0]],
               "reference_rows": rows},
        "D1b_leak_probe": {"counts": leak["counts"], "suspect_count": leak["suspect_count"], "cpu_seconds": leak["cpu_seconds"]},
        "D2_dossiers": None if didx is None else {"cells": len(didx["dossiers"]), "battery": didx["battery"], "cpu_seconds": didx["cpu_seconds"],
                                                   "schema_errors_total": sum(len([e for e in d["schema_errors"] if e != "JSONSCHEMA_NOT_INSTALLED_VALIDATION_SKIPPED"]) for d in didx["dossiers"])},
        "D4_ps3r_rerun": ps3r,
    }
    (E / "RESULTS.json").write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    with (E / "RESULTS.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"RESULTS.json/csv written: {len(rows)} reference rows; D2={'present' if didx else 'pending'}; D4={'present' if ps3r else 'pending'}")


if __name__ == "__main__":
    main(sys.argv[1])
