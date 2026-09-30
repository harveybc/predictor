#!/usr/bin/env python3
"""M06 closure comparison for the adopted Traffic TimeFilter L96 h96 cells.

Composes, only from retained cell records, the per-cell comparison row (exact metric, same-row
persistence naive, skill, published value, comparability) with the SAME executor that sealed the
cells (the staged df_tsl_execute.py, imported read-only from its staging directory), and, when all
three sealed seeds of h96 are present, the three-seed class through the executor's own classify().

Writes RESULTS/traffic_h96_rows.json, RESULTS/traffic_h96_results.csv and prints a table.
The row is redacted of host names (the record's environment block is not copied).
"""
import argparse
import csv
import hashlib
import json
import os
import sys

HOME = os.path.expanduser("~")
STAGED = "/tmp/traffic-gamma-staging/code"
ROOT = HOME + "/.local/state/crispdm-data-foundation/traffic_scored_codex_20260930"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="cell record JSON files (local copies)")
    ap.add_argument("--out", required=True, help="RESULTS directory")
    ap.add_argument("--staged", default=STAGED, help="directory holding tools/df_tsl_execute.py of the sealing executor")
    ap.add_argument("--design", default=ROOT + "/DESIGN.traffic.L96.json")
    a = ap.parse_args()
    STAGED_DIR = a.staged
    sys.path.insert(0, STAGED_DIR)
    from tools import df_tsl_execute as X  # noqa: E402  (read-only import of the sealing executor)
    design = json.load(open(a.design))
    rows, recs = [], []
    for p in a.records:
        rec = json.load(open(p))
        row = X.comparison_row(design, rec)          # refuses a record not under this design / bad digest
        row["record_file_sha256"] = sha(p)
        row["checkpoint_sha256"] = rec.get("checkpoint_sha256")
        rows.append(row)
        recs.append(rec)
    by_h = [r for r in recs if int(r["horizon_steps"]) == 96]
    seeds = sorted(int(r["seed"]) for r in by_h)
    pub = X.published_row(design)["per_horizon"]["96"]
    if seeds == sorted(design["seeds"]):
        cls = {m: X.classify(design, [r["metric"]["author_float32"][m] for r in by_h], float(pub[m]), m)
               for m in ("mse", "mae")}
        state = "THREE_SEED_CLASS_ASSIGNED"
    else:
        cls = None
        state = f"NO_CLASS_YET: {len(seeds)}/{len(design['seeds'])} sealed seeds present {seeds}; a per-seed row carries a difference, not a class"
    out = {"schema": "m06.traffic_h96_closure.v1", "design_sha256": design["design_sha256"],
           "executor_file_sha256": sha(STAGED_DIR + "/tools/df_tsl_execute.py"),
           "rows": rows, "h96_three_seed": {"state": state, "classes": cls}}
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, ".traffic_h96_rows.json.tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "traffic_h96_rows.json"))
    cols = ["task", "split", "rows_windows", "elements", "model", "seed", "metric_space", "reduction",
            "mse", "mae", "naive_mse", "naive_mae", "skill_mse", "skill_mae", "published_mse", "published_mae",
            "diff_mse", "diff_mae", "comparability", "evidence_state", "wall_seconds", "device_uuid",
            "checkpoint_sha256", "record_sha256"]
    with open(os.path.join(a.out, "traffic_h96_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["traffic L96->h96", r["split"], r["dataset_identity"]["windows"], r["dataset_identity"]["elements"],
                        "TimeFilter (author code dffde87e)", r["seed"], r["metric"]["space"], "mean over all window x step x channel elements",
                        r["model_error"]["mse"], r["model_error"]["mae"], r["paired_naive"]["mse"], r["paired_naive"]["mae"],
                        r["skill_vs_naive"]["mse"], r["skill_vs_naive"]["mae"], r["published"]["mse"], r["published"]["mae"],
                        r["difference_vs_published"]["mse"], r["difference_vs_published"]["mae"],
                        r["comparability"]["class"] + " (per-seed row: difference only)",
                        "measured (NOT_INDEPENDENTLY_VERIFIED)", r["resources"]["wall_seconds"],
                        r["resources"]["device_uuid"], r["checkpoint_sha256"], r["from_record_sha256"]])
    for r in rows:
        print(f"seed {r['seed']}: MSE {r['model_error']['mse']:.10f} MAE {r['model_error']['mae']:.10f} | naive MSE "
              f"{r['paired_naive']['mse']:.10f} MAE {r['paired_naive']['mae']:.10f} | skill MAE {r['skill_vs_naive']['mae']:.4f} | "
              f"published {r['published']['mse']}/{r['published']['mae']} | diff {r['difference_vs_published']['mse']:+.5f}/"
              f"{r['difference_vs_published']['mae']:+.5f} | wall {r['resources']['wall_seconds']:.0f}s")
    print(state)
    if cls:
        for m, c in cls.items():
            print(m, c["class"], "mean", c["mean"], "sd", c["seed_sd_ddof1"], "diff", c["difference"], "tol", c["tolerance_agreement"])


if __name__ == "__main__":
    main()
