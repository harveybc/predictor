#!/usr/bin/env python3
"""R1/R2 contrast table and campaign closure table for the corrected campaign (from receipts only).

* R1/R2 table: every verified R1/R2 configuration paired BY SEED with the declared R0
  reference (and any superseded reference, both shown), aggregate MAE, paired difference
  with the two-seed spreads, a STRICT_MINIMUM label (never "advantage" unless the gap
  exceeds both spreads, and then still only stated with the numbers), per-horizon skill
  versus same-row persistence and versus the declared seasonal naive, the h1/h23/h24
  call-out, the pretraining cost column (not part of the selection rule) and every
  REFUSED_BY_ENGINE row by name.
* Closure table (owner closure-table rule): per verified configuration the model error
  with its scale, the same-row naive, skill, the literature value with its source or
  NOT_AVAILABLE with the reason, and comparability (NOT_COMPARABLE with the reason).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_doin_per_horizon_table as pht  # noqa: E402

SEASONS = (1, 23, 24)


def build(queue, validation, period, reference, superseded_reference, pretraining_cost, literature):
    table = pht.build(queue, validation, period)
    db = sqlite3.connect(f"file:{queue}?mode=ro", uri=True)
    refused = [{"label": r[0], "seed": r[1], "cid": r[2], "reason": r[3]} for r in db.execute(
        "select label, seed, cid, blocked_reason from candidates where status='REFUSED_BY_ENGINE' order by position")]
    by_config = {}
    for row in table["rows"]:
        by_config.setdefault(row["config_id"], []).append(row)
    ref = {r["seed"]: r for r in by_config.get(reference["config_id"], [])}
    contrasts = []
    for config_id, rows in by_config.items():
        label = rows[0]["label"]
        if "_R1_" not in label and "_R2_" not in label:
            continue
        seeds = {r["seed"]: r for r in rows}
        diffs = {s: seeds[s]["MAE"] - ref[s]["MAE"] for s in seeds if s in ref}
        spread = max(r["MAE"] for r in rows) - min(r["MAE"] for r in rows)
        ref_spread = max(r["MAE"] for r in ref.values()) - min(r["MAE"] for r in ref.values())
        mean_diff = sum(diffs.values()) / len(diffs) if diffs else None
        exceeds = mean_diff is not None and abs(mean_diff) > spread and abs(mean_diff) > ref_spread
        per_h = {}
        for h in range(1, 25):
            per_h[h] = {"skill_vs_persistence": [seeds[s].get(f"h{h}_skill_MAE") for s in sorted(seeds)],
                        "skill_vs_seasonal": [seeds[s].get(f"h{h}_skill_vs_seasonal") for s in sorted(seeds)]}
        contrasts.append({
            "config_id": config_id, "label": label, "regime": "R1" if "_R1_" in label else "R2",
            "seeds": {s: seeds[s]["MAE"] for s in sorted(seeds)},
            "mean": sum(r["MAE"] for r in rows) / len(rows), "spread": spread,
            "paired_difference_vs_R0": diffs, "mean_difference": mean_diff, "R0_spread": ref_spread,
            "label_rule": ("STRICT_MINIMUM; gap exceeds both two-seed spreads (stated with numbers, not proof)"
                           if exceeds else "STRICT_MINIMUM; gap within the two-seed spread"),
            "negative_skill_vs_persistence_at": {h: per_h[h]["skill_vs_persistence"] for h in SEASONS},
            "beats_seasonal_naive_any_horizon": any((v or -1) > 0 for h in per_h for v in per_h[h]["skill_vs_seasonal"]),
            "per_horizon": per_h, "pretraining_cost": pretraining_cost})
    closure = []
    for config_id, rows in by_config.items():
        mae = sum(r["MAE"] for r in rows) / len(rows)
        naive = sum(r["naive_MAE"] for r in rows) / len(rows)
        closure.append({"config_id": config_id, "label": rows[0]["label"], "seeds": len(rows),
                        "model_MAE": mae, "scale": "z_train (train-only StandardScaler), validation, mean over "
                        "2609 windows x 24 horizons x 321 channels", "naive_same_rows_MAE": naive,
                        "skill": 1 - mae / naive, "literature": literature,
                        "comparability": "NOT_COMPARABLE: L24/H1..24 on the validation split; published ECL rows "
                                         "use L96 with H96..720 on the test split"})
    closure.sort(key=lambda r: r["model_MAE"])
    return {"reference": reference, "superseded_reference": superseded_reference, "contrasts": contrasts,
            "refused_by_engine": refused, "closure": closure, "seasonal_period": period}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queue")
    parser.add_argument("out")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--seasonal-period", type=int, default=24)
    parser.add_argument("--meta", required=True, help="JSON with reference, superseded_reference, pretraining_cost, literature")
    args = parser.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    report = build(args.queue, args.validation, args.seasonal_period, meta["reference"], meta["superseded_reference"],
                   meta["pretraining_cost"], meta["literature"])
    Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"contrasts": [(c["label"], round(c["mean"], 10), c["mean_difference"]) for c in report["contrasts"]],
                      "refused": len(report["refused_by_engine"]), "closure_rows": len(report["closure"])}, default=str))


if __name__ == "__main__":
    main()
