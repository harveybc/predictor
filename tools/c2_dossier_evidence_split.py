"""Lane C2: add `evidence_classes` and `time_semantics` to every dossier (owner order MAINLINE_PARALLEL_CONTINUATION 4.C.2/4.C.4).

Derived from what each dossier already holds; nothing is re-estimated. association = rung 1 (+ the calibrated interval),
observed natural interventions = NOT_AVAILABLE (no quasi-experiment exists for a derived feature of one price history),
paired counterfactuals = REPORTED_UNDER_DECLARED_MODEL when rung 3 carries a model-based paired row, else NOT_EVALUATED.
Time: event time = the bar-close span of the scored origins; availability time = UNDECLARED (the view declares no timestamp
semantics, FEATURE_DAG.v3), never replaced by the bar stamp; population = the origin rule, row count and split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def derive(doc: dict) -> dict:
    r2, r3 = doc["rung2"], doc["rung3"]
    sens = r2.get("sensitivity", {})
    dm = doc["data_manifest"]
    app = dm["asset_appearance"]
    assoc = {"state": "REPORTED_ASSOCIATION" if doc["rung1"]["state"] == "ASSOCIATION_REPORTED" else "NOT_EVALUATED",
             "assumptions": ["held-out within TRAIN blocks with embargo; association is predictive, not causal", "stationarity within blocks is NOT assumed"],
             "support": {"effective_n": doc["rung1"].get("effective_n", 0), "interval": "block bootstrap" if "theta_se_block_bootstrap_L" in sens else "HAC h+6 (uncalibrated)",
                         "block_length": sens.get("calibration_block_length")}}
    nat = {"state": "NOT_AVAILABLE", "assumptions": [], "support": {}, "reason": "no natural or quasi-experiment exists for a derived feature of one price history; an RSI is not set without moving prices"}
    has_pair = r3.get("prediction", {}).get("model_based") is not None
    pair = {"state": "REPORTED_UNDER_DECLARED_MODEL" if has_pair else "NOT_EVALUATED",
            "assumptions": [k for k, v in r2.get("assumptions_declared", {}).items() if v] + ["partially linear effect", "additive noise"] if has_pair else [],
            "support": {"state": r2["support"]["state"], "residual_variance_share": r2["support"].get("residual_variance_share", 0.0)},
            "reason": "paired model-based counterfactual on the same row; label MODEL_BASED_COUNTERFACTUAL, retrospective only"}
    first, last = app["period"]
    avail = {"state": "UNDECLARED", "rule": "the view declares no timestamp semantics (bar open, bar close or publication); bar-close is an uncertified working assumption"}
    time = {"event_time": {"definition": "bar stamp of the scored origin t (DATE_TIME of the view, undeclared semantics)", "span": [first, last]},
            "availability_time": avail,
            "population": {"rule": doc["subject"]["population"], "rows": int(dm["n_episodes"]), "split": f"TRAIN rows [{app['train_rows'][0]},{app['train_rows'][1]})"}}
    return {"evidence_classes": {"association": assoc, "observed_natural_interventions": nat, "paired_counterfactuals": pair}, "time_semantics": time}


def main(directory):
    d = Path(directory)
    n = 0
    for f in sorted(d.glob("c2-eth4h-*.json")):
        doc = json.loads(f.read_text())
        doc.update(derive(doc))
        f.write_text(json.dumps(doc, indent=1, sort_keys=True))
        n += 1
    print("split added to", n, "dossiers")


if __name__ == "__main__":
    main(sys.argv[1])
