#!/usr/bin/env python
"""2024 CLOSE-change MAE of the unfreeze arms against CLOSE persistence.

Does not call predict and does not write Prediction columns.
"""
import json
from pathlib import Path

PHASE = Path("/var/tmp/retsu-phase1")
PERSIST = json.loads((PHASE / "followup" / "close_persistence.json").read_text())
GATES = {6: 0.0014153621159493923, 24: 0.0033699586056172848}


def index(rows):
    return {int(row["horizon"]): float(row["mae"]) for row in rows}


SHORT = index(PERSIST["holdout_persistence_short"])
LONG = index(PERSIST["holdout_persistence_long"])


def main() -> None:
    payload = {"target": "CLOSE_future_minus_origin", "gates": {"h6": GATES[6], "h24": GATES[24]}, "jobs": []}
    beats = []
    for mode in ("frozen", "tuned"):
        root = Path(f"/var/tmp/retsu-phase1-unfreeze-change-{mode}/jobs")
        for path in sorted(root.iterdir()):
            hold = json.loads((path / "holdout.json").read_text())
            task = hold["task"]
            naive = SHORT if task == "short" else LONG
            per = {int(k): float(v) for k, v in hold["holdout_2024_mae_per_horizon"].items()}
            horizons = []
            for horizon, mae in per.items():
                base = naive[horizon]
                row = {
                    "horizon": horizon,
                    "mae_2024": mae,
                    "close_persistence_2024": base,
                    "beats_persistence": mae < base,
                    "beats_published_gate": horizon in GATES and mae < GATES[horizon],
                }
                horizons.append(row)
                if row["beats_published_gate"]:
                    beats.append({"mode": mode, "job": path.name, **row})
            es = hold.get("early_stopping") or {}
            rl = hold.get("reduce_lr") or {}
            payload["jobs"].append({
                "mode": mode,
                "job": path.name,
                "epoch": hold.get("epoch"),
                "best_epoch": es.get("best_epoch"),
                "es_wait": es.get("wait"),
                "lr": hold.get("lr"),
                "reductions": rl.get("reductions"),
                "train_mae": hold.get("train_mae"),
                "best_val_mae": hold.get("best_val_mae"),
                "holdout_2024_mae": hold.get("holdout_2024_mae"),
                "horizons": horizons,
            })
    payload["beats_published_gate"] = beats
    payload["prediction_columns_written"] = False
    (PHASE / "followup" / "unfreeze_change_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("beats", len(beats))
    print("UNFREEZE_CHANGE_SUMMARY_OK")


if __name__ == "__main__":
    main()
