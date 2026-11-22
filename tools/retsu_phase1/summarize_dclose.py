#!/usr/bin/env python
"""2024 CLOSE-change MAE against CLOSE persistence. Does not call evaluate.

Beating a gate makes a reconstructed price eligible. This script does not
write Prediction_h_* or Prediction_d_*.
"""
import json
from pathlib import Path

PHASE = Path("/var/tmp/retsu-phase1")
PERSIST = json.loads((PHASE / "followup" / "dclose_persistence.json").read_text())
QUEUES = {
    "seed7": Path("/var/tmp/retsu-phase1-dclose-seed7"),
    "pretrained": Path("/var/tmp/retsu-phase1-dclose-pretrained"),
}
GATES = {6: 0.0014153621159493923, 24: 0.0033699586056172848}


def index_persistence(rows: list[dict]) -> dict[int, float]:
    return {int(row["horizon"]): float(row["mae"]) for row in rows}


SHORT = index_persistence(PERSIST["holdout_persistence_short"])
LONG = index_persistence(PERSIST["holdout_persistence_long"])


def one(root: Path) -> list[dict]:
    jobs = []
    for path in sorted((root / "jobs").iterdir()):
        hold_path = path / "holdout.json"
        if not hold_path.exists():
            continue
        hold = json.loads(hold_path.read_text())
        per = {int(k): float(v) for k, v in (hold.get("holdout_2024_mae_per_horizon") or {}).items()}
        task = hold["task"]
        naive = SHORT if task == "short" else LONG
        horizons = []
        for horizon, mae in per.items():
            base = naive[horizon]
            horizons.append({
                "horizon": horizon,
                "mae_2024": mae,
                "close_persistence_2024": base,
                "delta_mae_minus_persistence": mae - base,
                "beats_persistence": mae < base,
                "beats_published_gate": horizon in GATES and mae < GATES[horizon],
            })
        es = hold.get("early_stopping") or {}
        rl = hold.get("reduce_lr") or {}
        jobs.append({
            "job": path.name,
            "task": task,
            "epoch": hold.get("epoch"),
            "best_epoch": es.get("best_epoch"),
            "es_wait": es.get("wait"),
            "es_patience": es.get("patience"),
            "lr": hold.get("lr"),
            "reductions": rl.get("reductions"),
            "train_mae": hold.get("train_mae"),
            "best_val_mae": hold.get("best_val_mae"),
            "holdout_2024_mae": hold.get("holdout_2024_mae"),
            "target_mean": hold.get("target_mean"),
            "target_std": hold.get("target_std"),
            "horizons": horizons,
        })
    return jobs


def main() -> None:
    payload = {
        "target": "CLOSE_future_minus_origin",
        "mae_unit": "close_change",
        "gates": {"h6": GATES[6], "h24": GATES[24]},
        "persistence_source": "followup/dclose_persistence.json",
        "prediction_columns_written": False,
        "queues": {},
    }
    beats = []
    for name, root in QUEUES.items():
        jobs = one(root)
        payload["queues"][name] = jobs
        for job in jobs:
            for row in job["horizons"]:
                if row["beats_published_gate"]:
                    beats.append({"encoder": name, "job": job["job"], **row})
    payload["beats_published_gate"] = beats
    payload["phase_closed_without_prediction_columns"] = len(beats) == 0
    (PHASE / "followup" / "dclose_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("beats", len(beats))
    for row in beats:
        print(row["encoder"], row["job"], "h", row["horizon"], "mae", round(row["mae_2024"], 6))
    print("DCLOSE_SUMMARY_OK")


if __name__ == "__main__":
    main()
