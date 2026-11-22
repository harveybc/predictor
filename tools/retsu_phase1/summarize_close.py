#!/usr/bin/env python
"""2024 CLOSE MAE against CLOSE persistence. Does not call evaluate."""
import json
from pathlib import Path

PHASE = Path("/var/tmp/retsu-phase1")
PERSIST = json.loads((PHASE / "followup" / "close_persistence.json").read_text())
QUEUES = {
    "seed7": Path("/var/tmp/retsu-phase1-close-seed7"),
    "pretrained": Path("/var/tmp/retsu-phase1-close-pretrained"),
}


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
            "horizons": horizons,
        })
    return jobs


def main() -> None:
    payload = {
        "target": "future_CLOSE",
        "mae_unit": "close_price",
        "gates": {"h6": 0.001415, "h24": 0.003370},
        "persistence_source": "followup/close_persistence.json",
        "queues": {},
    }
    beats = []
    for name, root in QUEUES.items():
        jobs = one(root)
        payload["queues"][name] = jobs
        for job in jobs:
            for row in job["horizons"]:
                if row["beats_persistence"]:
                    beats.append({"encoder": name, "job": job["job"], **row})
    payload["beats_persistence"] = beats
    payload["phase_closed_without_prediction_columns"] = len(beats) == 0
    (PHASE / "followup" / "close_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("beats", len(beats))
    for row in beats:
        print(row["encoder"], row["job"], "h", row["horizon"], "mae", round(row["mae_2024"], 6), "naive", round(row["close_persistence_2024"], 6))
    print("CLOSE_SUMMARY_OK")


if __name__ == "__main__":
    main()
