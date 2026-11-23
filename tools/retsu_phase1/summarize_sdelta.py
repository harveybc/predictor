#!/usr/bin/env python
"""2024 seasonal-increment MAE against seasonal persistence. No second predict."""
import json
from pathlib import Path

PHASE = Path("/var/tmp/retsu-phase1")
PERSIST = json.loads((PHASE / "followup" / "sdelta_persistence.json").read_text())


def index(rows):
    return {int(row["horizon"]): float(row["mae"]) for row in rows}


SHORT = index(PERSIST["holdout_short"])
LONG = index(PERSIST["holdout_long"])


def main() -> None:
    payload = {"target": "causal_stl_seasonal_increment", "jobs": [], "misses": []}
    for mode in ("frozen", "tuned"):
        root = Path(f"/var/tmp/retsu-phase1-sdelta-{mode}/jobs")
        for path in sorted(root.iterdir()):
            hold = json.loads((path / "holdout.json").read_text())
            naive = SHORT if hold["task"] == "short" else LONG
            per = {int(k): float(v) for k, v in hold["holdout_2024_mae_per_horizon"].items()}
            horizons = []
            for horizon, mae in per.items():
                base = naive[horizon]
                row = {
                    "horizon": horizon,
                    "mae_2024": mae,
                    "seasonal_persistence_2024": base,
                    "beats_persistence": mae < base,
                }
                horizons.append(row)
                if not row["beats_persistence"]:
                    payload["misses"].append({"mode": mode, "job": path.name, "horizon": horizon, "mae_2024": mae, "persistence": base})
            es = hold.get("early_stopping") or {}
            payload["jobs"].append({
                "mode": mode,
                "job": path.name,
                "epoch": hold.get("epoch"),
                "best_epoch": es.get("best_epoch"),
                "es_wait": es.get("wait"),
                "lr": hold.get("lr"),
                "train_mae": hold.get("train_mae"),
                "best_val_mae": hold.get("best_val_mae"),
                "holdout_2024_mae": hold.get("holdout_2024_mae"),
                "horizons": horizons,
            })
    payload["all_horizons_beat_persistence"] = len(payload["misses"]) == 0
    (PHASE / "followup" / "sdelta_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("misses", len(payload["misses"]))
    print("SDELTA_SUMMARY_OK")


if __name__ == "__main__":
    main()
