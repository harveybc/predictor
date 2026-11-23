#!/usr/bin/env python
"""Tables for the causal seasonal-increment repeat. No second predict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE = Path("/var/tmp/retsu-phase1-causal")


def persistence(y, mask, horizons):
    rows = []
    for i, horizon in enumerate(horizons):
        ok = mask & np.isfinite(y[:, i])
        rows.append({"horizon": int(horizon), "rows": int(ok.sum()), "mae": float(np.mean(np.abs(y[ok, i])))})
    return rows


def index_rows(rows):
    return {int(row["horizon"]): row for row in rows}


def champion_line(path: Path, best_epoch: int) -> dict:
    found = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["epoch"]) == int(best_epoch):
                found = row
    if found is None:
        raise SystemExit(f"CHAMPION_LINE {path}")
    return found


def main() -> None:
    protocol = json.loads((PHASE / "protocol.json").read_text())
    data = np.load(PHASE / "dataset.npz")
    recomputed = {}
    for split in ("pool", "monitor", "holdout"):
        recomputed[split] = {
            "short": persistence(data["y_short"], data[split] & data["ok_short"], protocol["horizons_short"]),
            "long": persistence(data["y_long"], data[split] & data["ok_long"], protocol["horizons_long"]),
        }
    for split, tasks in recomputed.items():
        for task, rows in tasks.items():
            saved = protocol["naive_mae"][split][task]
            for got, ref in zip(rows, saved):
                if got["horizon"] != ref["horizon"] or got["rows"] != ref["rows"] or abs(got["mae"] - ref["mae"]) > 1e-6:
                    raise SystemExit(f"NAIVE_MISMATCH {split} {task} {got} {ref}")
    jobs = []
    misses = []
    for mode in ("frozen", "tuned"):
        root = Path(f"/var/tmp/retsu-phase1-causal-{mode}/jobs")
        for path in sorted(p for p in root.iterdir() if p.is_dir()):
            hold = json.loads((path / "holdout.json").read_text())
            task = hold["task"]
            best_epoch = int(hold["best_epoch"])
            line = champion_line(path / "epochs.jsonl", best_epoch)
            if abs(float(line["val_mae"]) - float(hold["champion_val_mae"])) > 1e-12:
                raise SystemExit(f"CHAMPION_VAL {path}")
            naive_val = index_rows(recomputed["monitor"][task])
            naive_test = index_rows(recomputed["holdout"][task])
            naive_train = index_rows(recomputed["pool"][task])
            horizons = []
            for horizon_text, mae in hold["holdout_2024_mae_per_horizon"].items():
                horizon = int(horizon_text)
                val = float(line["val_mae_per_horizon"][str(horizon)])
                row = {
                    "horizon_hours": horizon,
                    "train_rows": naive_train[horizon]["rows"],
                    "validation_h2_2023_rows": naive_val[horizon]["rows"],
                    "test_2024_rows": naive_test[horizon]["rows"],
                    "mae_model_validation_h2_2023": val,
                    "mae_naive_seasonal_persistence_validation_h2_2023": naive_val[horizon]["mae"],
                    "mae_model_test_2024": float(mae),
                    "mae_naive_seasonal_persistence_test_2024": naive_test[horizon]["mae"],
                    "model_beats_naive_on_test_2024": float(mae) < naive_test[horizon]["mae"],
                    "model_beats_naive_on_validation_h2_2023": val < naive_val[horizon]["mae"],
                }
                horizons.append(row)
                if not row["model_beats_naive_on_test_2024"]:
                    misses.append({"mode": mode, "job": path.name, "horizon_hours": horizon})
            jobs.append(
                {
                    "mode": mode,
                    "job": path.name,
                    "encoder": "seed-7 fresh build; encoder frozen" if mode == "frozen" else "seed-7 fresh build; encoder updated",
                    "recipe_identity": hold.get("recipe_identity"),
                    "last_epoch": hold.get("epoch"),
                    "champion_epoch": best_epoch,
                    "early_stopping_wait": (hold.get("early_stopping") or {}).get("wait"),
                    "learning_rate_after_restore": hold.get("lr"),
                    "mae_train_champion_mean_across_horizons": hold.get("champion_train_mae"),
                    "train_mae_note": "mean across horizons only; per-horizon train MAE was not logged",
                    "mae_validation_h2_2023_champion_mean_across_horizons": hold.get("champion_val_mae"),
                    "horizons": horizons,
                }
            )
    payload = {
        "target": "increment of the causal STL seasonal component",
        "naive": "seasonal-level persistence: forecast equals the seasonal value at the origin, which is a zero increment",
        "metric": "MAE, mean absolute error, in price units of the seasonal component",
        "inputs": "causal rebuild; 2024 was not used to choose columns, Kalman, GMM, the scaler, or the champion",
        "per_horizon_train_mae": "not logged",
        "jobs": jobs,
        "misses_on_test_2024": misses,
        "all_existing_horizons_beat_naive_on_test_2024": len(misses) == 0 and len(jobs) == 4,
    }
    follow = PHASE / "followup"
    follow.mkdir(parents=True, exist_ok=True)
    (follow / "causal_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Incremento estacional causal, repetición sin las dos fugas",
        "",
        "Métrica: MAE (error absoluto medio) del incremento de la componente estacional STL causal, en unidades de precio.",
        "Naive: persistencia del nivel estacional. El pronóstico es el valor estacional en el origen, es decir un incremento cero.",
        "Train: media del MAE entre horizontes en la época campeona. No hay MAE de train por horizonte.",
        "Validación: segundo semestre de 2023, época campeona. Test: 2024, un solo predict después de restaurar el campeón.",
        "",
    ]
    for job in jobs:
        lines.append(
            f"## {job['mode']} {job['job']} — época campeona {job['champion_epoch']}, "
            f"MAE train (media entre horizontes) {job['mae_train_champion_mean_across_horizons']:.6g}, "
            f"MAE validación media {job['mae_validation_h2_2023_champion_mean_across_horizons']:.6g}"
        )
        lines.append("")
        lines.append("| Horizonte (horas) | Filas train | Filas validación H2 2023 | Filas test 2024 | MAE modelo validación H2 2023 | MAE naive persistencia estacional validación H2 2023 | MAE modelo test 2024 | MAE naive persistencia estacional test 2024 | Modelo gana al naive en test 2024 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
        for row in job["horizons"]:
            lines.append(
                "| {horizon_hours} | {train_rows} | {validation_h2_2023_rows} | {test_2024_rows} | "
                "{mae_model_validation_h2_2023:.6g} | {mae_naive_seasonal_persistence_validation_h2_2023:.6g} | "
                "{mae_model_test_2024:.6g} | {mae_naive_seasonal_persistence_test_2024:.6g} | {win} |".format(
                    win="sí" if row["model_beats_naive_on_test_2024"] else "no",
                    **row,
                )
            )
        lines.append("")
    (follow / "causal_results.md").write_text("\n".join(lines) + "\n")
    print("misses", len(misses))
    print("CAUSAL_SUMMARY_OK")


if __name__ == "__main__":
    main()
