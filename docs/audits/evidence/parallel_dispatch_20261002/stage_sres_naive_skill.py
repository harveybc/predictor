#!/usr/bin/env python3
"""Same-row naive + intercept-only controls on the ETH 4h 6..36 validation rows, and skill of a given objective.
Reads train.npz targets (intercept fit) and validation.npz only. Never reads test rows."""
import io, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools")
from tools import eth_forecast_naives as N
D = Path(sys.argv[1]); out = Path(sys.argv[2]); model_obj = [float(v) for v in sys.argv[3].split(",")]
val, _ = N._load_validation(D / "validation.npz")
mu, sigma = N._manifest_scaler(D / "MANIFEST.json", val)
tr = np.load(D / "train.npz", allow_pickle=False)
assert str(tr["split"]) == "train"
ytr = tr["targets"].astype(np.float64); yv = val["targets"].astype(np.float64)
H = val["horizons"].tolist()
table = N.naive_table(val, mu, sigma, 6)
res = {"validation_rows": int(len(yv)), "train_rows_for_intercept": int(len(ytr)), "horizons": H, "per_horizon": {}}
agg = {}
preds, _ = N.naive_predictions(val, mu, sigma, 6)
for k, h in enumerate(H):
    c_mean = ytr[:, k, :].mean(axis=0); c_med = np.median(ytr[:, k, :], axis=0)
    row = {"intercept_train_mean_MAE": float(np.abs(yv[:, k, :] - c_mean).mean()),
           "intercept_train_median_MAE": float(np.abs(yv[:, k, :] - c_med).mean()),
           "intercept_train_mean_value": c_mean.tolist(), "intercept_train_median_value": c_med.tolist()}
    for name, t in table["per_naive"].items():
        row[name + "_MAE"] = t[str(h)]["MAE"]
    res["per_horizon"][str(h)] = row
names = ["intercept_train_mean", "intercept_train_median", "persistence_last_value", "zero_return", "train_mean"]  # seasonal_6 is NOT_AVAILABLE for h>6 (needs h<=P): no 6..36 aggregate
res["objective_equivalent_mean_over_horizons_MAE"] = {n: float(np.mean([res["per_horizon"][str(h)][n + "_MAE"] for h in H])) for n in names}
best = min(res["objective_equivalent_mean_over_horizons_MAE"].values())
res["best_naive_objective"] = best
res["skill_vs"] = {}
for mo in model_obj:
    res["skill_vs"][str(mo)] = {n: 1 - mo / v for n, v in res["objective_equivalent_mean_over_horizons_MAE"].items()}
    res["skill_vs"][str(mo)]["vs_best_naive"] = 1 - mo / best
out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
print(json.dumps({"objective_equivalent": res["objective_equivalent_mean_over_horizons_MAE"], "skill": res["skill_vs"]}, indent=1))
