"""Price error of trend-at-origin plus a seasonal forecast.

The seasonal table is not a price. This script only scores the
reconstruction trend(origin) + seasonal(horizon) against future CLOSE.
It does not write Prediction_h_* columns.
"""
import json
from pathlib import Path

import numpy as np

PHASE = Path("/var/tmp/retsu-phase1")
HOUR = 3600 * 10**9
series = np.load(PHASE / "stl_series.npz")
table = np.load(PHASE / "seasonal_predictions_best.npz")
avail = series["avail_ns"].astype(np.int64)
close = series["close"].astype(np.float64)
trend = series["trend"].astype(np.float64)
order = np.argsort(avail, kind="mergesort")
avail, close, trend = avail[order], close[order], trend[order]
# Last write wins if a timestamp repeats.
index = {int(ts): i for i, ts in enumerate(avail)}

meta = json.loads((PHASE / "analysis_tracked.json").read_text())
jobs = meta["table_jobs"]
report = {"unit": "CLOSE_price", "reconstruction": "trend_at_origin + seasonal_forecast", "jobs": {}}
for task, name in jobs.items():
    horizons = table[f"{name}_horizons"]
    job_rows = []
    for split in ("monitor", "holdout_2024"):
        ts = table[f"{name}_{split}_ts"].astype(np.int64)
        y = table[f"{name}_{split}_y"].astype(np.float64)
        pred = table[f"{name}_{split}_pred"].astype(np.float64)
        origin_s = table[f"{name}_{split}_origin"].astype(np.float64)
        for k, horizon in enumerate(horizons):
            label = ts + int(horizon) * HOUR
            ok = np.array([int(a) in index and int(b) in index for a, b in zip(ts, label)])
            io = np.array([index[int(a)] for a in ts[ok]])
            il = np.array([index[int(b)] for b in label[ok]])
            future = close[il]
            recon = trend[io] + pred[ok, k]
            oracle = trend[io] + y[ok, k]
            persist_price = close[io]
            persist_comp = trend[io] + origin_s[ok]
            def mae(hat):
                return float(np.mean(np.abs(hat - future)))
            job_rows.append({
                "split": split,
                "horizon": int(horizon),
                "rows": int(ok.sum()),
                "recon_mae": mae(recon),
                "oracle_trend_plus_true_seasonal_mae": mae(oracle),
                "price_persistence_mae": mae(persist_price),
                "trend_plus_seasonal_persistence_mae": mae(persist_comp),
            })
    report["jobs"][name] = job_rows
    print(name)
    for row in job_rows:
        if row["split"] != "holdout_2024":
            continue
        print(
            " h", row["horizon"],
            "recon", round(row["recon_mae"], 6),
            "oracle", round(row["oracle_trend_plus_true_seasonal_mae"], 6),
            "price_persist", round(row["price_persistence_mae"], 6),
            "comp_persist", round(row["trend_plus_seasonal_persistence_mae"], 6),
            "rows", row["rows"],
        )

(PHASE / "price_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
print("WROTE", PHASE / "price_reconstruction.json")
