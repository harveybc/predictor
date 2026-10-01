#!/usr/bin/env python3
"""Horizon-pattern diagnostic for verified corrected candidates (CPU, validation only, no retuning).

Label DIAGNOSTIC_NOT_A_RESULT. For every VERIFIED candidate of the campaign it reloads
the saved checkpoint, predicts the declared validation rows on CPU and computes per
horizon h = 1..H, on exactly the same rows:

  target_std             std of the z-scored target at origin + h
  naive_MAE              persistence (last observed value at the origin)
  seasonal_naive_MAE     value 24 h before the target time (existing seasonal control;
                         inside the 24-step input window for h <= 24)
  model_MAE              this candidate (CPU inference; GPU receipts are the reference)
  residual_mean          mean(prediction - target): bias
  prediction_std_ratio   std(prediction) / std(target): dispersion
  corr_target_origin     correlation of the target with the last observed value

Endpoint check: target_timestamps must equal origin + h x sample period for every row,
and for sampled rows/channels the stored target must equal the source CSV value at
that timestamp re-scaled with the train-only scaler (recomputed from the source).

Statement carried in the output: a positive aggregate does not justify suppressing
losing horizons; the pattern alone proves neither a bug nor financial uselessness;
any objective/horizon-weight change is a new TRAIN/validation ablation.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.checkout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.validation, allow_pickle=False) as z:
        x = z["windows"].astype(np.float32)
        y = z["targets"].astype(np.float64)
        origins = z["timestamps"].astype(np.int64)
        tts = z["target_timestamps"].astype(np.int64)
        horizons = z["horizons"].astype(int).tolist()
        names = z["feature_names"].astype(str).tolist()
    n, H, T = y.shape
    period = int(tts[0, 0] - origins[0]) // horizons[0]
    index_ok = bool(np.array_equal(tts, origins[:, None] + np.asarray(horizons)[None, :] * period))

    # source cross-check: recompute the train-only scaler and compare sampled targets
    import csv
    with open(args.source_csv) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader]
    dates = np.array([r[0] for r in rows]).astype("datetime64[s]").astype(np.int64)
    values = np.array([[float(v) for v in r[1:]] for r in rows])
    n_train = int(len(values) * 0.7)
    mean, scale = values[:n_train].mean(0), values[:n_train].std(0)
    where = {int(t): i for i, t in enumerate(dates)}
    rng = np.random.default_rng(0)
    worst = 0.0
    checks = 0
    for i in rng.choice(n, size=min(200, n), replace=False):
        for hi in (0, H - 1):
            row = where[int(tts[i, hi])]
            for c in rng.choice(T, size=5, replace=False):
                expected = (values[row, c] - mean[c]) / scale[c]
                worst = max(worst, abs(expected - y[i, hi, c]))
                checks += 1
    endpoint = {"target_timestamps_equal_origin_plus_h_x_period": index_ok, "sample_period_seconds": period,
                "source_crosscheck_values": checks, "max_abs_difference_z": worst,
                "header_matches": header[1:] == names}

    naive = np.repeat(x[:, -1:, :], H, axis=1).astype(np.float64)
    seasonal = np.stack([x[:, h - 1, :] for h in horizons], axis=1).astype(np.float64) if max(horizons) <= x.shape[1] else None
    base = []
    for k, h in enumerate(horizons):
        yt = y[:, k, :]
        base.append({"horizon": h, "target_std": float(yt.std()),
                     "naive_MAE": float(np.abs(naive[:, k] - yt).mean()),
                     "seasonal_naive_MAE": None if seasonal is None else float(np.abs(seasonal[:, k] - yt).mean()),
                     "corr_target_origin": float(np.corrcoef(yt.ravel(), x[:, -1, :].ravel())[0, 1])})

    import tensorflow as tf
    from predictor_plugins import modular_temporal  # noqa: F401
    db = sqlite3.connect(f"file:{args.queue}?mode=ro", uri=True)
    cands = db.execute("select c.cid, c.label, c.seed, a.receipt_path from candidates c join attempts a using(cid) "
                       "where c.status='verified' and a.kind='train' and a.status='completed' order by c.position").fetchall()
    models = []
    for cid, label, seed, receipt in cands:
        r = json.loads(Path(receipt).read_text())
        model = tf.keras.models.load_model(r["artifacts"]["best_model"], compile=False, safe_mode=True)
        pred = np.concatenate([np.asarray(model(x[s:s + 512], training=False)) for s in range(0, n, 512)]).astype(np.float64)
        per = []
        for k, h in enumerate(horizons):
            p, yt = pred[:, k, :], y[:, k, :]
            per.append({"horizon": h, "model_MAE_cpu": float(np.abs(p - yt).mean()),
                        "receipt_MAE_gpu": r["per_horizon"][str(h)]["MAE"],
                        "residual_mean": float((p - yt).mean()), "prediction_std_ratio": float(p.std() / yt.std())})
        models.append({"cid": cid, "label": label, "seed": seed, "per_horizon": per})
        del model
        tf.keras.backend.clear_session()

    report = {"label": "DIAGNOSTIC_NOT_A_RESULT", "population": {"rows": n, "horizons": H, "targets": T},
              "endpoint_check": endpoint, "baselines": base, "models": models,
              "statement": ["A positive aggregate does not justify suppressing losing horizons.",
                            "The pattern alone proves neither a bug nor financial uselessness.",
                            "Any objective or horizon-weight change is a new TRAIN/validation ablation, not a rewrite of this batch."]}
    (out / "HORIZON_DIAGNOSTIC.json").write_text(json.dumps(report, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = np.array(horizons)
    model_mae = np.array([[p["receipt_MAE_gpu"] for p in m["per_horizon"]] for m in models])
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.fill_between(hs, model_mae.min(0), model_mae.max(0), color="#2a78d6", alpha=0.18, linewidth=0,
                    label=f"corrected candidates, min-max over {len(models)}")
    ax.plot(hs, np.median(model_mae, 0), color="#2a78d6", linewidth=2, label="corrected candidates, median")
    ax.plot(hs, [b["naive_MAE"] for b in base], color="#eb6834", linewidth=2, marker="o", markersize=4,
            label="persistence (last value)")
    if seasonal is not None:
        ax.plot(hs, [b["seasonal_naive_MAE"] for b in base], color="#1baf7a", linewidth=2, marker="s", markersize=4,
                label="seasonal naive (24 h earlier)")
    ax.set_xlabel("horizon (hours ahead)")
    ax.set_ylabel("validation MAE, z_train")
    ax.set_title("ECL validation MAE by horizon - corrected R0 batch (DIAGNOSTIC_NOT_A_RESULT)", fontsize=10)
    ax.set_xticks(hs[::1] if H <= 24 else hs[::2])
    ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "HORIZON_DIAGNOSTIC.png")
    print(json.dumps({"endpoint_check": endpoint, "models": len(models)}))


if __name__ == "__main__":
    main()
