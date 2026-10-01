#!/usr/bin/env python3
"""Same-row naive controls, per-row prediction export and the strict evidence record (lane F2).

Naives on the validation rows of the lane F2 NPZ (target = cumulative standardized 1-bar
close log-return, see ``tools/eth_forecast_dataset.py``), per horizon h:

* ``persistence_last_value``: the evaluator's baseline, the last observed standardized 1-bar
  return at the origin repeated for every horizon (``windows[:, -1, target]``);
* ``zero_return``: MANDATORY first bar. CLOSE[t+h] = CLOSE[t], i.e. ``Y_h = -h*mu/sigma``
  (mu, sigma from the data manifest, bound to the NPZ scaler identity);
* ``train_mean``: ``Y_h = 0`` (the train drift ``h*mu`` in return units);
* ``seasonal_<P>``: ``Y_h(t) := Y_h(t-P)``, the cumulative target one period earlier, read
  from the input window (requires h <= P and P + h - 1 < window);
* ``strict_minimum``: per horizon the lowest-MAE naive among the above.

``strict_record`` takes a ``predictor.forecast_naive_evidence.v1`` record built by M04's
``tools/modular_forecast_evidence.py`` and rewrites ``per_horizon[].naive_*`` to the strict
minimum, adds ``zero_return_naive``, ``train_mean_naive``, ``seasonal_naive`` blocks per
horizon and a top-level ``naives`` block, then re-signs ``evidence_sha256`` with the same
canonical form (``verify`` of the M04 tool still holds).

``predictions_csv`` writes one row per validation origin (row id, DATE_TIME, z forecasts,
log-return forecasts, price forecasts from the view's CLOSE at the origin) from the float32
prediction bytes whose sha256 the independent scorer reproduced.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from tools import modular_forecast_evidence as mfe


def _load_validation(path):
    raw = Path(path).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if str(data["split"]) != "validation":
        raise ValueError("naives are computed on the validation split only")
    return data, hashlib.sha256(raw).hexdigest()


def naive_predictions(data, mu, sigma, seasonal_period):
    """Dict name -> array [N, H, T] of naive forecasts on the NPZ rows (same shape as targets)."""
    x, y = data["windows"].astype(np.float64), data["targets"].astype(np.float64)
    names = data["feature_names"].astype(str).tolist()
    idx = [names.index(t) for t in data["target_names"].astype(str).tolist()]
    horizons = data["horizons"].tolist()
    n, w, _ = x.shape
    out = {"persistence_last_value": np.repeat(x[:, -1:, idx], len(horizons), axis=1)}
    out["zero_return"] = np.broadcast_to(np.asarray([[-h * mu / sigma] for h in horizons]), y.shape).copy()
    out["train_mean"] = np.zeros_like(y)
    seasonal = np.full_like(y, np.nan)
    for k, h in enumerate(horizons):
        if h <= seasonal_period <= w:
            # rows t-P+1 .. t-P+h  <->  window positions (w-1)-P+1 .. (w-1)-P+h
            lo, hi = (w - 1) - seasonal_period + 1, (w - 1) - seasonal_period + h
            seasonal[:, k, :] = x[:, lo:hi + 1, :][:, :, idx].sum(axis=1)
    out[f"seasonal_{seasonal_period}"] = seasonal
    return out, horizons


def naive_table(data, mu, sigma, seasonal_period=6):
    preds, horizons = naive_predictions(data, mu, sigma, seasonal_period)
    y = data["targets"].astype(np.float64)
    table = {"rows": int(len(y)), "horizons": horizons, "per_naive": {}, "strict_minimum": {}}
    for name, p in preds.items():
        rows = {}
        for k, h in enumerate(horizons):
            e = p[:, k, :] - y[:, k, :]
            if np.all(np.isfinite(e)):
                rows[str(h)] = {"MAE": float(np.abs(e).mean()), "MSE": float((e ** 2).mean())}
            else:
                rows[str(h)] = {"MAE": None, "MSE": None, "status": "NOT_AVAILABLE",
                                "reason": "target time minus one period is not inside the input window"}
        table["per_naive"][name] = rows
    for h in horizons:
        candidates = {name: rows[str(h)]["MAE"] for name, rows in table["per_naive"].items()
                      if rows[str(h)]["MAE"] is not None}
        best = min(candidates, key=candidates.get)
        table["strict_minimum"][str(h)] = {"naive": best, "MAE": candidates[best],
                                           "MSE": table["per_naive"][best][str(h)]["MSE"]}
    return table


def strict_record(record, table, *, seasonal_period):
    out = json.loads(json.dumps(record))
    out.pop("evidence_sha256", None)
    out["naive"] = {"definition": "STRICT MINIMUM per horizon: the lowest-MAE naive among persistence (last "
                                  "observed standardized 1-bar return repeated), zero-return (CLOSE[t+h]=CLOSE[t]), "
                                  "train-mean (Y_h=0) and the seasonal naive; same rows, targets, scaler, reduction",
                    "components": sorted(table["per_naive"]), "mandatory_first_bar": "zero_return"}
    out["naives"] = {"per_naive": table["per_naive"], "strict_minimum": table["strict_minimum"],
                     "rows": table["rows"]}
    for entry in out["per_horizon"]:
        h = str(entry["horizon"])
        strict = table["strict_minimum"][h]
        if table["rows"] != entry["rows"]:
            raise ValueError("naive table rows differ from the record rows")
        entry["naive_MAE"], entry["naive_MSE"], entry["strict_naive"] = strict["MAE"], strict["MSE"], strict["naive"]
        entry["MAE"] = mfe._pair(entry["model_MAE"], strict["MAE"])
        entry["MSE"] = mfe._pair(entry["model_MSE"], strict["MSE"])
        for name, key in (("zero_return", "zero_return_naive"), ("train_mean", "train_mean_naive"),
                          (f"seasonal_{seasonal_period}", "seasonal_naive")):
            row = table["per_naive"][name][h]
            if row["MAE"] is None:
                entry[key] = {"status": "NOT_AVAILABLE", "reason": row.get("reason")}
            else:
                entry[key] = {"naive_MAE": row["MAE"], "naive_MSE": row["MSE"],
                              "MAE": mfe._pair(entry["model_MAE"], row["MAE"]),
                              "MSE": mfe._pair(entry["model_MSE"], row["MSE"])}
        entry["beats_zero_return"] = (entry["zero_return_naive"].get("MAE", {}).get("delta") is not None
                                      and entry["zero_return_naive"]["MAE"]["delta"] < 0)
    out["seasonal_naive"] = {"period_steps": seasonal_period, "declared": True,
                             "definition": f"Y_h(t) := Y_h(t-{seasonal_period}) on the cumulative target, read from "
                                           "the input window; same rows, targets, scaler and reduction"}
    out["evidence_sha256"] = hashlib.sha256(mfe.canonical(out).encode()).hexdigest()
    return out


def predictions_csv(prediction, data, mu, sigma, view_csv, out_path, *, expected_sha256=None):
    """One row per validation origin; price forecasts use the view's CLOSE at the origin row."""
    pred = np.ascontiguousarray(np.asarray(prediction, dtype="<f4"))
    sha = hashlib.sha256(pred.tobytes()).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"prediction bytes {sha} differ from the verified digest {expected_sha256}")
    row_ids = data["row_ids"].astype(str).tolist()
    horizons = data["horizons"].tolist()
    rows = [int(r.split(":")[1][3:]) for r in row_ids]
    with open(view_csv) as stream:
        reader = csv.reader(stream)
        header = next(reader)
        ci, di = header.index("CLOSE"), header.index("DATE_TIME")
        view = {i: (line[di], float(line[ci])) for i, line in enumerate(reader)}
    with open(out_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "DATE_TIME", "close_origin", *[f"y_hat_z_h{h}" for h in horizons],
                         *[f"logret_hat_h{h}" for h in horizons], *[f"close_hat_h{h}" for h in horizons]])
        for n, (rid, row) in enumerate(zip(row_ids, rows)):
            date, close = view[row]
            z = [float(pred[n, k, 0]) for k in range(len(horizons))]
            lr = [sigma * v + h * mu for v, h in zip(z, horizons)]
            writer.writerow([rid, date, f"{close:.8f}", *[f"{v:.8f}" for v in z], *[f"{v:.10f}" for v in lr],
                             *[f"{close * math.exp(v):.8f}" for v in lr]])
    return {"path": str(out_path), "sha256": mfe.hashlib.sha256(Path(out_path).read_bytes()).hexdigest(),
            "predictions_sha256": sha, "rows": len(row_ids),
            "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
            "inverse": "logret = sigma*y_hat_z + h*mu; close_hat = close_origin*exp(logret)",
            "timing": "origin row t uses rows t-23..t through the CLOSE of row t"}


def _manifest_scaler(manifest_path, data):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["scaler_identity"] != str(data["scaler_identity"]):
        raise ValueError("manifest scaler identity differs from the NPZ")
    return manifest["target"]["mu"], manifest["target"]["sigma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("table")
    p.add_argument("--validation", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seasonal-period", type=int, default=6)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evidence", help="strict record from a receipt (M04 builder + strict naives)")
    p.add_argument("--receipt", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--campaign-id")
    p.add_argument("--asset", default="ETHUSDT 4h spot (trading asset; DEVELOPMENT validation 2024)")
    p.add_argument("--seasonal-period", type=int, default=6)
    p.add_argument("--out", required=True)
    p = sub.add_parser("csv")
    p.add_argument("--predictions", required=True, help="float32 .npy of the verified validation predictions")
    p.add_argument("--validation", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--expected-sha256")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    data, _ = _load_validation(args.validation)
    mu, sigma = _manifest_scaler(args.manifest, data)
    if args.command == "table":
        table = naive_table(data, mu, sigma, args.seasonal_period)
        Path(args.out).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
        print(json.dumps(table["strict_minimum"]))
    elif args.command == "evidence":
        record = mfe.build(json.loads(Path(args.receipt).read_text()), args.validation,
                           campaign_id=args.campaign_id, asset=args.asset)
        table = naive_table(data, mu, sigma, args.seasonal_period)
        strict = strict_record(record, table, seasonal_period=args.seasonal_period)
        Path(args.out).write_text(json.dumps(strict, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"evidence_sha256": strict["evidence_sha256"],
                          "beats_zero_return": {e["horizon"]: e["beats_zero_return"] for e in strict["per_horizon"]}}))
    else:
        info = predictions_csv(np.load(args.predictions), data, mu, sigma, args.view, args.out,
                               expected_sha256=args.expected_sha256)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
