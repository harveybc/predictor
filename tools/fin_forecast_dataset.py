#!/usr/bin/env python3
"""Generalized financial view -> identified NPZ builder (lane F2, campaigns 2+: EURUSD/GBPUSD 1h, BTC 4h).

Same output contract as ``tools/eth_forecast_dataset.py`` (the evaluator's NPZ fields,
``MANIFEST.json``, ``SPLIT.json``) with three generalizations required by the FX
resources (irregular sampling, weekend gaps, OHLC-only inputs):

* **Derived features** declared by the frozen manifest are computed here from the raw
  columns, causally at row t: ``log_return_1 = ln(CLOSE_t/CLOSE_{t-1})``,
  ``log_high_low = ln(HIGH/LOW)``, ``close_location = (CLOSE-LOW)/(HIGH-LOW)`` (NaN when
  HIGH == LOW), ``log_close_open = ln(CLOSE/OPEN)``. ``log_return_1`` is always present as
  the TARGET SUPPORT channel (appended when the manifest does not list it; declared as such).
* **Labels located by elapsed seconds** (lane B rule): the label of origin t at horizon h is
  the row whose timestamp is exactly ``t + h * step``; origins lacking a label at ANY
  declared horizon are dropped and counted (joint drop; lane B's probe drops per horizon,
  so its populations are supersets). Inputs are the last ``window`` rows in trading time.
  Windows spanning an irregular step are excluded only when ``exclude_irregular_windows``
  (the regular ETH 4h view); for FX they are kept and counted.
* **Target** ``Y_h = sum of z(log_return_1)`` over the ROWS in (origin, label_row]: the
  plain row sum of the standardized channel (telescopes to ``(ln(C_label/C_origin) -
  n_rows*mu)/sigma``). ``label_support.npz`` beside the NPZ records ``label_rows`` and
  ``n_rows`` per origin and horizon for the exact inverse and the zero-return naive.
* **Purge** by elapsed time: a train origin is kept iff ``label_time(h_max) + purge_seconds
  < first validation input time``; a validation origin iff ``label_time(h_max) +
  purge_seconds < first test row time``. Test rows are dropped before any statistic.

One train-only StandardScaler (ddof=0) over the feature channels; metric space z_train.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.eth_forecast_dataset import sha_file, sha_text

SCHEMA = "f2.fin_forecast_npz.v1"
TARGET = "log_return_1"
DERIVED = {
    "log_return_1": lambda c: np.concatenate([[np.nan], np.log(c["CLOSE"][1:] / c["CLOSE"][:-1])]),
    "log_high_low": lambda c: np.log(c["HIGH"] / c["LOW"]),
    "close_location": lambda c: np.where(c["HIGH"] > c["LOW"], (c["CLOSE"] - c["LOW"]) / np.where(
        c["HIGH"] > c["LOW"], c["HIGH"] - c["LOW"], 1.0), np.nan),
    "log_close_open": lambda c: np.log(c["CLOSE"] / c["OPEN"]),
}


def read_csv(path, expected_sha, date_column="DATE_TIME", date_format="%Y-%m-%d %H:%M:%S"):
    actual = sha_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"source digest mismatch: {actual} != {expected_sha}")
    with open(path) as stream:
        header = stream.readline().rstrip("\n").split(",")
    if header[0] != date_column:
        raise ValueError(f"first column must be {date_column}")
    raw_dates = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    times = np.array([int(dt.datetime.strptime(d, date_format).replace(tzinfo=dt.timezone.utc).timestamp())
                      for d in raw_dates], dtype=np.int64)
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, len(header)), dtype=np.float64, ndmin=2)
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return header[1:], times, values, actual


def materialize_features(names, values, features):
    columns = {n: values[:, i] for i, n in enumerate(names)}
    out, declared = [], list(features)
    if TARGET not in declared:
        declared.append(TARGET)
    for f in declared:
        if f in columns:
            out.append(columns[f])
        elif f in DERIVED:
            out.append(DERIVED[f](columns))
        else:
            raise ValueError(f"feature {f} is neither a view column nor a declared derived transform")
    return declared, np.stack(out, axis=1)


def build(source, out, *, features, window, horizons, sample_hours, split, purge_seconds, expected_sha,
          dataset_id, exclude_irregular_windows=False, feature_manifest=None, date_column="DATE_TIME"):
    names, times, values, source_sha = read_csv(source, expected_sha, date_column)
    tr_lo, tr_hi = split["train_rows"]
    va_lo, va_hi = split["validation_rows"]
    te_lo, te_hi = split["test_rows"]
    if not (0 == tr_lo < tr_hi == va_lo < va_hi == te_lo <= te_hi == len(values)):
        raise ValueError(f"declared split {split} does not tile the view of {len(values)} rows")
    test_start_time = int(times[te_lo]) if te_lo < len(times) else None
    times, values = times[:te_lo], values[:te_lo]  # test rows dropped before any statistic
    declared, raw = materialize_features(names, values, features)
    step = int(round(sample_hours * 3600))
    hmax = max(horizons)
    finite_rows = np.all(np.isfinite(raw), axis=1)
    train_rows = finite_rows.copy()
    train_rows[tr_hi:] = False
    mean = raw[train_rows].mean(axis=0)
    scale = raw[train_rows].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError(f"constant train channels: {[f for f, s in zip(declared, scale) if s <= 0]}")
    z = (raw - mean) / scale
    t_idx = declared.index(TARGET)
    scaler = {"kind": "StandardScaler(ddof=0)", "fit_rows": [tr_lo, tr_hi], "fit_rows_finite": int(train_rows.sum()),
              "feature_count": len(declared),
              "mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
              "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
              "target_feature": TARGET, "target_mu": float(mean[t_idx]), "target_sigma": float(scale[t_idx])}
    scaler_identity = "fin-train-standard:" + sha_text(json.dumps(scaler, sort_keys=True))
    row_of = {int(t): i for i, t in enumerate(times)}
    h_arr = np.asarray(horizons, dtype=np.int64)

    def origins_in(lo, hi, boundary_time):
        kept, dropped_label, dropped_window, dropped_purge, irregular, label_rows = [], 0, 0, 0, 0, []
        for o in range(max(lo, window - 1), hi):
            labels = [row_of.get(int(times[o]) + int(h) * step) for h in horizons]
            if any(l is None for l in labels):
                dropped_label += 1
                continue
            if not finite_rows[o - window + 1:o + 1].all() or not finite_rows[o + 1:labels[-1] + 1].all():
                dropped_window += 1
                continue
            if boundary_time is not None and int(times[labels[-1]]) + purge_seconds >= boundary_time:
                dropped_purge += 1
                continue
            span = times[o - window + 1:labels[-1] + 1]
            if np.any(np.diff(span) != step):
                irregular += 1
                if exclude_irregular_windows:
                    continue
            kept.append(o)
            label_rows.append(labels)
        return (np.asarray(kept, np.int64), np.asarray(label_rows, np.int64).reshape(len(kept), len(horizons)),
                {"dropped_missing_label": dropped_label, "dropped_nonfinite_support": dropped_window,
                 "dropped_purge": dropped_purge, "irregular_windows": irregular,
                 "irregular_excluded": exclude_irregular_windows})
    val_o, val_lab, val_counts = origins_in(va_lo, va_hi, test_start_time)
    if not len(val_o):
        raise ValueError("no validation origin survives the label/purge rules")
    validation_input_time = int(times[int(val_o.min()) - window + 1])
    tr_o, tr_lab, tr_counts = origins_in(tr_lo, tr_hi, validation_input_time)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(z[:, t_idx]).astype(np.float64))])
    manifest = {"schema": SCHEMA, "dataset_id": dataset_id, "source_sha256": source_sha, "view_rows": te_hi,
                "features": len(declared), "feature_names": declared, "feature_order_sha256": sha_text(",".join(declared)),
                "target_support_channel_appended": TARGET not in features, "feature_manifest": feature_manifest,
                "window": window, "horizons": list(horizons), "sample_hours": sample_hours,
                "label_location": "elapsed_seconds (origin_time + h*step must exist as a bar; joint drop over horizons)",
                "input_window": "last `window` rows in trading time", "declared_split": split,
                "purge_seconds": purge_seconds, "scaler": scaler, "scaler_identity": scaler_identity,
                "target": {"feature": TARGET, "definition": "Y_h = sum of z(log_return_1) over rows (origin, label_row] "
                           "= (ln(CLOSE[label]/CLOSE[origin]) - n_rows*mu)/sigma; mu, sigma fitted on train rows only",
                           "inverse": "R = sigma*Y_h + n_rows*mu (n_rows from label_support.npz; = h on regular data); "
                                      "CLOSE[label] = CLOSE[origin]*exp(R)",
                           "mu": scaler["target_mu"], "sigma": scaler["target_sigma"]},
                "metric_space": "z_train", "test_rows_read_for_statistics": False, "test_windows_constructed": False,
                "splits": {}}
    offsets = np.arange(-window + 1, 1)
    support = {}
    iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for name, origins, labels, counts in (("train", tr_o, tr_lab, tr_counts), ("validation", val_o, val_lab, val_counts)):
        x = z[origins[:, None] + offsets[None, :]].astype(np.float32)
        y = (csum[labels + 1] - csum[origins[:, None] + 1]).astype(np.float32)[:, :, None]
        ts = times[origins]
        row_ids = np.array([f"{dataset_id.split(':')[0]}:row{o}:{t}" for o, t in zip(origins.tolist(), ts.tolist())])
        arrays = dict(windows=x, targets=y, row_ids=row_ids, timestamps=ts, target_timestamps=ts[:, None] + h_arr * step,
                      dataset_id=np.array(dataset_id), split=np.array(name), feature_names=np.array(declared),
                      target_names=np.array([TARGET]), horizons=h_arr, timestamp_unit=np.array("seconds"),
                      metric_space=np.array("z_train"), scaler_identity=np.array(scaler_identity),
                      scaler_scale=np.array([scaler["target_sigma"]], np.float64))
        path = out / f"{name}.npz"
        np.savez(path, **arrays)
        support[f"{name}_origins"], support[f"{name}_label_rows"] = origins, labels
        support[f"{name}_n_rows"] = labels - origins[:, None]
        manifest["splits"][name] = {
            "path": path.name, "sha256": sha_file(path), "windows": int(len(origins)),
            "origin_rows": [int(origins.min()), int(origins.max())],
            "input_rows": [int(origins.min()) - window + 1, int(origins.max())],
            "label_rows": [int(labels.min()), int(labels.max())],
            "origin_start": iso(ts.min()), "origin_end": iso(ts.max()), "origin_start_epoch": int(ts.min()),
            "origin_end_epoch": int(ts.max()), **counts,
            "row_ids_sha256": sha_text("\n".join(row_ids.tolist())), "bytes": path.stat().st_size}
    np.savez(out / "label_support.npz", **support)
    manifest["label_support_sha256"] = sha_file(out / "label_support.npz")
    split_doc = {"schema": "f2.fin_forecast_split.v1", "dataset_id": dataset_id, "source_sha256": source_sha,
                 "declared_split": split, "purge_seconds": purge_seconds, "window": window, "horizons": list(horizons),
                 "sample_hours": sample_hours, "label_location": manifest["label_location"],
                 "test": {"rows": [te_lo, te_hi], "status": "PROTECTED_NEVER_READ"},
                 "splits": {k: {kk: v[kk] for kk in v if kk not in ("path", "sha256", "bytes")}
                            for k, v in manifest["splits"].items()},
                 "scaler_identity": scaler_identity, "target": manifest["target"]}
    (out / "SPLIT.json").write_text(json.dumps(split_doc, indent=2, sort_keys=True) + "\n")
    manifest["split_sha256"] = sha_file(out / "SPLIT.json")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--feature-manifest", required=True, help="frozen selected-feature manifest JSON (lane B)")
    parser.add_argument("--dataset-id", required=True, help="e.g. eurusd1h:heuristic-strategy:939f5e6:<sha>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizons", required=True, help="'a-b' range or comma list, in steps")
    parser.add_argument("--sample-hours", type=float, required=True)
    parser.add_argument("--purge-hours", type=float, default=0.0)
    parser.add_argument("--split-variant", default=None, help="key inside the manifest's split block, if variants exist")
    parser.add_argument("--exclude-irregular-windows", action="store_true")
    args = parser.parse_args()
    text = Path(args.feature_manifest).read_text()
    fm = json.loads(text)
    features = list(fm["features"])
    block = fm["split"][args.split_variant] if args.split_variant else fm["split"]
    split = {"declared_by": block.get("declared_by", fm["split"].get("declared_by")),
             "train_rows": list(block["train"]["rows"]), "validation_rows": list(block["validation"]["rows"]),
             "test_rows": list(block["test"]["rows"])}
    if "-" in args.horizons:
        a, b = map(int, args.horizons.split("-"))
        horizons = list(range(a, b + 1))
    else:
        horizons = [int(h) for h in args.horizons.split(",")]
    feature_manifest = {"file_sha256": sha_text(text), "status": fm.get("status"), "variant": fm.get("variant"),
                        "manifest_sha256_canonical": fm.get("manifest_sha256_canonical"),
                        "valid_horizons": fm.get("valid_horizons"), "feature_count": len(features)}
    manifest = build(args.source, args.out, features=features, window=args.window, horizons=horizons,
                     sample_hours=args.sample_hours, split=split, purge_seconds=int(args.purge_hours * 3600),
                     expected_sha=args.expected_sha256, dataset_id=args.dataset_id,
                     exclude_irregular_windows=args.exclude_irregular_windows, feature_manifest=feature_manifest)
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "splits", "scaler_identity", "split_sha256")}, indent=1))


if __name__ == "__main__":
    main()
