#!/usr/bin/env python3
"""Build identified train/validation NPZ inputs for the lane F2 ETH 4h forecast campaign.

Source: the git-pinned model-ready view (predictor b1f8a74f,
``examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv``), content-addressed
by SHA-256 and verified before parsing. Features: the frozen variant A manifest (lane B /
M03, admissible declaration f3c0beca, 83 features in ``all_admissible_control`` order),
whose canonical digest is bound into the manifest written here.

Split: the view's DECLARED calendar split (predictor manifest 14a1077f), the authority for
the whole ETH 4h front: TRAIN rows [0, 13699) = 2017-09-28..2023-12-31, VALIDATION rows
[13699, 15895) = calendar 2024, TEST rows [15895, 18085) = 2025, protected. The test rows
are dropped immediately after parsing, before any statistic or window is formed. Within
the split a purge of ``h_max = max(horizons)`` bars is applied at each boundary: train
origins satisfy ``origin + h_max < first_validation_input_row`` (the h_max origins whose
targets would overlap the validation input support are purged), and validation origins
satisfy ``origin + h_max < first_test_row`` (the h_max origins whose targets would touch
the test year are purged).

Scaler: one StandardScaler (population std, ddof=0) fitted on the TRAIN rows only, for
every feature. Metric space ``z_train``.

Target: the cumulative close-price log-return over (origin, origin + h] for each horizon
h, expressed in train-standardized 1-bar units: ``Y_h = sum_{k=1..h} z(log_return_1[t+k])``
where ``z(r) = (r - mu) / sigma`` with mu, sigma the train mean and std of ``log_return_1``
(which the view defines as ``log(CLOSE_t / CLOSE_{t-1})``, verified here). The target
feature is therefore ``log_return_1`` (``scaler_scale = [sigma]``), and the evaluator's
persistence naive is the last observed 1-bar standardized return repeated per horizon.
Inverse transform for consumers: ``R_h = sigma * Y_h + h * mu``;
``CLOSE[t+h] = CLOSE[t] * exp(R_h)``.

Windows spanning an irregular timestamp step (not exactly ``sample_hours``) are excluded
and counted. Output: ``train.npz``, ``validation.npz`` in the evaluator's exact contract,
``MANIFEST.json`` and ``SPLIT.json`` (the shareable split artefact: row ranges, timestamps,
row-id digests). Local arithmetic only; no governance authority is granted here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

VIEW_SHA256 = "1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f"
VIEW_COMMIT = "b1f8a74f"
DECLARED_SPLIT = {"declared_by": "predictor manifest 14a1077f (view calendar split)",
                  "train_rows": [0, 13699], "validation_rows": [13699, 15895], "test_rows": [15895, 18085]}
TARGET_FEATURE = "log_return_1"
SCHEMA = "f2.eth_forecast_npz.v1"


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_view(path, expected_sha, date_column="DATE_TIME"):
    actual = sha_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"source digest mismatch: {actual} != {expected_sha}")
    with open(path) as stream:
        header = stream.readline().rstrip("\n").split(",")
    if header[0] != date_column:
        raise ValueError(f"first column must be {date_column}")
    raw_dates = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str)
    times = np.array([int(dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
                      for d in raw_dates], dtype=np.int64)
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, len(header)), dtype=np.float64, ndmin=2)
    return header[1:], times, values, actual


def windows_for(origin_lo, origin_hi, times, z, window, horizons, sample_seconds):
    """Origins in [origin_lo, origin_hi] with regular, complete input and target support."""
    hmax = max(horizons)
    origins, gaps = [], 0
    for o in range(max(origin_lo, window - 1), origin_hi + 1):
        if o + hmax >= len(times):
            break
        span = times[o - window + 1:o + hmax + 1]
        if np.any(np.diff(span) != sample_seconds):
            gaps += 1
            continue
        origins.append(o)
    return np.asarray(origins, dtype=np.int64), gaps


def cumulative_targets(z_target, origins, horizons):
    """Y[n, k] = sum_{j=1..h_k} z_target[origin_n + j] (cumulative standardized 1-bar returns)."""
    csum = np.concatenate([[0.0], np.cumsum(z_target.astype(np.float64))])
    h = np.asarray(horizons, dtype=np.int64)
    return (csum[origins[:, None] + h[None, :] + 1] - csum[origins[:, None] + 1]).astype(np.float32)


def build(source, out, *, features, window, horizons, split=DECLARED_SPLIT, expected_sha=VIEW_SHA256,
          sample_hours=4.0, feature_manifest=None, close_column="CLOSE"):
    names, times, values, source_sha = read_view(source, expected_sha)
    out = Path(out)
    hmax = max(horizons)
    seconds = int(sample_hours * 3600)
    columns = {n: i for i, n in enumerate(names)}
    missing = [f for f in features if f not in columns]
    if missing:
        raise ValueError(f"features absent from the view: {missing[:5]}")
    if TARGET_FEATURE not in features:
        raise ValueError(f"target feature {TARGET_FEATURE} must be among the declared features")
    tr_lo, tr_hi = split["train_rows"]
    va_lo, va_hi = split["validation_rows"]
    te_lo, te_hi = split["test_rows"]
    if not (0 == tr_lo < tr_hi == va_lo < va_hi == te_lo <= te_hi == len(values)):
        raise ValueError(f"declared split {split} does not tile the view of {len(values)} rows")
    # Drop the test rows before any statistic, check or window is formed.
    keep = te_lo
    times, values = times[:keep], values[:keep]
    # Verify the view's log_return_1 definition against CLOSE on the kept rows.
    close = values[:, columns[close_column]]
    lr = values[:, columns[TARGET_FEATURE]]
    implied = np.log(close[1:] / close[:-1])
    if not np.allclose(implied, lr[1:], atol=1e-6, rtol=0):
        raise ValueError("log_return_1 is not log(CLOSE_t / CLOSE_{t-1}) in the view")
    feature_idx = [columns[f] for f in features]
    raw = values[:, feature_idx]
    if not np.all(np.isfinite(raw)):
        raise ValueError("nonfinite feature values in train/validation rows")
    train_raw = raw[tr_lo:tr_hi]
    mean = train_raw.mean(axis=0)
    scale = train_raw.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError(f"constant train channels: {[f for f, s in zip(features, scale) if s <= 0]}")
    z = (raw - mean) / scale
    t_idx = features.index(TARGET_FEATURE)
    target_mu, target_sigma = float(mean[t_idx]), float(scale[t_idx])
    scaler = {"kind": "StandardScaler(ddof=0)", "fit_rows": [tr_lo, tr_hi], "feature_count": len(features),
              "mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
              "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
              "target_feature": TARGET_FEATURE, "target_mu": target_mu, "target_sigma": target_sigma}
    scaler_identity = "eth4h-train-standard:" + sha_text(json.dumps(scaler, sort_keys=True))
    dataset_id = f"git:predictor:{VIEW_COMMIT}:ethusdt_4h_tech_stat_full_model_ready.csv:{source_sha}"
    # Validation origins: inside the validation rows, targets never touch the test rows.
    val_o, vgaps = windows_for(va_lo, va_hi - 1 - hmax, times, z, window, horizons, seconds)
    validation_input_row = int(val_o.min()) - window + 1
    # Train origins: last target row strictly before the first validation input row.
    tr_o, tgaps = windows_for(tr_lo, validation_input_row - 1 - hmax, times, z, window, horizons, seconds)
    # Origins between the last train origin and the first validation origin that are neither:
    # the h_max purge (targets would overlap the validation input support) and the window-1 rows
    # that ARE the validation input support (an origin there would also put targets inside it).
    purged_train = list(range(validation_input_row - hmax, va_lo))
    purged_validation = list(range(va_hi - hmax, va_hi))
    out.mkdir(parents=True, exist_ok=False)
    names_arr = np.array(features)
    manifest = {"schema": SCHEMA, "dataset_id": dataset_id, "source_sha256": source_sha, "source_commit": VIEW_COMMIT,
                "view_rows": te_hi, "features": len(features),
                "feature_order_sha256": sha_text(",".join(features)), "feature_manifest": feature_manifest,
                "window": window, "horizons": list(horizons), "sample_hours": sample_hours,
                "declared_split": split, "purge_bars": hmax, "scaler": scaler, "scaler_identity": scaler_identity,
                "target": {"feature": TARGET_FEATURE, "definition": "Y_h = sum_{k=1..h} z(log_return_1[t+k]) = "
                           "(log(CLOSE[t+h]/CLOSE[t]) - h*mu)/sigma; mu, sigma fitted on train rows only",
                           "inverse": "R_h = sigma*Y_h + h*mu; CLOSE[t+h] = CLOSE[t]*exp(R_h)",
                           "mu": target_mu, "sigma": target_sigma},
                "metric_space": "z_train", "test_rows_read_for_statistics": False,
                "test_windows_constructed": False, "splits": {}}
    offsets = np.arange(-window + 1, 1)
    for name, origins, gaps, purged in (("train", tr_o, tgaps, purged_train), ("validation", val_o, vgaps, purged_validation)):
        x = z[origins[:, None] + offsets[None, :]].astype(np.float32)
        y = cumulative_targets(z[:, t_idx], origins, horizons)[:, :, None]
        ts = times[origins]
        row_ids = np.array([f"eth4h:row{o}:{t}" for o, t in zip(origins.tolist(), ts.tolist())])
        arrays = dict(windows=x, targets=y, row_ids=row_ids, timestamps=ts,
                      target_timestamps=ts[:, None] + np.asarray(horizons, np.int64) * seconds,
                      dataset_id=np.array(dataset_id), split=np.array(name), feature_names=names_arr,
                      target_names=np.array([TARGET_FEATURE]), horizons=np.asarray(horizons, np.int64),
                      timestamp_unit=np.array("seconds"), metric_space=np.array("z_train"),
                      scaler_identity=np.array(scaler_identity), scaler_scale=np.array([target_sigma], np.float64))
        path = out / f"{name}.npz"
        np.savez(path, **arrays)
        iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["splits"][name] = {
            "path": path.name, "sha256": sha_file(path), "windows": int(len(origins)),
            "origin_rows": [int(origins.min()), int(origins.max())],
            "input_rows": [int(origins.min()) - window + 1, int(origins.max())],
            "target_rows": [int(origins.min()) + min(horizons), int(origins.max()) + hmax],
            "origin_start": iso(ts.min()), "origin_end": iso(ts.max()),
            "origin_start_epoch": int(ts.min()), "origin_end_epoch": int(ts.max()),
            "gap_excluded_windows": gaps, "purged_origins": purged,
            "purge_components": ({"h_max_purge": [validation_input_row - hmax, validation_input_row - 1],
                                  "validation_input_support": [validation_input_row, va_lo - 1]} if name == "train"
                                 else {"h_max_purge": [va_hi - hmax, va_hi - 1]}),
            "row_ids_sha256": sha_text("\n".join(row_ids.tolist())), "bytes": path.stat().st_size}
    split_doc = {"schema": "f2.eth_forecast_split.v1", "dataset_id": dataset_id, "source_sha256": source_sha,
                 "declared_split": split, "purge_bars": hmax, "window": window, "horizons": list(horizons),
                 "sample_hours": sample_hours, "test": {"rows": [te_lo, te_hi], "status": "PROTECTED_NEVER_READ"},
                 "splits": {k: {kk: v[kk] for kk in ("windows", "origin_rows", "input_rows", "target_rows",
                                                     "origin_start", "origin_end", "row_ids_sha256",
                                                     "gap_excluded_windows", "purged_origins", "purge_components")}
                            for k, v in manifest["splits"].items()},
                 "scaler_identity": scaler_identity, "target": manifest["target"]}
    (out / "SPLIT.json").write_text(json.dumps(split_doc, indent=2, sort_keys=True) + "\n")
    manifest["split_sha256"] = sha_file(out / "SPLIT.json")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_horizons(text):
    if "-" in text:
        a, b = map(int, text.split("-"))
        return list(range(a, b + 1))
    return [int(h) for h in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--feature-manifest", required=True, help="M03 frozen selected-feature manifest JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizons", default="1-6")
    parser.add_argument("--expected-sha256", default=VIEW_SHA256)
    args = parser.parse_args()
    manifest_text = Path(args.feature_manifest).read_text()
    fm = json.loads(manifest_text)
    features = list(fm["features"])
    if len(features) != fm.get("feature_count"):
        raise ValueError("feature manifest count disagrees with its list")
    feature_manifest = {"file_sha256": sha_text(manifest_text), "status": fm.get("status"), "variant": fm.get("variant"),
                        "manifest_sha256_canonical": fm.get("manifest_sha256_canonical"),
                        "admissible_declaration_sha256": fm.get("admissible_declaration_sha256"),
                        "feature_count": len(features)}
    manifest = build(args.source, args.out, features=features, window=args.window, horizons=parse_horizons(args.horizons),
                     expected_sha=args.expected_sha256, feature_manifest=feature_manifest)
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "splits", "scaler_identity", "split_sha256")}, indent=1))


if __name__ == "__main__":
    main()
