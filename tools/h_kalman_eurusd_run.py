#!/usr/bin/env python3
"""Lane H runner: causal Kalman family on EURUSD 1h OHLC (lane B manifest 326aee0c, git-pinned file). DEVELOPMENT, CPU.

Split: TRAIN rows [0, 65158), VALIDATION [65158, 79121), TEST [79121, 93084) protected and dropped at load
(lane B ruling: 70/15/15 chronological blocks, 144 h purge). Targets are cumulative log returns over h = 1..6 hours
LOCATED BY ELAPSED SECONDS (a window with a missing hour is excluded and counted), in train-standardized 1-bar units
(same definition as the lane F2 ETH target). Features: the four price levels, train-standardized.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eth_run = _load("h_kalman_eth_run")
campaign = _load("h_kalman_campaign")
pipe = campaign.pipe
kf = pipe.kf
arms_lib = pipe.arms_lib

CSV_SHA256 = "72b8271d2a6ab7fc5de8c64dc626f128dbb11a1127c333e4039edecbdcc41783"
DECLARED_SPLIT = {"declared_by": "lane B / coordinator ruling Q13: 70/15/15 chronological row blocks, 144 h purge",
                  "train_rows": [0, 65158], "validation_rows": [65158, 79121], "test_rows": [79121, 93084]}
DATASET_ID = "git:heuristic-strategy:939f5e6:tests/data/eurusd_hour_2005_2020.csv:" + CSV_SHA256


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_eurusd(path, split=None, expected_sha=CSV_SHA256, window=24, horizons=(1, 2, 3, 4, 5, 6), purge=144):
    split = split or DECLARED_SPLIT
    horizons = list(horizons)
    sha = _sha_file(path)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(f"source digest mismatch: {sha} != {expected_sha}")
    tr_lo, tr_hi = split["train_rows"]
    va_lo, va_hi = split["validation_rows"]
    te_lo = split["test_rows"][0]
    if not (tr_lo == 0 and tr_hi == va_lo and va_hi == te_lo):
        raise ValueError("declared split does not tile the rows")
    times, vals = [], []
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        if header != ["DATE_TIME", "OPEN", "LOW", "HIGH", "CLOSE"]:
            raise ValueError("unexpected header")
        for i, line in enumerate(rd):
            if i >= te_lo:                                  # protected test rows are never parsed
                break
            times.append(int(dt.datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()))
            vals.append([float(x) for x in line[1:5]])
    times = np.array(times, dtype=np.int64)
    raw = np.array(vals, dtype=np.float64)
    if len(times) != te_lo or (np.diff(times) <= 0).any():
        raise ValueError("timestamps must strictly increase and cover the declared rows")
    close = raw[:, 3]
    lr = np.log(close[1:] / close[:-1])
    mu, sigma = float(lr[:tr_hi - 1].mean()), float(lr[:tr_hi - 1].std())
    zt = np.zeros(len(close))
    zt[1:] = (lr - mu) / sigma
    mean, scale = raw[tr_lo:tr_hi].mean(axis=0), raw[tr_lo:tr_hi].std(axis=0)
    Z = (raw - mean) / scale
    hmax = max(horizons)
    irregular = np.concatenate([[0], np.cumsum((np.diff(times) != 3600).astype(np.int64))])   # irregular steps before row i

    def regular(o):             # steps o-window+1 .. o+hmax-1 are all 3600 s
        a, b = o - window + 1, o + hmax
        return irregular[b] - irregular[a] == 0

    def origins_in(lo, hi):
        ok, gaps = [], 0
        for o in range(max(lo, window - 1), hi + 1):
            if o + hmax >= len(times):
                break
            if regular(o):
                ok.append(o)
            else:
                gaps += 1
        return np.array(ok, dtype=np.int64), gaps

    val_o, vgaps = origins_in(va_lo, va_hi - 1 - purge)
    first_val_input = int(val_o.min()) - window + 1
    tr_o, tgaps = origins_in(tr_lo, first_val_input - 1 - purge)
    def targets(o):
        cols = []
        for h in horizons:
            cols.append((np.log(close[o + h] / close[o]) - h * mu) / sigma)
        return np.stack(cols, axis=1)
    split_doc = {"schema": "h.eurusd_split.v1", "dataset_id": DATASET_ID, "split": split, "window": window,
                 "horizons": horizons, "purge_rows": purge, "train_origins": [int(tr_o.min()), int(tr_o.max())],
                 "validation_origins": [int(val_o.min()), int(val_o.max())], "n_train": int(len(tr_o)), "n_val": int(len(val_o)),
                 "test": "PROTECTED_NEVER_READ"}
    return {"Z": Z, "names": ["OPEN", "LOW", "HIGH", "CLOSE"], "train_rows": [tr_lo, tr_hi], "val_rows": [va_lo, va_hi],
            "origins": {"train": tr_o, "validation": val_o}, "Y": {"train": targets(tr_o), "validation": targets(val_o)},
            "target_series": zt, "mu": mu, "sigma": sigma, "horizons": horizons, "window": window, "seasonal_period": 6,
            "dataset_id": DATASET_ID, "view_sha256": sha,
            "split_sha256": hashlib.sha256(json.dumps(split_doc, sort_keys=True).encode()).hexdigest(),
            "split_doc": split_doc, "gap_excluded": {"train": int(tgaps), "validation": int(vgaps)},
            "row_ids_sha256": {s: hashlib.sha256("\n".join(f"eurusd1h:row{o}:{times[o]}" for o in oo.tolist()).encode()).hexdigest()
                               for s, oo in (("train", tr_o), ("validation", val_o))}}


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = eth_run.Heartbeat(out, args.name, interval=args.heartbeat)
    hb.start()
    t_start, cpu0 = time.time(), time.process_time()
    hb.stage("load")
    d = load_eurusd(args.csv)
    cands = json.loads(Path(args.candidates).read_text())
    cand_sha = hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest()
    ds = [i for i, x in enumerate(cands["datasets"]) if x["manifest_canonical"].startswith("326aee0c")][0]
    groups = eth_run.groups_from_candidates(cands, ds, d["names"], exclude_prefixes=())
    nv = pipe._naive_block(d)
    naive_summary = {n: [float(np.mean(np.abs(nv[n][:, k] - d["Y"]["validation"][:, k]))) for k in range(6)] for n in nv}
    results = {"schema": "lane_h_kalman_eurusd_results.v1", "label": "DEVELOPMENT", "role": args.role, "groups": groups,
               "inputs": {"csv_sha256": d["view_sha256"], "candidates_file_sha256": cand_sha, "split_sha256": d["split_sha256"],
                          "dataset_id": d["dataset_id"], "row_ids_sha256": d["row_ids_sha256"]},
               "split": d["split_doc"], "gap_excluded_windows": d["gap_excluded"], "naive_validation_MAE_by_horizon": naive_summary,
               "rows": {"train_origins": int(len(d["origins"]["train"])), "validation_origins": int(len(d["origins"]["validation"])),
                        "test_used": False},
               "environment": kf.environment_record(args.role), "variants": {}}
    variants_res, replay = campaign.run_variants(d, groups, args.variants, args.heavy_variant, False, hb, args.role, 74,
                                                 {"csv_sha256": d["view_sha256"], "split_sha256": d["split_sha256"],
                                                  "row_ids_sha256": d["row_ids_sha256"]})
    results["variants"] = variants_res
    results["totals"] = {"wall_s": time.time() - t_start, "cpu_s": time.process_time() - cpu0,
                         "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    replay["replay_digest_exact_core"] = eth_run._sha_json({"inputs": replay["inputs"], "kalman": replay["kalman"],
                                                             "arms_exact_inputs": replay["arms_exact_inputs"]})
    replay["replay_digest_numeric"] = eth_run._sha_json(replay["arms_numeric_predictions"])
    (out / "REPLAY_DIGESTS.json").write_text(json.dumps(replay, indent=1, sort_keys=True) + "\n")
    (out / "RESULTS.json").write_text(json.dumps(pipe.public(results), indent=1, sort_keys=True, allow_nan=False) + "\n")
    hb.stage("done")
    hb.stop()
    return results, replay


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--name", default="h1-eurusd-arms")
    ap.add_argument("--variants", nargs="+", default=["moments_train", "declared_1e-3", "declared_1e-2", "declared_1e-1"])
    ap.add_argument("--heavy-variant", default="moments_train")
    ap.add_argument("--heartbeat", type=float, default=60.0)
    args = ap.parse_args()
    results, replay = run(args)
    print(json.dumps({"replay_digest_exact_core": replay["replay_digest_exact_core"],
                      "replay_digest_numeric": replay["replay_digest_numeric"], "totals": results["totals"]}))


if __name__ == "__main__":
    main()
