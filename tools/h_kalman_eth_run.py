#!/usr/bin/env python3
"""Lane H runner: causal Kalman family on the ETH 4h variant-A panel (lane F2 split). DEVELOPMENT, CPU only.

Run through crispdm-run on a worker; threads pinned to 1. Writes RESULTS.json, RESULTS.csv, REPLAY_DIGESTS.json,
heartbeat.json (every <= 2 minutes) under --out. The protected test rows are dropped by the lane F2 builder
before any statistic and never enter this process.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import resource
import sys
import threading
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


eth = _load("eth_forecast_dataset")
pipe = _load("h_kalman_pipeline")
kf = pipe.kf
arms_lib = pipe.arms_lib

PINNED = {"view_sha256": eth.VIEW_SHA256, "feature_manifest_file_sha256_prefix": "fdff0c85",
          "split_file_sha256_prefix": "116a5b64"}


class Heartbeat:
    def __init__(self, out, name, interval=60.0):
        self.path = Path(out) / "heartbeat.json"
        self.name, self.interval = name, interval
        self._stage, self._tick, self._t0 = "init", 0, time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _write(self):
        self._tick += 1
        doc = {"name": self.name, "pid": os.getpid(), "stage": self._stage, "tick": self._tick, "time": time.time(),
               "elapsed_s": round(time.time() - self._t0, 3),
               "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "cpu_seconds": round(time.process_time(), 3)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        os.replace(tmp, self.path)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._write()

    def start(self):
        self._write()
        self._thread.start()

    def stage(self, s):
        self._stage = s
        self._write()

    def stop(self):
        self._stop.set()
        self._write()


def load_data(view, features, build_dir, window=24, horizons=(1, 2, 3, 4, 5, 6), split=None, expected_sha=eth.VIEW_SHA256,
              sample_hours=4.0, feature_manifest=None):
    split = split or eth.DECLARED_SPLIT
    horizons = list(horizons)
    build_dir = Path(build_dir)
    names, times, values, source_sha = eth.read_view(view, expected_sha)           # raises on a digest mismatch
    if not build_dir.exists():
        eth.build(view, build_dir, features=list(features), window=window, horizons=horizons, split=split,
                  expected_sha=expected_sha, sample_hours=sample_hours, feature_manifest=feature_manifest)
    manifest = json.loads((build_dir / "MANIFEST.json").read_text())
    tr_lo, tr_hi = split["train_rows"]
    va_lo, va_hi = split["validation_rows"]
    te_lo = split["test_rows"][0]
    col = {n: i for i, n in enumerate(names)}
    raw = values[:te_lo][:, [col[f] for f in features]]                          # test rows dropped before any statistic
    mean, scale = raw[tr_lo:tr_hi].mean(axis=0), raw[tr_lo:tr_hi].std(axis=0)
    z = (raw - mean) / scale
    t_idx = list(features).index(eth.TARGET_FEATURE)
    data = {"Z": z, "names": list(features), "train_rows": [tr_lo, tr_hi], "val_rows": [va_lo, va_hi],
            "origins": {}, "Y": {}, "target_series": z[:, t_idx], "mu": manifest["target"]["mu"],
            "sigma": manifest["target"]["sigma"], "horizons": horizons, "window": window, "seasonal_period": 6,
            "dataset_id": manifest["dataset_id"], "split_sha256": eth.sha_file(build_dir / "SPLIT.json"),
            "view_sha256": source_sha, "scaler_identity": manifest["scaler_identity"],
            "npz_sha256": {s: manifest["splits"][s]["sha256"] for s in ("train", "validation")},
            "row_ids_sha256": {s: manifest["splits"][s]["row_ids_sha256"] for s in ("train", "validation")}}
    for s in ("train", "validation"):
        with np.load(build_dir / f"{s}.npz", allow_pickle=False) as npz:
            rows = np.array([int(r.split(":")[1][3:]) for r in npz["row_ids"].astype(str).tolist()], dtype=np.int64)
            data["origins"][s] = rows
            data["Y"][s] = npz["targets"][:, :, 0].astype(np.float64)
            w = npz["windows"]
        got = np.stack([z[o - window + 1:o + 1] for o in rows]).astype(np.float32)
        if not np.array_equal(got, w):
            raise ValueError(f"reconstructed {s} windows differ from the lane F2 NPZ")
    return data


def groups_from_candidates(cands, dataset_index, feature_names, exclude_prefixes=("ema_cross",)):
    feats = cands["datasets"][dataset_index]["features"]
    ok = set(feature_names)
    def keep(f):
        return f["feature"] in ok and not any(f["feature"].startswith(p) for p in exclude_prefixes)
    return {"local_level": [f["feature"] for f in feats if f["form"] == "LOCAL_LEVEL" and keep(f)],
            "local_linear_trend": [f["feature"] for f in feats if f["form"] == "LEVEL_PLUS_SLOPE" and keep(f)]}


def _sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(out, args.name, interval=args.heartbeat)
    hb.start()
    t_start = time.time()
    cpu0 = time.process_time()
    hb.stage("load")
    fm = json.loads(Path(args.feature_manifest).read_text())
    features = list(fm["features"])
    fm_sha = hashlib.sha256(Path(args.feature_manifest).read_bytes()).hexdigest()
    d = load_data(args.view, features, out / "data", window=24, horizons=(1, 2, 3, 4, 5, 6),
                  expected_sha=eth.VIEW_SHA256, feature_manifest={"file_sha256": fm_sha})
    cands = json.loads(Path(args.candidates).read_text())
    cand_sha = hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest()
    ds_index = [i for i, x in enumerate(cands["datasets"]) if x["manifest_canonical"].startswith("30b78078")][0]
    groups = groups_from_candidates(cands, ds_index, features)
    pinned_ok = {"feature_manifest_sha_ok": fm_sha.startswith(PINNED["feature_manifest_file_sha256_prefix"]),
                 "split_sha_equals_pinned_prefix": d["split_sha256"].startswith(PINNED["split_file_sha256_prefix"]),
                 "view_sha_ok": d["view_sha256"] == PINNED["view_sha256"]}
    # reproduce the lane F2 same-row zero-return table on the validation rows
    nv = pipe._naive_block(d)
    pinned_naive = json.loads(Path(args.naive_table).read_text())["per_naive"]
    chk = {}
    for name in ("zero_return", "persistence_last_value", "train_mean", "seasonal_6"):
        mine = [float(np.mean(np.abs(nv[name][:, k] - d["Y"]["validation"][:, k]))) for k in range(6)]
        theirs = [pinned_naive[name][str(h)]["MAE"] for h in range(1, 7)]
        chk[name] = {"max_abs_diff_vs_lane_F2_table": max(abs(a - b) for a, b in zip(mine, theirs)), "mine": mine}
    results = {"schema": "lane_h_kalman_eth_results.v1", "label": "DEVELOPMENT", "role": args.role,
               "pinned": pinned_ok, "naive_reproduction": chk, "groups": groups, "inputs": {
                   "view_sha256": d["view_sha256"], "feature_manifest_file_sha256": fm_sha, "candidates_file_sha256": cand_sha,
                   "split_file_sha256": d["split_sha256"], "dataset_id": d["dataset_id"], "npz_sha256": d["npz_sha256"],
                   "row_ids_sha256": d["row_ids_sha256"], "scaler_identity": d["scaler_identity"]},
               "rows": {"train_origins": int(len(d["origins"]["train"])), "validation_origins": int(len(d["origins"]["validation"])),
                        "train_rows": d["train_rows"], "validation_rows": d["val_rows"], "test_used": False},
               "environment": kf.environment_record(args.role), "variants": {}}
    replay = {"inputs": {k: results["inputs"][k] for k in ("view_sha256", "split_file_sha256", "npz_sha256")},
              "kalman": {}, "arms_exact_inputs": {}, "arms_numeric_predictions": {}}
    L = 74
    for vname in args.variants:
        hb.stage(f"variant:{vname}:kalman")
        variant = pipe.VARIANTS[vname]
        t0, c0 = time.time(), time.process_time()
        kal = pipe.build_kalman(d, groups, variant, host_role=args.role)
        kal_cost = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        diag = pipe.kalman_diagnostics(d, kal)
        cost = {g: kf.measure_cost(k["artifact"], d["Z"][:d["val_rows"][1]][:, k["idx"]], repeats=1) for g, k in kal.items()}
        replay["kalman"][vname] = {g: {"artifact_sha256": v["artifact_sha256"], "fitted_state_digest": v["fitted_state_digest"],
                                       "output_digest": v["output_digest"]} for g, v in diag.items()}
        vres = {"kalman_fit_and_transform_cost": kal_cost, "kalman_cost": cost, "diagnostics": diag, "arms": {}}
        for lags in ([1, 24] if (vname == args.heavy_variant and not args.skip_heavy) else [1]):
            hb.stage(f"variant:{vname}:arms:lags{lags}")
            arms = pipe.arm_matrices(d, kal, lags=lags, controls=True)
            use = ["A", "B", "C", "IDENTITY", "B_PERMUTED", "B_NOISE", "C_EWMA", "C_SMOOTHER_NONCAUSAL"] if lags == 1 \
                else ["A", "B", "C", "C_EWMA"]
            evs = {}
            for a in use:
                hb.stage(f"variant:{vname}:lags{lags}:arm:{a}")
                t0, c0 = time.time(), time.process_time()
                ev = pipe.evaluate_arm(d, arms[a])
                ev["cost"] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
                ev["input_matrix_validation_sha256"] = arms_lib.sha_array(arms[a]["validation"])
                ev["input_matrix_train_sha256"] = arms_lib.sha_array(arms[a]["train"])
                evs[a] = ev
                replay["arms_exact_inputs"][f"{vname}|lags{lags}|{a}"] = {
                    "train": ev["input_matrix_train_sha256"], "validation": ev["input_matrix_validation_sha256"]}
                replay["arms_numeric_predictions"][f"{vname}|lags{lags}|{a}"] = ev["prediction_sha256"]
            if lags == 1:
                assert evs["IDENTITY"]["input_matrix_validation_sha256"] == evs["A"]["input_matrix_validation_sha256"]
            paired = {a: pipe.paired_against(d, evs["A"], evs[a], L) for a in use if a not in ("A", "IDENTITY")}
            vres["arms"][f"lags{lags}"] = {"evaluations": pipe.public(evs), "paired_vs_A": paired}
            for a in use:
                evs[a].pop("_pred", None)
        results["variants"][vname] = vres
    hb.stage("write")
    results["totals"] = {"wall_s": time.time() - t_start, "cpu_s": time.process_time() - cpu0,
                         "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    replay["replay_digest_exact_core"] = _sha_json({"inputs": replay["inputs"], "kalman": replay["kalman"],
                                                     "arms_exact_inputs": replay["arms_exact_inputs"]})
    replay["replay_digest_numeric"] = _sha_json(replay["arms_numeric_predictions"])
    (out / "REPLAY_DIGESTS.json").write_text(json.dumps(replay, indent=1, sort_keys=True) + "\n")
    (out / "RESULTS.json").write_text(json.dumps(pipe.public(results), indent=1, sort_keys=True, allow_nan=False) + "\n")
    with open(out / "RESULTS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "lags", "arm", "label", "eligible", "horizon", "rows", "model_MAE", "model_MSE",
                    "zero_return_MAE", "zero_return_MSE", "persistence_MAE", "persistence_MSE", "seasonal6_MAE", "seasonal6_MSE",
                    "train_mean_MAE", "strict_naive", "skill_vs_zero_return_MAE", "skill_vs_zero_return_MSE", "alpha", "channels"])
        for vname, v in results["variants"].items():
            for lg, blk in v["arms"].items():
                for arm, ev in blk["evaluations"].items():
                    for r in ev["per_horizon"]:
                        n = r["naive"]
                        w.writerow([vname, lg, arm, ev["label"], ev["eligible"], r["horizon"], r["rows"], r["model_MAE"], r["model_MSE"],
                                    n["zero_return"]["MAE"], n["zero_return"]["MSE"], n["persistence_last_value"]["MAE"],
                                    n["persistence_last_value"]["MSE"], n["seasonal_6"]["MAE"], n["seasonal_6"]["MSE"],
                                    n["train_mean"]["MAE"], r["strict_naive"], n["zero_return"]["skill_MAE"],
                                    n["zero_return"]["skill_MSE"], ev["alpha"], ev["channels"]])
    hb.stage("done")
    hb.stop()
    return results, replay


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--naive-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True, help="host role label only, e.g. worker_a / worker_b")
    ap.add_argument("--name", default="h1-eth-arms")
    ap.add_argument("--variants", nargs="+", default=["moments_train", "declared_1e-3", "declared_1e-2", "declared_1e-1"])
    ap.add_argument("--heavy-variant", default="moments_train")
    ap.add_argument("--skip-heavy", action="store_true")
    ap.add_argument("--heartbeat", type=float, default=60.0)
    args = ap.parse_args()
    results, replay = run(args)
    print(json.dumps({"replay_digest_exact_core": replay["replay_digest_exact_core"],
                      "replay_digest_numeric": replay["replay_digest_numeric"], "totals": results["totals"]}))


if __name__ == "__main__":
    main()
