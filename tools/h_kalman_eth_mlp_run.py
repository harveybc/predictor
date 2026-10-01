#!/usr/bin/env python3
"""Lane H: three arms with the flatten+MLP control learner on the ETH 4h panel, three seeds. DEVELOPMENT, CPU only."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import importlib.util
import json
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
pipe = eth_run.pipe
mlp = _load("h_kalman_mlp")
arms_lib = pipe.arms_lib


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = eth_run.Heartbeat(out, args.name, interval=args.heartbeat)
    hb.start()
    t0, c0 = time.time(), time.process_time()
    fm = json.loads(Path(args.feature_manifest).read_text())
    features = list(fm["features"])
    d = eth_run.load_data(args.view, features, out / "data", expected_sha=eth_run.eth.VIEW_SHA256,
                          feature_manifest={"file_sha256": hashlib.sha256(Path(args.feature_manifest).read_bytes()).hexdigest()})
    cands = json.loads(Path(args.candidates).read_text())
    ds = [i for i, x in enumerate(cands["datasets"]) if x["manifest_canonical"].startswith("30b78078")][0]
    groups = eth_run.groups_from_candidates(cands, ds, features)
    kal = pipe.build_kalman(d, groups, pipe.VARIANTS[args.variant], host_role=args.role)
    use = ["A", "B", "C", "B_PERMUTED", "B_NOISE", "C_EWMA"]
    res = {"schema": "lane_h_kalman_eth_mlp_results.v1", "label": "DEVELOPMENT", "role": args.role, "variant": args.variant,
           "lags": args.lags, "seeds": args.seeds, "hidden": args.hidden, "groups": groups, "arms": {}, "paired_vs_A": {},
           "environment": eth_run.kf.environment_record(args.role)}
    preds = {}
    for a in use:
        arm = pipe.arm_matrices(d, kal, lags=args.lags, controls=True, only={a})[a]     # one wide matrix in memory at a time
        res["arms"][a] = {"channels": int(arm["train"].shape[1]), "seeds": {}, "label": arm.get("label")}
        for seed in args.seeds:
            hb.stage(f"arm:{a}:seed:{seed}")
            t1, c1 = time.time(), time.process_time()
            o = mlp.mlp_fit_predict(arm["train"], d["Y"]["train"], [arm["validation"]], seed=seed,
                                    hidden=tuple(args.hidden), max_epochs=args.max_epochs, patience=args.patience)
            pred = o["predictions"][0]
            preds[(a, seed)] = pred
            rows = pipe.score_predictions(d, pred)
            res["arms"][a]["seeds"][str(seed)] = {
                "per_horizon": rows, "mean_model_MAE": float(np.mean([r["model_MAE"] for r in rows])),
                "selected_epoch": o["selected_epoch"], "epochs_run": o["epochs_run"], "weights_sha256": o["weights_sha256"],
                "prediction_sha256": arms_lib.sha_array(pred), "trainable_parameters": o["trainable_parameters"],
                "cost": {"wall_s": time.time() - t1, "cpu_s": time.process_time() - c1}}
        del arm
    for a in use:
        if a == "A":
            continue
        per_seed = {}
        for seed in args.seeds:
            base, arm = {"_pred": preds[("A", seed)]}, {"_pred": preds[(a, seed)]}
            per_seed[str(seed)] = pipe.paired_against(d, base, arm, L=74, B=500, seed=seed)
        # seed spread of the per-horizon delta of the mean MAE
        spread = []
        for k, h in enumerate(d["horizons"]):
            ds_ = [per_seed[str(s)][k]["delta_MAE_mean"] for s in args.seeds]
            spread.append({"horizon": h, "delta_MAE_by_seed": ds_, "mean": float(np.mean(ds_)), "seed_range": float(max(ds_) - min(ds_)),
                           "seed_std": float(np.std(ds_, ddof=1)) if len(ds_) > 1 else None})
        res["paired_vs_A"][a] = {"per_seed": per_seed, "seed_spread": spread}
    res["totals"] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    (out / "RESULTS_MLP.json").write_text(json.dumps(pipe.public(res), indent=1, sort_keys=True, allow_nan=False) + "\n")
    digests = {f"{a}|{s}": v["prediction_sha256"] for a in use for s, v in res["arms"][a]["seeds"].items()}
    digests_w = {f"{a}|{s}": v["weights_sha256"] for a in use for s, v in res["arms"][a]["seeds"].items()}
    (out / "REPLAY_DIGESTS_MLP.json").write_text(json.dumps({"predictions": digests, "weights": digests_w}, indent=1, sort_keys=True) + "\n")
    hb.stage("done")
    hb.stop()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--name", default="h1-eth-mlp")
    ap.add_argument("--variant", default="moments_train")
    ap.add_argument("--lags", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023])
    ap.add_argument("--hidden", type=int, nargs="+", default=[44, 44])
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--heartbeat", type=float, default=60.0)
    args = ap.parse_args()
    res = run(args)
    print(json.dumps(res["totals"]))


if __name__ == "__main__":
    main()
