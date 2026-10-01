#!/usr/bin/env python3
"""Lane H runner: Kalman arms on the ECL testbed (M04 v2 NPZ). DEVELOPMENT, CPU, worker_b. Modes: ridge | mlp."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
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


ecl = _load("h_kalman_ecl")
eth_run = _load("h_kalman_eth_run")
pipe, kf, arms_lib = ecl.pipe, ecl.kf, ecl.arms_lib
PINNED = {"MANIFEST.json": "16a3e5354383b364a6becc7720625a4625e3c990bfbebcb9a62c94eb3faac7f1",
          "train.npz": "9215b099442928a811fddaa6fbde07ed26a625f1707f7f45d8fec24fcfda25c2",
          "validation.npz": "e37125655ebeb161837fcc826073597e1266470b49a4a40f90a018e989e6a7e9"}
M04_SEASONAL_MEAN = 0.247966


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        pos = 0
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
            try:          # drop the page cache just read: the 1.1 GB input must not inflate this job's cgroup charge
                os.posix_fadvise(f.fileno(), pos, len(b), os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            pos += len(b)
    return h.hexdigest()


def paired(perorig_base, perorig_arm, L=48, B=1000, seed=0):
    out = []
    for k in range(perorig_base.shape[1]):
        dlt = perorig_arm[:, k] - perorig_base[:, k]
        m, lo, hi = arms_lib.block_bootstrap_ci(dlt, L, B=B, seed=seed + k)
        out.append({"horizon": k + 1, "delta_MAE_mean": m, "ci95": [lo, hi], "quarter_deltas": arms_lib.quarter_deltas(dlt)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--mode", choices=("ridge", "mlp"), default="ridge")
    ap.add_argument("--variants", nargs="+", default=["moments_train", "declared_1e-1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    ap.add_argument("--mlp-rows", type=int, default=300000)
    ap.add_argument("--name", default="h1-ecl")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = eth_run.Heartbeat(out, a.name, interval=60.0)
    hb.start()
    t0, c0 = time.time(), time.process_time()
    D = Path(a.data_dir)
    hb.stage("digests")
    shas = {k: sha_file(D / k) for k in PINNED}
    if shas != PINNED:
        raise SystemExit(f"input digest mismatch: {shas}")
    hb.stage("reconstruct")
    rows, origins = ecl.reconstruct_rows(D / "train.npz", D / "validation.npz")
    if np.isnan(rows).any():
        raise SystemExit("row panel has gaps")
    d = ecl.panel(rows, origins, (0, 18412))
    Z, o_tr, o_va = d["Z"], origins["train"], origins["validation"]
    base_rows, base_perorig = ecl.score(Z, o_va, np.zeros((len(o_va), 24, d["C"])))
    seas_mean = float(np.mean([r["seasonal_MAE"] for r in base_rows]))
    res = {"schema": f"lane_h_kalman_ecl_{a.mode}.v1", "label": "DEVELOPMENT", "role": a.role, "inputs": shas,
           "rows_sha256": d["rows_sha256"], "train_origins": int(len(o_tr)), "validation_origins": int(len(o_va)),
           "channels": d["C"], "seasonal_naive_mean_MAE": seas_mean,
           "seasonal_matches_M04": abs(seas_mean - M04_SEASONAL_MEAN) < 5e-7, "naive_by_horizon": base_rows,
           "environment": kf.environment_record(a.role), "variants": {}}
    replay = {"rows_sha256": d["rows_sha256"], "kalman": {}, "predictions": {}}
    tgt = lambda oo: ecl.residual_targets(Z, oo).transpose(0, 2, 1).reshape(-1, 24)
    for vn in a.variants:
        hb.stage(f"{vn}:kalman")
        tk, ck = time.time(), time.process_time()
        kal = ecl.kalman_blocks(d, pipe.VARIANTS[vn])
        kcost = {"wall_s": time.time() - tk, "cpu_s": time.process_time() - ck}
        pc = kal["per_column"]
        replay["kalman"][vn] = {"artifact_sha256s": kal["artifact_sha256s"], "fitted_state_digests": kal["fitted_state_digests"],
                                "output_digest": kal["output_digest"], "reason_counts": kal["reason_counts"]}
        arms = ecl.arm_blocks(d, kal)
        v = {"kalman_cost": kcost, "fit_rows": kal["fit_rows"], "clipped_r": int(sum(c["r_clipped"] for c in pc)),
             "clipped_q": int(sum(c["q_clipped"] for c in pc)), "arms": {}, "paired_vs_A": {}}
        use = ["A", "B", "C", "C_EWMA", "B_PERMUTED", "B_NOISE", "C_SMOOTHER_NONCAUSAL"] if a.mode == "ridge" else \
            ["A", "B", "C", "B_PERMUTED"]
        perorig = {}
        for arm in use:
            blocks = arms[arm]["blocks"]()
            feats = lambda oo, blocks=blocks: ecl.channel_features(blocks, oo, 24)
            if a.mode == "ridge":
                hb.stage(f"{vn}:ridge:{arm}")
                ta, ca = time.time(), time.process_time()
                fit = ecl.pooled_ridge(feats, tgt, o_tr)
                pr = ecl.ridge_predict(fit, feats, o_va).reshape(len(o_va), d["C"], 24).transpose(0, 2, 1)
                rows_, po = ecl.score(Z, o_va, pr)
                perorig[arm] = po
                replay["predictions"][f"{vn}|{arm}"] = arms_lib.sha_array(pr)
                v["arms"][arm] = {"alpha": fit["alpha"], "inner_holdout_MAE": fit["inner_holdout_MAE"], "per_horizon": rows_,
                                  "mean_MAE": float(np.mean([r["model_MAE"] for r in rows_])), "eligible": arms[arm]["eligible"],
                                  "features_per_channel": len(blocks) * 24,
                                  "cost": {"wall_s": time.time() - ta, "cpu_s": time.process_time() - ca}}
            else:
                mlp = _load("h_kalman_mlp")
                rng = np.random.RandomState(7)
                pick = np.sort(rng.choice(len(o_tr) * d["C"], size=min(a.mlp_rows, len(o_tr) * d["C"]), replace=False))
                oi, ch = pick // d["C"], pick % d["C"]
                Xtr = np.empty((len(pick), len(blocks) * 24), dtype=np.float32)
                Ytr = np.empty((len(pick), 24), dtype=np.float32)
                for s in range(0, len(pick), 200000):
                    sl = slice(s, s + 200000)
                    oo = o_tr[oi[sl]]
                    cols = [M[oo - l, ch[sl]] for M in blocks for l in range(24)]
                    Xtr[sl] = np.stack(cols, axis=1)
                    Ytr[sl] = np.stack([Z[oo + h, ch[sl]] - Z[oo + h - 24, ch[sl]] for h in range(1, 25)], axis=1)
                Xva = feats(o_va).astype(np.float32)
                v["arms"][arm] = {"seeds": {}, "features_per_channel": len(blocks) * 24, "train_rows_sampled": int(len(pick))}
                pos = []
                for seed in a.seeds:
                    hb.stage(f"{vn}:mlp:{arm}:{seed}")
                    ta, ca = time.time(), time.process_time()
                    o_ = mlp.mlp_fit_predict(Xtr, Ytr, [Xva], seed=seed, hidden=(64, 64), max_epochs=30, patience=5,
                                             batch_size=256)
                    pr = o_["predictions"][0].reshape(len(o_va), d["C"], 24).transpose(0, 2, 1)
                    rows_, po = ecl.score(Z, o_va, pr)
                    pos.append(po)
                    replay["predictions"][f"{vn}|{arm}|{seed}"] = arms_lib.sha_array(pr)
                    v["arms"][arm]["seeds"][str(seed)] = {
                        "per_horizon": rows_, "mean_MAE": float(np.mean([r["model_MAE"] for r in rows_])),
                        "selected_epoch": o_["selected_epoch"], "epochs_run": o_["epochs_run"],
                        "trainable_parameters": o_["trainable_parameters"],
                        "cost": {"wall_s": time.time() - ta, "cpu_s": time.process_time() - ca}}
                perorig[arm] = pos
                del Xtr, Ytr, Xva
        for arm in use:
            if arm == "A":
                continue
            if a.mode == "ridge":
                v["paired_vs_A"][arm] = paired(perorig["A"], perorig[arm])
            else:
                per_seed = [paired(perorig["A"][i], perorig[arm][i], B=300, seed=s) for i, s in enumerate(a.seeds)]
                spread = []
                for k in range(24):
                    ds = [p[k]["delta_MAE_mean"] for p in per_seed]
                    spread.append({"horizon": k + 1, "delta_by_seed": ds, "mean": float(np.mean(ds)),
                                   "seed_range": float(max(ds) - min(ds))})
                v["paired_vs_A"][arm] = {"per_seed": per_seed, "seed_spread": spread}
        res["variants"][vn] = v
        del arms, kal
    res["totals"] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0,
                     "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    (out / f"RESULTS_ECL_{a.mode}.json").write_text(json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (out / f"REPLAY_ECL_{a.mode}.json").write_text(json.dumps(replay, indent=1, sort_keys=True) + "\n")
    hb.stage("done")
    hb.stop()
    print(json.dumps({"seasonal_mean": seas_mean, "totals": res["totals"]}))


if __name__ == "__main__":
    main()
