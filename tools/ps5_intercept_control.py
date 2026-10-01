"""PS5 intercept-only control (lane B request): a constant equal to the fold-train mean (and median) of y,
scored on EXACTLY the record's validation rows (val_rows_sha256 re-derived and checked), next to the
zero-return naive and the joint arm. No model is fitted; pandas/numpy only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tools import ps5_joint_reentry as ps5


def control(record, cols, train, ts):
    a, b = record["pair"]
    hours = int(record["target"].split("@")[1].rstrip("h"))
    y = ps5._target(train, ts, hours)
    fold = ps5.FOLDS[record["fold"]]
    names = record["arms"][f"base+{a}+{b}"]["features"]
    enc, enc_val, val, cut = ps5._segments(fold, len(y), 24, hours // 4)
    keep = lambda o: ps5._finite_windows(cols, names, o[np.isfinite(y[o])], 24)
    enc, enc_val, val = keep(enc), keep(enc_val), keep(val)
    sha = hashlib.sha256(val.astype("int64").tobytes()).hexdigest()
    if sha != record["naive"]["rows_sha256"]:
        raise SystemExit(f"row set differs for {record['fold']} {record['pair']} {record['target']}")
    fit = np.concatenate([enc, enc_val])             # every fold-train origin the model's fit could see
    truth = y[val]
    mean_all, mean_enc, med = float(y[fit].mean()), float(y[enc].mean()), float(np.median(y[fit]))
    joint = record["arms"][f"base+{a}+{b}"]
    seeds = [s["val_mae"] for s in joint["per_seed"].values()]
    best_const = min(float(np.mean(np.abs(truth - c))) for c in (mean_all, med, 0.0))   # beat ALL three
    return {"fold": record["fold"], "pair": f"{a}*{b}", "target": record["target"], "rows": int(len(val)),
            "zero_naive": float(np.mean(np.abs(truth))),
            "train_mean_const": float(np.mean(np.abs(truth - mean_all))),
            "encoder_train_mean_const": float(np.mean(np.abs(truth - mean_enc))),
            "train_median_const": float(np.mean(np.abs(truth - med))),
            "train_mean": mean_all, "joint_mean": joint["mean_val_mae"], "joint_seeds": seeds,
            "decision": record["decision"],
            "joint_beats_zero_mean_median_beyond_spread_both_seeds": (best_const - joint["mean_val_mae"]
                                                         > (max(seeds) - min(seeds)) / 2) and max(seeds) < best_const}


def main(data, run_dir, out):
    train, ts = ps5._eth(data)
    cols = {c: train[c].to_numpy("float64") for c in train.columns if c != "DATE_TIME"}
    rows = [control(json.loads(p.read_text()), cols, train, ts) for p in sorted(Path(run_dir).glob("ps5*_[0-9]*.json"))]
    Path(out).write_text(json.dumps({"label": "DEVELOPMENT", "rows": rows}, indent=1) + "\n")
    for r in rows:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "joint_seeds"}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
