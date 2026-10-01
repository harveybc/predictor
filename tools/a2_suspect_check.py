"""Scrutiny of the a2 real-cell MAE (coordinator order (a)-(e)); the hash verdicts do not depend on it.

(a) same-row naives: zero-return (z of a 0 log return), train-mean (mean of forecast-train targets),
    last-value persistence (the evaluator's); (b) R0 on the same split and budget through the real evaluator;
(c) pretraining population supports vs the validation input support; (d) future-shift probe: rebuild each
    probed window from a CSV whose rows > t are perturbed, outputs at t must not move; (e) target identity:
    log_return_1[t] == log(CLOSE[t]/CLOSE[t-1]) (so the window holds no t+1 return) and the cell target
    equals M07's Y_1 = z(log_return_1[t+1]) on the same rows.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


def main(proof_dir, data_path, out_json):
    import pandas as pd
    import tensorflow as tf
    import predictor_plugins.modular_temporal  # noqa: F401  (registers the custom layers for load_model)
    from tools import a2_regime_hash_proof as proof
    from tools.modular_candidate_evaluator import evaluate_candidate
    from tools.modular_pretrain import regime_config
    d = Path(proof_dir)
    rec = json.loads((d / "PROOF.json").read_text())
    feats, horizons = rec["cell"]["features"], rec["cell"]["horizons_bars"]
    va = np.load(d / "cell" / "validation.npz")
    tr = np.load(d / "cell" / "train.npz")
    y, x = va["targets"][:, 0, 0].astype("float64"), va["windows"]
    frame = pd.read_csv(data_path)
    frame = frame[pd.to_datetime(frame["DATE_TIME"]) < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    raw = frame[feats].to_numpy("float64")
    fit = raw[proof.FORECAST_TRAIN[0]:proof.FORECAST_TRAIN[1]]
    mean, std = np.nanmean(fit, 0), np.nanstd(fit, 0)
    z0 = (0.0 - mean[0]) / std[0]
    out = {"label": "PILOT_SUSPECT_UNVERIFIED until (a)-(e) pass", "rows": int(len(y))}
    # (a)
    preds = {}
    for regime in ("R1", "R2"):
        m = tf.keras.models.load_model(d / f"forecast_{regime}" / "best.keras", compile=False)
        preds[regime] = np.asarray(m.predict(x, batch_size=512, verbose=0))[:, 0, 0]
    out["a_same_row_MAE_z"] = {
        "zero_return_naive": float(np.mean(np.abs(y - z0))),
        "train_mean_naive": float(np.mean(np.abs(y - tr["targets"][:, 0, 0].mean()))),
        "train_median_naive": float(np.mean(np.abs(y - np.median(tr["targets"][:, 0, 0])))),
        "persistence_naive": float(np.mean(np.abs(y - x[:, -1, 0]))),
        **{f"{r}_model": float(np.mean(np.abs(y - p))) for r, p in preds.items()},
        "R1_pred_std": float(preds["R1"].std()), "R2_pred_std": float(preds["R2"].std()), "y_std": float(y.std())}
    # (b) R0, same split, same budget, same seed
    pre = json.loads((d / "pretrain" / "PRETRAIN.json").read_text())
    r0 = evaluate_candidate({"model": regime_config(pre["fine_tune_config"], "R0"), "target_feature_indices": [0],
                             "evaluator": {**rec["fit"], "seed": rec["seed"]}},
                            d / "cell" / "train.npz", d / "cell" / "validation.npz", d / "forecast_R0_check")
    m0 = tf.keras.models.load_model(d / "forecast_R0_check" / "best.keras", compile=False)
    p0 = np.asarray(m0.predict(x, batch_size=512, verbose=0))[:, 0, 0]
    out["b_R0_same_budget"] = {"MAE_z": float(np.mean(np.abs(y - p0))), "evaluator_MAE": r0["metrics"]["MAE"],
                               "observed_updates": r0["training"]["observed_updates"]}
    # (c) pretraining supports vs validation input support
    vstart = int(va["timestamps"].min() - (x.shape[1] - 1) * 4 * 3600)
    pops = {k: {"start": pre[k]["support_start"], "end": pre[k]["support_end"]}
            for k in ("train_population", "validation_population")}
    out["c_pretraining_population"] = {
        "pretrain_supports_epoch_s": pops, "validation_input_start_epoch_s": vstart,
        "pretrain_npz_split": str(tr["split"]), "disjoint": all(p["end"] < vstart for p in pops.values())}
    # (d) future-shift probe on 64 validation origins
    ts = ((pd.to_datetime(frame["DATE_TIME"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy("int64")
    pos = {int(t): i for i, t in enumerate(ts)}
    rng = np.random.default_rng(0)
    probe = rng.choice(len(y), size=min(64, len(y)), replace=False)
    m1 = tf.keras.models.load_model(d / "forecast_R1" / "best.keras", compile=False)
    moved = 0.0
    for j in probe:
        t = pos[int(va["timestamps"][j])]
        pert = raw.copy()
        pert[t + 1:] += rng.normal(scale=10.0, size=pert[t + 1:].shape) * std
        z = ((pert - mean) / std).astype("float32")
        win = z[t - x.shape[1] + 1:t + 1][None]
        if not np.array_equal(win[0], x[j]):
            moved = float("inf")
            break
        moved = max(moved, float(np.max(np.abs(np.asarray(m1(win))[0, 0, 0] - np.asarray(m1(x[j:j + 1]))[0, 0, 0]))))
    out["d_future_shift_probe"] = {"origins": int(len(probe)), "max_abs_output_change": moved,
                                   "passed": moved == 0.0}
    # (e) target identity
    close = frame["CLOSE"].to_numpy("float64")
    lr = frame["log_return_1"].to_numpy("float64")
    backward = np.log(close[1:] / close[:-1])
    ok = np.isfinite(lr[1:]) & np.isfinite(backward)
    t_idx = np.array([pos[int(t)] for t in va["timestamps"]])
    m07 = (lr[t_idx + 1] - mean[0]) / std[0]
    out["e_target"] = {
        "log_return_1_is_backward_max_abs_diff": float(np.max(np.abs(lr[1:][ok] - backward[ok]))),
        "log_return_1_equals_forward_return_max_abs_diff": float(np.max(np.abs(lr[:-1][ok] - backward[ok]))),
        "cell_target_vs_M07_Y1_max_abs_diff": float(np.max(np.abs(m07 - y))),
        "window_last_step_is_t": bool(np.allclose(x[:, -1, 0], (lr[t_idx] - mean[0]) / std[0], atol=1e-5))}
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
