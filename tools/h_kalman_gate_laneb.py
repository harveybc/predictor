"""Lane H: replicate lane B's Kalman gate so a pass here means what it means in lane B's ledger. DEVELOPMENT.

Recipe (lane B, feature-eng STAGE_OUTCOME.*.v1): Kalman local level on log close; q and r from the fold-TRAIN moments of the
first differences (var(dx) = q + 2r, lag-1 covariance = -r); causal filter; features deviation (log close - level),
innovation and level change; redundancy filter |r| >= 0.95 (keep in that order); ridge alpha 1.0 on fold-TRAIN
standardized features; three expanding chronological inner folds inside TRAIN (feature-eng tools/progressive_selection.py
``inner_folds``, copied here, source sha d0679a3d); gate = MAE AND MSE strictly below BOTH the zero-return naive and the
intercept-only (fold-TRAIN mean) control on identical rows, labels located at the horizon (here the lane F2 cumulative
standardized target, equivalent units: zero-return predicts Y_h = -h*mu/sigma).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")


def inner_folds(n, k=3, val_frac=0.15, purge=0):
    """Expanding chronological folds inside TRAIN [0, n): train [0, e), purge gap, then validation (lane B)."""
    v = max(1, int(round(n * val_frac)))
    folds = []
    for j in range(1, k + 1):
        val_end = n - (k - j) * v
        val_start = val_end - v
        train_end = val_start - purge
        if train_end <= 1:
            raise ValueError("fold has no training rows; reduce k, val_frac or purge")
        folds.append({"name": f"inner_{j}", "train": (0, train_end), "val": (val_start, val_end), "purge": purge})
    return folds


def redundancy_filter(F, names, threshold=0.95):
    keep = []
    for j in range(F.shape[1]):
        if all(abs(np.corrcoef(F[:, j], F[:, i])[0, 1]) < threshold for i in keep):
            keep.append(j)
    return keep


def _ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xva - mu) / sd
    ym = ytr.mean(axis=0)
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym


def lane_b_gate(logclose, origins, Y, n_train, horizons, mu, sigma, purge=30, k=3, val_frac=0.15):
    """``origins``/``Y`` are the TRAIN-split origins and their cumulative standardized targets (rows < n_train)."""
    lc = np.asarray(logclose, dtype=np.float64)
    origins = np.asarray(origins)
    hmax = max(horizons)
    folds = inner_folds(n_train, k=k, val_frac=val_frac, purge=purge)
    out = {"label": "DEVELOPMENT", "recipe": __doc__.splitlines()[0], "per_horizon": {str(h): {"folds": {}} for h in horizons},
           "features_after_redundancy_filter": None, "fold_parameters": {}}
    for fd in folds:
        a, b = fd["train"]
        va0, va1 = fd["val"]
        art = kf.fit(kf.default_spec(kf.LOCAL_LEVEL), lc[a:b].reshape(-1, 1),
                     {"dataset_id": "lane_b_gate", "role": "TRAIN", "row_range": [a, b], "column_ids": ["logclose"]})
        if art["status"] != "FITTED":
            raise kf.OperatorAbstain(art["abstain_reason"])
        o = kf.transform_batch(art, lc[:va1].reshape(-1, 1))
        level, innov = o.arrays["level"][:, 0], o.arrays["innov"][:, 0]
        dev = lc[:va1] - level
        dlev = np.diff(level, prepend=level[0])
        F = np.stack([dev, innov, dlev], axis=1)
        names = ["deviation", "innovation", "level_change"]
        tr = origins[(origins >= 1) & (origins + hmax < b)]
        va = origins[(origins >= va0) & (origins + hmax < va1)]
        ti = np.searchsorted(origins, tr)
        vi = np.searchsorted(origins, va)
        keep = redundancy_filter(F[tr], names, 0.95)
        out["features_after_redundancy_filter"] = [names[i] for i in keep]
        out["fold_parameters"][fd["name"]] = {"r": art["fitted"]["per_column"][0]["r"], "q": art["fitted"]["per_column"][0]["q"],
                                              "r_clipped": art["fitted"]["per_column"][0]["r_clipped"],
                                              "q_clipped": art["fitted"]["per_column"][0]["q_clipped"], "kept": [names[i] for i in keep]}
        Ftr, Fva = F[tr][:, keep], F[va][:, keep]
        Ytr, Yva = Y[ti], Y[vi]
        pred = _ridge(Ftr, Ytr, Fva, 1.0)
        icp = np.broadcast_to(Ytr.mean(axis=0), Yva.shape)
        for k_, h in enumerate(horizons):
            zero = -h * mu / sigma
            e = pred[:, k_] - Yva[:, k_]
            ez = zero - Yva[:, k_]
            ei = icp[:, k_] - Yva[:, k_]
            m = {"model_MAE": float(np.mean(np.abs(e))), "model_MSE": float(np.mean(e ** 2)),
                 "zero_return_MAE": float(np.mean(np.abs(ez))), "zero_return_MSE": float(np.mean(ez ** 2)),
                 "intercept_MAE": float(np.mean(np.abs(ei))), "intercept_MSE": float(np.mean(ei ** 2)),
                 "train_rows": int(len(tr)), "val_rows": int(len(va))}
            m["beats_both_MAE_and_MSE"] = bool(m["model_MAE"] < min(m["zero_return_MAE"], m["intercept_MAE"]) and
                                               m["model_MSE"] < min(m["zero_return_MSE"], m["intercept_MSE"]))
            out["per_horizon"][str(h)]["folds"][fd["name"]] = m
    for h in horizons:
        ph = out["per_horizon"][str(h)]
        ph["passes_all_folds"] = all(v["beats_both_MAE_and_MSE"] for v in ph["folds"].values())
        ph["mean_model_MAE"] = float(np.mean([v["model_MAE"] for v in ph["folds"].values()]))
        ph["mean_zero_return_MAE"] = float(np.mean([v["zero_return_MAE"] for v in ph["folds"].values()]))
        ph["mean_intercept_MAE"] = float(np.mean([v["intercept_MAE"] for v in ph["folds"].values()]))
    return out
