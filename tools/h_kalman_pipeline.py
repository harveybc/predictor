"""Lane H: build the Kalman outputs from TRAIN-fitted artifacts, assemble the paired arms and controls, evaluate.

Data contract (``data`` dict, built by the dataset runners): Z (rows x features, train-standardized), names,
train_rows [0, n_tr), val_rows [n_tr, n_end) (rows beyond n_end, the protected test rows, are never passed in),
origins {train, validation}, Y {train, validation} (N x H cumulative standardized returns), target_series (the
standardized 1-bar target per row, for the persistence and seasonal naives), mu, sigma, horizons, window,
seasonal_period, optional dataset_id / split_sha256.
"""
from __future__ import annotations

import importlib.util
import math
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
arms_lib = _load("h_kalman_arms")

# declared variants: parameter overrides per group (nothing outside the predeclared grid of the family)
VARIANTS = {
    "moments_train": {"local_level": {}, "local_linear_trend": {}},
    "declared_1e-3": {"local_level": {"param_source": "declared_ratio", "ratio_level": 0.001},
                      "local_linear_trend": {"param_source": "declared_ratio", "ratio_level": 0.001, "ratio_slope": 1e-04}},
    "declared_1e-2": {"local_level": {"param_source": "declared_ratio", "ratio_level": 0.01},
                      "local_linear_trend": {"param_source": "declared_ratio", "ratio_level": 0.01, "ratio_slope": 1e-04}},
    "declared_1e-1": {"local_level": {"param_source": "declared_ratio", "ratio_level": 0.1},
                      "local_linear_trend": {"param_source": "declared_ratio", "ratio_level": 0.1, "ratio_slope": 1e-04}},
}
KIND_OF = {"local_level": kf.LOCAL_LEVEL, "local_linear_trend": kf.LOCAL_LINEAR_TREND}


def public(obj):
    """Strip private (underscore) keys so a result dict is plain JSON."""
    if isinstance(obj, dict):
        return {k: public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [public(v) for v in obj]
    return obj


def build_kalman(data, groups, variant, host_role=None):
    names = list(data["names"])
    seen = set()
    for g, cols in groups.items():
        for c in cols:
            if c not in names:
                raise kf.OperatorRefusal(f"declared feature {c!r} is not a column of the panel")
            if c in seen:
                raise kf.OperatorRefusal(f"feature {c!r} is declared in two groups")
            seen.add(c)
    tr0, tr1 = data["train_rows"]
    end = data["val_rows"][1]
    out = {}
    for g in ("local_level", "local_linear_trend"):
        cols = groups.get(g) or []
        if not cols:
            continue
        idx = [names.index(c) for c in cols]
        spec = kf.default_spec(KIND_OF[g], **variant.get(g, {}))
        binding = {"dataset_id": data.get("dataset_id", "synthetic"), "role": "TRAIN", "row_range": [tr0, tr1],
                   "split_sha256": data.get("split_sha256"), "column_ids": list(cols),
                   "units": "train-standardized feature units (z_train)",
                   "scale_note": "each feature standardized with its TRAIN mean/std before filtering"}
        art = kf.fit(spec, data["Z"][tr0:tr1][:, idx], binding, host_role=host_role)
        if art["status"] != "FITTED":
            raise kf.OperatorAbstain(art["abstain_reason"])
        o = kf.transform_batch(art, data["Z"][:end][:, idx])
        out[g] = {"names": list(cols), "idx": idx, "artifact": art, "output": o}
    return out


def _windows(C, data, lags):
    return {s: arms_lib.gather(C, data["origins"][s], lags) for s in ("train", "validation")}


def _arm(C, names, data, lags, **extra):
    w = _windows(C, data, lags)
    d = {"train": w["train"], "validation": w["validation"], "names": list(names), "lags": lags, "eligible": True}
    d.update(extra)
    return d


def arm_matrices(data, kal, lags=1, controls=False):
    Z = data["Z"][:data["val_rows"][1]]
    names = list(data["names"])
    declared = [c for g in kal.values() for c in g["names"]]
    keep = [i for i, n in enumerate(names) if n not in declared]
    Ks, Kn = [], []
    for g in kal.values():
        M, lab = arms_lib.kalman_channels(g["output"])
        Ks.append(M)
        Kn += lab
    K = np.concatenate(Ks, axis=1) if Ks else np.zeros((Z.shape[0], 0))
    arms = {"A": _arm(Z, names, data, lags),
            "IDENTITY": _arm(kf.identity_control(Z), names, data, lags),
            "B": _arm(np.concatenate([Z, K], axis=1), names + Kn, data, lags),
            "C": _arm(np.concatenate([Z[:, keep], K], axis=1), [names[i] for i in keep] + Kn, data, lags)}
    if controls:
        n_tr = data["train_rows"][1]
        P = kf.permutation_control(K, [n_tr], seed=20261001)
        N = kf.noise_control(K, n_tr, seed=20261002)
        arms["B_PERMUTED"] = _arm(np.concatenate([Z, P], axis=1), names + Kn, data, lags, label="PERMUTATION_CONTROL")
        arms["B_NOISE"] = _arm(np.concatenate([Z, N], axis=1), names + Kn, data, lags, label="NOISE_CONTROL")
        ew_cols, ew_names = [], []
        sm_cols, sm_names = [], []
        for gname, g in kal.items():
            e = kf.ewma_comparable(g["artifact"], Z[:, g["idx"]])
            sm = kf.smoother_control(g["artifact"], Z[:, g["idx"]])
            for j, c in enumerate(g["names"]):
                ew_cols += [e["level"][:, j], e["innov"][:, j]]
                ew_names += [f"{c}__ewma_level", f"{c}__ewma_innov"]
                sm_cols.append(sm.arrays["level"][:, j])
                sm_names.append(f"{c}__smoothed_level_NONCAUSAL")
                if "slope" in sm.arrays:
                    sm_cols.append(sm.arrays["slope"][:, j])
                    sm_names.append(f"{c}__smoothed_slope_NONCAUSAL")
        E = np.stack(ew_cols, axis=1) if ew_cols else np.zeros((Z.shape[0], 0))
        S = np.stack(sm_cols, axis=1) if sm_cols else np.zeros((Z.shape[0], 0))
        arms["C_EWMA"] = _arm(np.concatenate([Z[:, keep], E], axis=1), [names[i] for i in keep] + ew_names, data, lags,
                              label="COMPARABLE_CAUSAL_EWMA")
        arms["C_SMOOTHER_NONCAUSAL"] = _arm(np.concatenate([Z[:, keep], S], axis=1), [names[i] for i in keep] + sm_names,
                                            data, lags, eligible=False, label="NON_CAUSAL_NEGATIVE_CONTROL")
    return arms


def _naive_block(data):
    return arms_lib.naive_predictions(data["target_series"], data["origins"]["validation"], data["horizons"],
                                      data["mu"], data["sigma"], data["seasonal_period"])


def evaluate_arm(data, arm):
    Yva, Ytr = data["Y"]["validation"], data["Y"]["train"]
    (pred,), alpha, inner = arms_lib.ridge_fit_predict(arm["train"], Ytr, [arm["validation"]])
    nv = _naive_block(data)
    rows = []
    q90 = np.quantile(np.abs(Yva), 0.9, axis=0)
    for k, h in enumerate(data["horizons"]):
        y, p = Yva[:, k], pred[:, k]
        mae, mse = float(np.mean(np.abs(p - y))), float(np.mean((p - y) ** 2))
        naive = {}
        for name, arr in nv.items():
            e = arr[:, k] - y
            if np.all(np.isfinite(e)):
                nm, ns = float(np.mean(np.abs(e))), float(np.mean(e ** 2))
                naive[name] = {"MAE": nm, "MSE": ns, "skill_MAE": 1.0 - mae / nm, "skill_MSE": 1.0 - mse / ns}
            else:
                naive[name] = {"MAE": None, "MSE": None, "skill_MAE": None, "skill_MSE": None,
                               "status": "NOT_AVAILABLE_HORIZON_EXCEEDS_PERIOD"}
        avail = {n: v["MAE"] for n, v in naive.items() if v["MAE"] is not None}
        best = min(avail, key=lambda n: (avail[n], n))
        top = np.abs(y) >= q90[k]
        rows.append({"horizon": int(h), "rows": int(len(y)), "model_MAE": mae, "model_MSE": mse, "naive": naive,
                     "strict_naive": best, "strict_naive_MAE": naive[best]["MAE"], "strict_naive_MSE": naive[best]["MSE"],
                     "beats_zero_return_MAE_and_MSE": bool(mae < naive["zero_return"]["MAE"] and mse < naive["zero_return"]["MSE"]),
                     "extremes": {"MAE_top_decile_abs_target": float(np.mean(np.abs(p[top] - y[top]))),
                                  "naive_zero_return_MAE_same_rows": float(np.mean(np.abs(nv["zero_return"][top, k] - y[top]))),
                                  "prediction_std_over_target_std": float(p.std() / y.std())}})
    return {"alpha": float(alpha), "inner_holdout_MAE_by_alpha": {str(a): v for a, v in inner.items()},
            "rows": int(Yva.shape[0]), "per_horizon": rows, "channels": int(arm["train"].shape[1]),
            "eligible": bool(arm["eligible"]), "label": arm.get("label"),
            "mean_model_MAE": float(np.mean([r["model_MAE"] for r in rows])),
            "prediction_sha256": arms_lib.sha_array(pred), "_pred": pred}


def paired_against(data, base_ev, arm_ev, L, B=2000, seed=0):
    Yva = data["Y"]["validation"]
    out = []
    for k, h in enumerate(data["horizons"]):
        y = Yva[:, k]
        eb, ea = base_ev["_pred"][:, k] - y, arm_ev["_pred"][:, k] - y
        d_mae = np.abs(ea) - np.abs(eb)
        d_mse = ea ** 2 - eb ** 2
        m, lo, hi = arms_lib.block_bootstrap_ci(d_mae, L, B=B, seed=seed + k)
        ms, los, his = arms_lib.block_bootstrap_ci(d_mse, L, B=B, seed=seed + 100 + k)
        out.append({"horizon": int(h), "base_MAE": float(np.mean(np.abs(eb))), "arm_MAE": float(np.mean(np.abs(ea))),
                    "delta_MAE_mean": m, "ci95": [lo, hi], "quarter_deltas": arms_lib.quarter_deltas(d_mae),
                    "delta_MSE_mean": ms, "ci95_MSE": [los, his], "block_length_rows": int(L),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return out


def kalman_diagnostics(data, kal):
    """Per group: innovation stability (train vs validation), phase/lag, extremes, flagged-row counts, state size."""
    tr0, tr1 = data["train_rows"]
    end = data["val_rows"][1]
    diag = {}
    for g, k in kal.items():
        o = k["output"]
        Zg = data["Z"][:end][:, k["idx"]]
        per = []
        for j, c in enumerate(k["names"]):
            z = o.arrays["zinnov"][:, j]
            lag = arms_lib.phase_lag(Zg[tr1:, j], o.arrays["level"][tr1:, j])
            ex = arms_lib.extremes_retained(Zg[tr1:, j], o.arrays["level"][tr1:, j], z[tr1:],
                                            float(np.median(Zg[tr0:tr1, j])), float(Zg[tr0:tr1, j].std()))
            per.append({"feature": c, "innovation": arms_lib.innovation_stability(z[tr0:tr1], z[tr1:]),
                        "phase_lag_validation": {"best_lag_bars": lag["best_lag_bars"], "corr_at_lag0": lag["corr_by_lag"][0]},
                        "extremes_validation": ex,
                        "params": {key: k["artifact"]["fitted"]["per_column"][j].get(key)
                                   for key in ("r", "q", "qs", "r_clipped", "q_clipped", "qs_clipped",
                                               "ratio_level_effective", "ratio_slope_effective")}})
        diag[g] = {"features": per, "reason_counts": o.reason_counts(), "output_digest": o.digest(),
                   "artifact_sha256": k["artifact"]["artifact_sha256"],
                   "fitted_state_digest": k["artifact"]["fitted_state_digest"]}
    return diag
