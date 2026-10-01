"""R1 frozen / R2 updated, proved by weight hashes through the REAL evaluator on a REAL cell (ETH 4h TRAIN).

Order MAINLINE_PARALLEL_CONTINUATION §4.A items 2-3. The cell is built from the git-pinned ETH 4h file
(sha 1b447c66, TRAIN rows [0, 13699) per M07's SPLIT 116a5b64) and features of the frozen variant-A
manifest (fdff0c85): forecast train = inner_3 train [0, 11584), validation = inner_3 validation
[11644, 13699), purged; per-feature z-score on forecast-train rows. Validation and test of the outer split are
never opened.

1. Pretrain donors with M02's tool (``pretrain_from_train_npz``, provenance local_file) on the TRAIN NPZ.
2. Evaluate R1 and R2 with ``evaluate_candidate`` (the real evaluator), same seed.
3. Hash every component (each branch and temporal_core):
   donor file -> R1/R2 bundle at build (before any update) -> best.keras after training.
   R1 must equal the donor before AND after; R2 must equal the donor before and DIFFER after.
4. Cost: pretraining wall seconds (per branch, fusion, core) reported separately from fit seconds.

Labels: the forecast numbers are DEVELOPMENT (single cell, single seed) and carry their same-row
persistence naive; the hash results are the claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

DATA_SHA = "1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f"
TRAIN_ROWS = 13699
FORECAST_TRAIN = (0, 11584)
VALIDATION = (11644, 13699)
DEFAULT_FEATURES = ("log_return_1", "rsi_14", "hist_vol_20")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cell(data_path, out, features=DEFAULT_FEATURES, horizons=(1, 6), window=24, sample_hours=4):
    """Evaluator-format TRAIN/VALIDATION NPZs; target = features[0] (z-scored) at each horizon."""
    import pandas as pd
    if _sha(data_path) != DATA_SHA:
        raise ValueError("data bytes are not the pinned ETH 4h file")
    frame = pd.read_csv(data_path)
    stamps = pd.to_datetime(frame["DATE_TIME"])
    frame = frame[stamps < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    if len(frame) != TRAIN_ROWS:
        raise ValueError("TRAIN row count differs from the SPLIT artefact")
    ts = ((pd.to_datetime(frame["DATE_TIME"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy("int64")
    raw = frame[list(features)].to_numpy("float64")
    fit = raw[FORECAST_TRAIN[0]:FORECAST_TRAIN[1]]
    mean, std = np.nanmean(fit, 0), np.nanstd(fit, 0)
    z = ((raw - mean) / std).astype("float32")
    step, hmax = sample_hours * 3600, max(horizons)

    def origins(lo, hi):
        o = np.arange(lo + window - 1, hi - hmax)
        ok = []
        for t in o:
            win = z[t - window + 1:t + 1]
            regular = np.all(np.diff(ts[t - window + 1:t + 1]) == step)
            targets = all(ts[t + h] == ts[t] + h * step for h in horizons)
            ok.append(regular and targets and np.isfinite(win).all()
                      and all(np.isfinite(z[t + h, 0]) for h in horizons))
        return o[np.asarray(ok, dtype=bool)]

    def pack(o, split):
        return dict(windows=np.stack([z[t - window + 1:t + 1] for t in o]),
                    targets=np.stack([z[o + h, 0] for h in horizons], axis=1)[..., None].astype("float32"),
                    row_ids=np.array([f"eth4h-{int(t)}" for t in o]), timestamps=ts[o],
                    target_timestamps=ts[o][:, None] + np.asarray(horizons, dtype=np.int64) * step,
                    dataset_id=np.array("ethusdt_4h_tech_stat_model_ready@1b447c66"), split=np.array(split),
                    feature_names=np.array(list(features)), target_names=np.array([features[0]]),
                    horizons=np.asarray(horizons, dtype=np.int64), timestamp_unit=np.array("seconds"),
                    metric_space=np.array("z_forecast_train"),
                    scaler_identity=np.array("eth4h-forecast-train-zscore-v1"),
                    scaler_scale=np.array([std[0]], dtype=np.float64))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr = origins(*FORECAST_TRAIN)
    va = origins(*VALIDATION)
    va = va[va - (window - 1) > tr.max() + hmax]
    np.savez(out / "train.npz", **pack(tr, "train"))
    np.savez(out / "validation.npz", **pack(va, "validation"))
    return out / "train.npz", out / "validation.npz", len(tr), len(va)


def _component_hashes(model, names):
    from predictor_plugins.modular_temporal import weights_hash
    return {n: weights_hash(model.get_layer(n)) for n in names}


def run_proof(data_path, out, *, features=DEFAULT_FEATURES, horizons=(1, 6), fit=None, seed=7):
    import tensorflow as tf
    from predictor_plugins.modular_temporal import build_modular
    from tools.modular_candidate_evaluator import evaluate_candidate
    from tools.modular_pretrain import pretrain_from_train_npz, regime_config
    out = Path(out)
    if out.exists():
        raise ValueError("output directory must not exist")
    fit = dict(fit or dict(max_epochs=10, patience=3, batch_size=64, loss="mae", learning_rate=1e-3))
    started = time.monotonic()
    train_npz, val_npz, n_tr, n_va = build_cell(data_path, out / "cell", features, horizons)
    cell_seconds = time.monotonic() - started
    from predictor_plugins.modular_temporal import default_config
    base = default_config(list(features))
    base.update(sample_hours=4, horizons=list(horizons))
    t0 = time.monotonic()
    pre = pretrain_from_train_npz(train_npz, out / "pretrain", fit, provenance="local_file", config=base,
                                  seed=seed)
    pretrain_seconds = time.monotonic() - t0
    names = [b["name"] for b in pre["branches"]] + ["temporal_core"]
    donors = {b["name"]: b["donor_weights_sha256"] for b in pre["branches"]}
    donors["temporal_core"] = pre["core"]["donor_weights_sha256"]
    regimes, contract = {}, "OPERATIONAL"
    for regime in ("R1", "R2"):
        model = regime_config(pre["fine_tune_config"], regime)
        try:
            tf.keras.backend.clear_session()
            built = build_modular(model)
        except ValueError as exc:                       # local_file donors may not be OPERATIONAL
            if "contract" not in str(exc).lower() and "provenance" not in str(exc).lower():
                raise
            contract = "UNKNOWN_ALLOWED"
            model["donor_contract"] = contract
            built = build_modular(model)
        before = _component_hashes(built.forecast_model, names)
        trainable = {n: len(built.forecast_model.get_layer(n).trainable_weights) for n in names}
        evaluator = {k: v for k, v in fit.items()}
        evaluator["seed"] = seed
        receipt = evaluate_candidate({"model": model, "target_feature_indices": [0], "evaluator": evaluator},
                                     train_npz, val_npz, out / f"forecast_{regime}")
        best = tf.keras.models.load_model(out / f"forecast_{regime}" / "best.keras", compile=False)
        after = _component_hashes(best, names)
        regimes[regime] = {
            "before_equals_donor": {n: before[n] == donors[n] for n in names},
            "after_equals_donor": {n: after[n] == donors[n] for n in names},
            "trainable_weight_tensors_at_build": trainable,
            "hashes": {"donor": donors, "before": before, "after": after},
            "fit_seconds": receipt["training"]["elapsed_seconds"],
            "observed_updates": receipt["training"]["observed_updates"],
            "initial_weights_sha256": receipt["digests"]["initial_weights_sha256"],
            "validation": {"MAE": receipt["metrics"]["MAE"], "persistence_MAE": receipt["metrics"]["baseline_MAE"],
                           "skill_MAE": receipt["metrics"]["skill_MAE"],
                           "per_horizon": {h: {"MAE": v["MAE"], "persistence_MAE": v["baseline_MAE"]}
                                           for h, v in receipt["per_horizon"].items()}}}
    r1, r2 = regimes["R1"], regimes["R2"]
    verdict = {
        "R1_frozen_bit_for_bit": all(r1["before_equals_donor"].values()) and all(r1["after_equals_donor"].values())
        and all(v == 0 for v in r1["trainable_weight_tensors_at_build"].values()),
        "R2_starts_from_same_donors": all(r2["before_equals_donor"].values()),
        "R2_updates_every_component": not any(r2["after_equals_donor"].values()),
        "R1_R2_same_initial_model": r1["initial_weights_sha256"] == r2["initial_weights_sha256"]}
    result = {"schema": "a2.regime_hash_proof.v1", "label": "DEVELOPMENT (hash proof; one cell, one seed)",
              "cell": {"data_sha256": DATA_SHA, "features": list(features), "target": features[0],
                       "horizons_bars": list(horizons), "forecast_train_rows": n_tr, "validation_rows": n_va,
                       "split": "inner_3 of TRAIN [0,13699) (M07 SPLIT 116a5b64); outer validation/test unopened",
                       "train_npz_sha256": _sha(train_npz), "validation_npz_sha256": _sha(val_npz)},
              "donor_contract_used": contract, "pretrain_provenance": "local_file", "fit": fit, "seed": seed,
              "cost_seconds": {"cell_build": cell_seconds, "pretraining_total": pretrain_seconds,
                               "pretraining_branches": {b["name"]: b["wall_seconds"] for b in pre["branches"]},
                               "pretraining_fusion": pre["fusion"]["wall_seconds"],
                               "pretraining_core": pre["core"]["wall_seconds"],
                               "fit_R1": r1["fit_seconds"], "fit_R2": r2["fit_seconds"]},
              "regimes": regimes, "verdict": verdict}
    (out / "PROOF.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-epochs", type=int, default=10)
    a = p.parse_args()
    r = run_proof(a.data, a.out, fit=dict(max_epochs=a.max_epochs, patience=3, batch_size=64, loss="mae",
                                          learning_rate=1e-3))
    print(json.dumps({"verdict": r["verdict"], "cost_seconds": r["cost_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
