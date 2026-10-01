"""Lane C2: paired per-row loss-difference inference for a forecast model against the zero-return and intercept-only controls,
with the calibrated block bootstrap of `c2_interval_rule` (critical-path step 4, DEVELOPMENT).

Inputs: M07's PREDICTIONS_<cid>.csv (model predictions y_hat_z_h{h} in z-units per validation origin, row id `...:row<R>:<epoch>`) and its
EVIDENCE_<cid>.json (read-only; sha256 recorded), the lane B lake view (sha-verified), bound through `PopulationSpec` for the
TRAIN-only scaler (mu, sigma) and the intercept (mean of Y_h over the scored TRAIN origins). The validation target is rebuilt as M07 defines it,
Y_h(t) = sum_{k=1..h} z(log_return_1[t+k]) with log_return_1 derived causally from CLOSE; the rebuild is CHECKED against the evidence: the
reconstructed model MAE and zero-return MAE per horizon must agree with `per_horizon[h].model_MAE / naive_MAE` to 1e-6 or the run is refused.

Per horizon h and loss in {abs, sq}: d_i = loss(model_i) - loss(control_i), the control being the zero-return forecast (log-return 0, i.e. z = -h*mu/sigma)
or the intercept-only forecast (TRAIN mean of Y_h). Negative = model better. Inference on mean(d): circular moving-block bootstrap (B resamples,
seeded) with block length L = max(tau of Y_h, tau of the model prediction, tau of d, h + 6) (tau = 1 + 2 sum rho_k to the first rho_k < 0.05, cap n/20; the
tau of d is added to the rule's L because the paired series has its own persistence) -> SE and a 95 % percentile interval; `excludes_zero` and
its side are reported. The equal-weight combination across horizons averages the h1..h4 d_i row by row (L = max over horizons).
Controls that must NOT favour the model: the predictions rolled by 5000 rows (time-scrambled) against the zero-return control.
Never a financial claim; margins are read against the naive MAE, and a result that excludes zero is a statement about this validation
sample under this interval, not confirmation (the S2 reserve and the test are not read).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import EURUSD_LAKE_A_S1_SPEC, bind_population  # noqa: E402
from c2_battery_calibration import integrated_autocorr_length  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path, horizons):
    rows, preds = [], {h: [] for h in horizons}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(int(r["row_id"].split(":row")[1].split(":")[0]))
            for h in horizons:
                preds[h].append(float(r[f"y_hat_z_h{h}"]))
    return np.asarray(rows), {h: np.asarray(v) for h, v in preds.items()}


def validation_targets(frame_lr1, mu, sigma, times, origins, h, bar_seconds):
    """Y_h at the given origin rows, from rows t+1..t+h; refused if the label span is not exactly h regular bars."""
    z = (frame_lr1 - mu) / sigma
    csum = np.concatenate([[0.0], np.cumsum(z)])
    if np.any(times[origins + h] - times[origins] != h * bar_seconds):
        raise ValueError("IRREGULAR_LABEL_SPAN_IN_EVIDENCE_ROWS")
    return csum[origins + h + 1] - csum[origins + 1]


def block_bootstrap_mean(d, L, b, rng):
    """Circular moving-block bootstrap of mean(d): (se, 2.5 %, 97.5 % percentiles of the resampled mean)."""
    n = len(d)
    L = max(1, min(int(L), n // 2))
    nblocks = int(np.ceil(n / L))
    step = max(1, 5_000_000 // n)
    means = []
    for lo in range(0, b, step):
        k = min(step, b - lo)
        starts = rng.integers(0, n, size=(k, nblocks))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(k, -1)[:, :n] % n
        means.append(d[idx].mean(axis=1))
    m = np.concatenate(means)
    return float(np.std(m, ddof=1)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_row(d, L, b, rng, naive_mae=None):
    se, lo, hi = block_bootstrap_mean(d, L, b, rng)
    mean = float(d.mean())
    out = {"mean_diff": mean, "se": se, "ci95": [lo, hi], "block_length": int(L), "n": int(len(d)),
           "excludes_zero": bool(lo > 0 or hi < 0), "side": "MODEL_BETTER" if hi < 0 else ("MODEL_WORSE" if lo > 0 else "INCLUDES_ZERO"),
           "interval_wald": [mean - 1.96 * se, mean + 1.96 * se]}
    if naive_mae:
        out["mean_diff_over_naive_mae"] = mean / naive_mae
    return out


def analyse(pop, rows, preds, evidence, *, boot=2000, seed=20261001, check_tol=1e-6):
    times, lr1 = pop.times, pop.frame["log_return_1"].to_numpy(dtype=np.float64)
    horizons = sorted(preds)
    mu, sigma = pop.mu, pop.sigma
    out = {"horizons": {}, "checks": {}}
    ds = {"abs_zero": {}, "sq_zero": {}, "abs_int": {}, "sq_int": {}}
    y_by, yhat_by, L_by = {}, {}, {}
    for h in horizons:
        y = validation_targets(lr1, mu, sigma, times, rows, h, pop.bar_seconds)
        yhat = preds[h]
        e_zero = (-h * mu / sigma) - y                      # zero log-return forecast in z-units
        tr_y = pop.m07_target(h)[pop.origins]
        intercept = float(np.mean(tr_y[np.isfinite(tr_y)]))   # TRAIN-only intercept
        e_int = intercept - y
        e_m = yhat - y
        ph = next(p for p in evidence["per_horizon"] if int(p["horizon"]) == h)
        chk = {"model_MAE_rebuilt": float(np.mean(np.abs(e_m))), "model_MAE_evidence": float(ph["model_MAE"]),
               "zero_MAE_rebuilt": float(np.mean(np.abs(e_zero))), "naive_MAE_evidence": float(ph["naive_MAE"])}
        chk["agrees"] = bool(abs(chk["model_MAE_rebuilt"] - chk["model_MAE_evidence"]) < check_tol and abs(chk["zero_MAE_rebuilt"] - chk["naive_MAE_evidence"]) < check_tol)
        out["checks"][h] = chk
        if not chk["agrees"]:
            raise ValueError(f"EVIDENCE_NOT_REPRODUCED at h{h}: {chk}")
        ds["abs_zero"][h] = np.abs(e_m) - np.abs(e_zero)
        ds["sq_zero"][h] = e_m ** 2 - e_zero ** 2
        ds["abs_int"][h] = np.abs(e_m) - np.abs(e_int)
        ds["sq_int"][h] = e_m ** 2 - e_int ** 2
        n = len(y)
        L_by[h] = int(max(integrated_autocorr_length(y, n // 20), integrated_autocorr_length(yhat, n // 20), h + 6))
        y_by[h], yhat_by[h] = y, yhat
        out["horizons"][h] = {"intercept_train_mean": intercept, "tau_y": integrated_autocorr_length(y, n // 20), "tau_yhat": integrated_autocorr_length(yhat, n // 20),
                              "naive_zero_MAE": chk["zero_MAE_rebuilt"], "model_MAE": chk["model_MAE_rebuilt"], "intercept_MAE": float(np.mean(np.abs(e_int)))}
    rng = np.random.default_rng(seed)
    for key, per_h in ds.items():
        for h in horizons:
            L = int(max(L_by[h], integrated_autocorr_length(per_h[h], len(per_h[h]) // 20)))
            nm = out["horizons"][h]["naive_zero_MAE"] if key.startswith("abs") and key.endswith("zero") else None
            out["horizons"][h][key] = paired_row(per_h[h], L, boot, rng, nm)
        comb = np.mean([per_h[h] for h in horizons], axis=0)
        Lc = int(max(max(L_by.values()), integrated_autocorr_length(comb, len(comb) // 20)))
        out.setdefault("combined_equal_weight", {})[key] = paired_row(comb, Lc, boot, rng)
    # control: predictions rolled by 5000 rows must not beat the zero-return control
    ctrl = {}
    for h in horizons:
        e_m = np.roll(yhat_by[h], 5000) - y_by[h]
        e_zero = (-h * mu / sigma) - y_by[h]
        d = np.abs(e_m) - np.abs(e_zero)
        ctrl[h] = paired_row(d, L_by[h], boot, rng)
    out["control_time_scrambled_predictions_vs_zero_abs"] = ctrl
    out["control_passes"] = all(c["side"] != "MODEL_BETTER" for c in ctrl.values())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", default="1,2,3,4")
    p.add_argument("--boot", type=int, default=2000)
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, spec=EURUSD_LAKE_A_S1_SPEC)
    hs = [int(v) for v in args.horizons.split(",")]
    rows, preds = load_predictions(args.predictions, hs)
    evidence = json.loads(Path(args.evidence).read_text())
    t0 = time.process_time()
    res = analyse(pop, rows, preds, evidence, boot=args.boot)
    res.update({"schema": "c2_paired_loss_inference.v1", "label": "DEVELOPMENT", "bindings": pop.bindings,
                "inputs": {"predictions_sha256": sha256_file(args.predictions), "evidence_sha256_file": sha256_file(args.evidence),
                           "evidence_candidate": evidence["artifact"]["candidate_cid"], "seed": evidence["cell"]["seed"], "label": evidence["cell"]["label"],
                           "rows": int(len(rows))},
                "cpu_seconds": time.process_time() - t0, "peak_rss_bytes_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    for h in hs:
        r = res["horizons"][h]
        print(f"h{h} model {r['model_MAE']:.6f} zero {r['naive_zero_MAE']:.6f} | abs-vs-zero d={r['abs_zero']['mean_diff']:+.2e} ci={r['abs_zero']['ci95'][0]:+.2e},{r['abs_zero']['ci95'][1]:+.2e} {r['abs_zero']['side']} "
              f"| sq-vs-zero {r['sq_zero']['side']} | abs-vs-int {r['abs_int']['side']} | sq-vs-int {r['sq_int']['side']}")
    print("combined", {k: v["side"] for k, v in res["combined_equal_weight"].items()}, "control_passes", res["control_passes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
