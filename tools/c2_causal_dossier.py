"""Lane C2 deliverable 2: a `causal_dossier.v1` document per (feature, horizon) for the top features of the ETH 4h
TRAIN split, with the identifying assumptions written down so they can be attacked, an intervention on the feature's
value estimated by a declared estimator, a same-episode counterfactual under a declared additive SCM, and the
controls that MUST fail actually run.

Estimand (rung 2): `E[Y_h | do(X_j = q75)] - E[Y_h | do(X_j = q25)]` on the TRAIN population, under the declared
partially linear model `Y_h = theta * X_j + g(W) + U`, with W the other admissible features at the same bar minus the
redundancy cluster of X_j (|corr| > 0.95 on TRAIN, named in `excluded_from_adjustment`).
Estimator: cross-fitted partially linear DML (Chernozhukov et al. 2018) with ridge nuisances, cross-fitting over the
five contiguous time blocks with the same embargo as deliverable 1, HAC (Newey-West, lag h + 6) interval on theta.
Support screen: residual variance share of X_j after adjustment on W (the provider's WP22 continuous-dose screen),
declared floor 0.05.

Rung 3 (same-episode counterfactual): abduction `u_e = y_e - f(x_e, w_e)` with the out-of-fold f, action
`X_j := x0` (TRAIN median), prediction `y_cf = f(x0, w_e) + u_e`; under the PLM `delta_e = theta * (x_e - x0)` exactly,
and the model-based `f(x0, w_e)` is reported beside so the two are never confused.

Controls that MUST fail (and are verified to fail):
* scrambled label: the blocks of Y are permuted and circularly shifted; theta's interval must contain zero and the
  incremental utility must not beat the zero-return naive;
* future-shifted feature: X_j(t + h) stands in for X_j(t); the probe must fire (interval far from zero, single-feature
  MAE well below the zero-return naive);
* noise treatment: an independent standard normal; interval must contain zero.

Identification status: every dossier is `NOT_IDENTIFIED` by construction, and says why: the view's timestamp
semantics are undeclared (FEATURE_DAG.v3), a derived feature is a deterministic function of past prices (no physical
intervention sets an RSI without moving prices), and the latent market state is unmeasured. The estimate is reported
as a DEVELOPMENT number conditional on the declared assumptions, never as an identified effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import bind_population, blocked_splits, naive_predictions  # noqa: E402
from c2_feature_contribution import _standardize, choose_alpha, ridge_fit, ridge_predict  # noqa: E402

RESIDUAL_SHARE_FLOOR = 0.05
REDUNDANCY_CORR = 0.95
SCHEMA_REL = Path("docs/contracts/causal_dossier.v1.schema.json")
DAG_V3 = {"artifact": "financial-data features/census/FEATURE_DAG.v3.json @ ac2e8072",
          "dag_sha256": "e15017cee5b0101352b527bee4253c6a9a047920af7e01534b82f2b0d047d8f5"}
CONTRACTS_DOC_SHA256 = "bfaf2cf63814a0be5756f648f3f5c949b97d96b1f0d0c29f674bc0d36a371406"
LAKE_PARENT_APPEARANCE = "app_00bd88770beb60e9ac4b30f4"


def hac_variance(score: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance of a scalar score series (Bartlett kernel)."""
    n = len(score)
    s = score - score.mean()
    v = float(s @ s) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * float(s[k:] @ s[:-k]) / n
    return v


def crossfit_residuals(x_t, w, y, splits, alpha_rule=choose_alpha):
    """Out-of-fold residuals of the treatment and the outcome on W over the time blocks (embargoed)."""
    n = len(y)
    r_x, r_y, g_hat = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    m_hat = np.full(n, np.nan)
    for sp in splits:
        fit, ev = sp["fit"], sp["eval"]
        wf, we = _standardize(w[fit], w[ev])
        a_y = alpha_rule(w[fit], y[fit])
        a_x = alpha_rule(w[fit], x_t[fit])
        my = ridge_predict(ridge_fit(wf, y[fit], a_y), we)
        mx = ridge_predict(ridge_fit(wf, x_t[fit], a_x), we)
        r_y[ev] = y[ev] - my
        r_x[ev] = x_t[ev] - mx
        g_hat[ev] = my
        m_hat[ev] = mx
    ok = np.isfinite(r_x) & np.isfinite(r_y)
    return r_x, r_y, g_hat, m_hat, ok


def plm_theta(r_x, r_y, ok, lag):
    rx, ry = r_x[ok], r_y[ok]
    denom = float(rx @ rx)
    theta = float(rx @ ry) / denom
    score = rx * (ry - theta * rx)
    var = hac_variance(score, lag) * len(rx) / denom ** 2
    se = float(np.sqrt(max(var, 0.0)))
    return theta, se


def _ridge_single_feature_mae(x_col, y, splits):
    """Held-out MAE of a one-feature ridge against the zero naive, in z-units, averaged over the blocks."""
    maes, naive = [], []
    for sp in splits:
        fit, ev = sp["fit"], sp["eval"]
        xf, xe = _standardize(x_col[fit, None], x_col[ev, None])
        pred = ridge_predict(ridge_fit(xf, y[fit], 1.0), xe)
        maes.append(float(np.mean(np.abs(pred - y[ev]))))
        naive.append(float(np.mean(np.abs(y[ev]))))
    return float(np.mean(maes)), float(np.mean(naive))


def study(pop, feature, h, *, seed=20261001, splits=None):
    """One (feature, horizon) study: rung 1 association, rung 2 PLM-DML effect and screens, rung 3 counterfactual,
    and the three controls. Returns the numbers; `dossier()` wraps them in the contract."""
    rng = np.random.default_rng(seed)
    y_all = pop.m07_target(h)
    y_raw_all = pop.raw_log_return(h)
    xall = pop.feature_matrix_train()
    j = pop.features.index(feature)
    rows = pop.origins[np.isfinite(y_all[pop.origins])]
    # the redundancy cluster: |corr| > 0.95 with the treatment on TRAIN rows
    xs = xall[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(xs[:, j], xs[:, k])[0, 1] if k != j else 1.0 for k in range(xs.shape[1])])
    redundant = [pop.features[k] for k in range(len(pop.features)) if k != j and abs(corr[k]) > REDUNDANCY_CORR]
    w_idx = [k for k in range(len(pop.features)) if k != j and pop.features[k] not in redundant]
    splits = splits or blocked_splits(rows, h)
    # restrict the splits to the finite rows (origins carry finite labels here already)
    x_t, w, y = xall[:, j], xall[:, w_idx], y_all
    lag = h + 6

    def effect(x_treat, y_out):
        r_x, r_y, g_hat, m_hat, ok = crossfit_residuals(x_treat, w, y_out, splits)
        theta, se = plm_theta(r_x, r_y, ok, lag)
        share = float(np.var(r_x[ok]) / np.var(x_treat[ok])) if np.var(x_treat[ok]) > 0 else 0.0
        return dict(theta=theta, se=se, residual_variance_share=share, n=int(ok.sum()), r_x=r_x, r_y=r_y,
                    g_hat=g_hat, m_hat=m_hat, ok=ok)

    main = effect(x_t, y)
    q = np.quantile(x_t[rows], [0.0, 0.25, 0.5, 0.75, 1.0])
    x0, x1 = float(q[1]), float(q[3])
    ok = main["ok"]
    # rung 1: association on the same rows
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(x_t[ok], y[ok])
    sr = spearmanr(x_t[ok], y[ok])
    per_block_sign = []
    for sp in splits:
        ev = sp["eval"][np.isfinite(y[sp["eval"]])]
        per_block_sign.append(float(np.sign(spearmanr(x_t[ev], y[ev])[0])))
    sign_stable = bool(len(set(per_block_sign)) == 1)
    # rung 3 under the PLM: f(x, w) = theta * x + (g_hat - theta * m_hat), out of fold
    theta = main["theta"]
    f_fact = theta * x_t[ok] + (main["g_hat"][ok] - theta * main["m_hat"][ok])
    u = y[ok] - f_fact
    xmed = float(q[2])
    f_cf = theta * xmed + (main["g_hat"][ok] - theta * main["m_hat"][ok])
    y_cf = f_cf + u
    delta = y[ok] - y_cf
    fit_params = {"theta": theta, "x0": xmed, "feature": feature, "horizon_bars": h,
                  "nuisance": "ridge on W, alpha by last-20% MAE, cross-fitted over blocks5"}
    fit_digest = hashlib.sha256(json.dumps(fit_params, sort_keys=True).encode()).hexdigest()
    # controls that MUST fail
    y_scr = y.copy()
    fin = np.where(np.isfinite(y))[0]
    perm_blocks = np.array_split(fin, 5)
    order = rng.permutation(5)
    scr_vals = np.concatenate([y[perm_blocks[i]] for i in order])
    scr_vals = np.roll(scr_vals, int(rng.integers(500, len(scr_vals) - 500)))
    y_scr[fin] = scr_vals
    scr = effect(x_t, y_scr)
    scr_mae, scr_naive = _ridge_single_feature_mae(x_t, y_scr, splits)
    x_fut = np.full_like(x_t, np.nan)
    x_fut[: len(x_t) - h] = x_t[h:]
    fut_ok_rows = np.isfinite(x_fut)
    fut_splits = [{"name": sp["name"], "fit": sp["fit"][fut_ok_rows[sp["fit"]]], "eval": sp["eval"][fut_ok_rows[sp["eval"]]]}
                  for sp in splits]
    r_x, r_y, _, _, okf = crossfit_residuals(x_fut, w, y, fut_splits)
    th_f, se_f = plm_theta(r_x, r_y, okf, lag)
    fut_mae, fut_naive = _ridge_single_feature_mae(np.nan_to_num(x_fut), y, fut_splits)
    base_mae, base_naive = _ridge_single_feature_mae(x_t, y, splits)
    noise = effect(rng.standard_normal(len(x_t)), y)
    z = 1.96
    controls = {
        "scrambled_label": {"theta": scr["theta"], "interval": [scr["theta"] - z * scr["se"], scr["theta"] + z * scr["se"]],
                            "single_feature_mae_z": scr_mae, "naive_zero_mae_z": scr_naive,
                            "failed_as_required": bool(abs(scr["theta"]) <= z * scr["se"] and scr_mae >= scr_naive * 0.999)},
        "future_shifted_feature": {"theta": th_f, "interval": [th_f - z * se_f, th_f + z * se_f], "single_feature_mae_z": fut_mae,
                                   "naive_zero_mae_z": fut_naive, "causal_single_feature_mae_z": base_mae,
                                   "failed_as_required": bool(abs(th_f) > z * se_f and fut_mae < 0.9 * fut_naive)},
        "noise_treatment": {"theta": noise["theta"], "interval": [noise["theta"] - z * noise["se"], noise["theta"] + z * noise["se"]],
                            "failed_as_required": bool(abs(noise["theta"]) <= z * noise["se"])},
    }
    reasons = ["TIMESTAMP_SEMANTICS_UNDECLARED", "TREATMENT_IS_A_DETERMINISTIC_FUNCTION_OF_PAST_PRICES_NO_PHYSICAL_INTERVENTION",
               "LATENT_MARKET_STATE_UNMEASURED", "ESTIMATE_REPORTED_UNDER_DECLARED_ASSUMPTIONS_DEVELOPMENT_ONLY"]
    support_state = "SUPPORTED"
    if main["residual_variance_share"] < RESIDUAL_SHARE_FLOOR:
        support_state = "TREATMENT_PREDICTED_BY_CONTROLS"
        reasons.insert(0, "TREATMENT_PREDICTED_BY_CONTROLS")
    return {
        "feature": feature, "horizon_bars": h, "n": int(ok.sum()), "rows": [int(rows.min()), int(rows.max())],
        "redundant": redundant, "adjustment": [pop.features[k] for k in w_idx], "theta": theta, "se": main["se"],
        "theta_interval": [theta - z * main["se"], theta + z * main["se"]],
        "contrast": [x0, x1], "effect_q25_to_q75_z": theta * (x1 - x0), "effect_q25_to_q75_log_return": pop.sigma * theta * (x1 - x0),
        "effect_interval_z": [(theta - z * main["se"]) * (x1 - x0), (theta + z * main["se"]) * (x1 - x0)],
        "residual_variance_share": main["residual_variance_share"], "support_state": support_state,
        "n_per_side": [int((x_t[rows] <= x0).sum()), int((x_t[rows] >= x1).sum())],
        "dose_support": {"min": float(q[0]), "max": float(q[4]), "q25": x0, "q50": xmed, "q75": x1},
        "rung1": {"pearson": float(pr[0]), "pearson_p": float(pr[1]), "spearman": float(sr[0]), "spearman_p": float(sr[1]),
                  "sign_by_block": per_block_sign, "signed_direction_stable": sign_stable},
        "rung3": {"x0": xmed, "u_mean": float(u.mean()), "u_std": float(u.std()), "factual_mean": float(y[ok].mean()),
                  "counterfactual_mean": float(y_cf.mean()), "delta_mean": float(delta.mean()),
                  "delta_q": [float(v) for v in np.quantile(delta, [0.05, 0.5, 0.95])], "model_based_mean": float(f_cf.mean()),
                  "fit_digest": fit_digest, "fit_params": fit_params},
        "controls": controls, "reasons": reasons, "lag": lag, "mu": pop.mu, "sigma": pop.sigma,
    }


def dossier(pop, s, *, revision, produced_at=None):
    h = s["horizon_bars"]
    minutes = 240 * h
    head, target = ("short", "Y_s") if h == 1 else ("long", "Y_l")
    first_ts = datetime.fromtimestamp(int(pop.times[s["rows"][0]]), tz=timezone.utc)
    last_ts = datetime.fromtimestamp(int(pop.times[s["rows"][1]]), tz=timezone.utc)
    emit_from = datetime.fromtimestamp(int(pop.times[pop.train_end - 1]) + h * 14400, tz=timezone.utc)
    f = s["feature"]
    controls_ok = all(c["failed_as_required"] for c in s["controls"].values())
    return {
        "schema": "causal_dossier.v1",
        "dossier_id": f"c2-eth4h-{f.lower().replace('__', '-').replace('_', '-')}-h{h}-development",
        "produced_at": produced_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": {"repository": "causal-inference", "revision": revision, "module": "predictor tools/c2_causal_dossier.py (lane C2; the PS3-C module causal_inference_provider.ps3c is not yet implemented, FS10-FS12 red)"},
        "subject": {"kind": "EPISODE_POPULATION",
                    "population": f"ETH 4h TRAIN bar closes, M07 scored origins rows [{s['rows'][0]},{s['rows'][1]}] with a label at exactly {h} regular bars; DEVELOPMENT",
                    "event_type": "bar_close", "asset": "ETHUSDT", "head": head, "target": target, "horizon_minutes": minutes},
        "data_manifest": {
            "asset_appearance": {"state": "CONTRACTED_MODEL_READY_VIEW", "dataset_id": pop.bindings["view"]["dataset_id"],
                                 "entity": "ethusdt", "resource_sha256": pop.bindings["view"]["sha256"],
                                 "contract": "predictor examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.manifest.json @ 14a1077f",
                                 "train_rows": list(pop.train_rows), "frequency": "4h",
                                 "period": [first_ts.isoformat(), last_ts.isoformat()],
                                 "contracts_document_sha256": CONTRACTS_DOC_SHA256,
                                 "lake_parent_appearance_id": LAKE_PARENT_APPEARANCE, "availability_class": "DEVELOPMENT"},
            "sources": [
                {"role": "bars", "resource": f"predictor {pop.bindings['view']['path']} @ {pop.bindings['view']['commit'][:8]}",
                 "sha256": pop.bindings["view"]["sha256"], "rows": pop.bindings["view"]["rows"], "governed": False},
                {"role": "covariates", "resource": "feature-eng docs/feature_metrics/laneB/SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json @ b90c4b3",
                 "sha256": pop.bindings["manifest"]["file_sha256"], "rows": pop.bindings["manifest"]["features"], "governed": False},
                {"role": "projections", "resource": "predictor docs/audits/evidence/lane_f2_eth_20261001/SPLIT_eth4h_l24_h6_v1.json @ 13ef175f (M07 split)",
                 "sha256": pop.bindings["split"]["file_sha256"], "rows": pop.bindings["split"]["train_windows"], "governed": False}],
            "publication_clock": "ASSUMED_SCHEDULED_PUBLICATION", "consensus_clock": "NONE", "expectation_kind": "NONE",
            "n_episodes": s["n"], "exclusions": {"NO_BAR_AT_EXACT_ELAPSED_TIME_OR_IRREGULAR_WINDOW": int(pop.gap_excluded_windows)},
            "train_folds": ["blocks5:B1", "blocks5:B2", "blocks5:B3", "blocks5:B4", "blocks5:B5"]},
        "treatment": {"name": f, "definition": f"stored column `{f}` of the view; producer financial-data stage22_trading_features_worker.py @ 19fe375a (sha 7495a0d9), value at the bar close t",
                      "kind": "CONTINUOUS", "standardization": "raw feature units; the contrast is the TRAIN interquartile range q25 -> q75",
                      "dose_support": {"min": s["dose_support"]["min"], "max": s["dose_support"]["max"], "n": s["n"],
                                       "quantiles": {"q25": s["dose_support"]["q25"], "q50": s["dose_support"]["q50"], "q75": s["dose_support"]["q75"]}},
                      "sequential_treatment_policy": "SEPARATED_EPISODES_ONLY"},
        "rung1": {"state": "ASSOCIATION_REPORTED", "estimators": ["pearson", "spearman", "ridge incremental utility (deliverable 1)"],
                  "effective_n": s["n"], "conditioning_set": ["W = the other admissible features at t minus the redundancy cluster"],
                  "multiplicity": {"family": "83 features x 7 horizons", "n_tests": 581, "permutations": 0, "p_floor": 0.0, "correction": "none (reported raw; selection reads ranks, not p)"},
                  "evidence": [{"measure": "pearson_r", "value": s["rung1"]["pearson"], "n": s["n"], "p": min(1.0, s["rung1"]["pearson_p"]), "q": None, "regime": None, "signed_direction_stable": s["rung1"]["signed_direction_stable"]},
                               {"measure": "spearman_rho", "value": s["rung1"]["spearman"], "n": s["n"], "p": min(1.0, s["rung1"]["spearman_p"]), "q": None, "regime": None, "signed_direction_stable": s["rung1"]["signed_direction_stable"]}]},
        "rung2": {"state": "NOT_IDENTIFIED", "reasons": s["reasons"],
                  "estimand": f"E[Y_h | do({f} = q75)] - E[Y_h | do({f} = q25)] on the TRAIN population, Y_h in M07 z-units, under the declared partially linear model",
                  "contrast": s["contrast"],
                  "dag": {"nodes": ["P_hist", "U", f, "W", "Y_h"], "edges": [["P_hist", f], ["P_hist", "W"], ["U", "P_hist"], ["U", "Y_h"], [f, "Y_h"], ["W", "Y_h"]]},
                  "adjustment": s["adjustment"],
                  "excluded_from_adjustment": [f"{r} (REDUNDANT_TRANSFORM_OF_TREATMENT |corr|>{REDUNDANCY_CORR})" for r in s["redundant"]] + ["any column at rows > t (future)", "Y at shorter horizons (descendant)"],
                  "assumptions_declared": {"lineage_dag_v3_names_the_producer": True, "partially_linear_effect": True, "additive_noise": True,
                                           "no_unmeasured_confounding_given_W": False, "timestamp_is_bar_close": False,
                                           "physical_intervention_on_a_derived_feature_exists": False},
                  "estimator": {"name": "partially_linear_DML_crossfit_ridge_HAC", "library": "numpy+scikit-learn", "version": __import__("sklearn").__version__, "revision": revision},
                  "support": {"state": s["support_state"], "n_per_side": s["n_per_side"], "residual_variance_share": s["residual_variance_share"]},
                  "placebo": {"state": "PASSED" if controls_ok else "FAILED",
                              "tests": [{"name": name, "verdict": ("FAILED_AS_REQUIRED" if c["failed_as_required"] else "DID_NOT_FAIL_PROBE_SUSPECT"), "n": s["n"]}
                                        for name, c in s["controls"].items()]},
                  "sensitivity": {"theta_per_unit_z": s["theta"], "theta_se_hac": s["se"], "hac_lag": s["lag"],
                                  "effect_q25_to_q75_z": s["effect_q25_to_q75_z"], "effect_q25_to_q75_log_return": s["effect_q25_to_q75_log_return"],
                                  "effect_interval_z_low": s["effect_interval_z"][0], "effect_interval_z_high": s["effect_interval_z"][1],
                                  "scrambled_label_theta": s["controls"]["scrambled_label"]["theta"],
                                  "future_shifted_theta": s["controls"]["future_shifted_feature"]["theta"],
                                  "future_shifted_single_feature_mae_z": s["controls"]["future_shifted_feature"]["single_feature_mae_z"],
                                  "causal_single_feature_mae_z": s["controls"]["future_shifted_feature"]["causal_single_feature_mae_z"],
                                  "naive_zero_mae_z": s["controls"]["future_shifted_feature"]["naive_zero_mae_z"],
                                  "noise_treatment_theta": s["controls"]["noise_treatment"]["theta"], "label": "DEVELOPMENT"},
                  "estimate": None},
        "rung3": {"state": "NOT_IDENTIFIED", "label": "MODEL_BASED_COUNTERFACTUAL", "reasons": ["INHERITED_FROM_RUNG2"] + s["reasons"][:3],
                  "scm": {"order": ["P_hist", "W", f, "Y_h"], "equations": {"Y_h": f"theta * {f} + g(W) + U", f: "deterministic producer of P_hist (not re-estimated)"},
                          "noise": "ADDITIVE", "invertible": True, "fit_digest": s["rung3"]["fit_digest"], "library": "numpy (PLM, out-of-fold)",
                          "alternatives_considered": ["non-linear g (HGB) not fitted here", "mediators none declared"]},
                  "abduction": {"u_mean": s["rung3"]["u_mean"], "u_std": s["rung3"]["u_std"], "n": s["n"]},
                  "action": {f: s["rung3"]["x0"]},
                  "prediction": {"factual": s["rung3"]["factual_mean"], "counterfactual": s["rung3"]["counterfactual_mean"], "delta": s["rung3"]["delta_mean"],
                                 "model_based": s["rung3"]["model_based_mean"], "propagated_nodes": [], "barrier_reread": None,
                                 "uncertainty": {"delta_q05": s["rung3"]["delta_q"][0], "delta_q50": s["rung3"]["delta_q"][1], "delta_q95": s["rung3"]["delta_q"][2]}},
                  "sensitivity": {"delta_equals_theta_times_shift_under_PLM": "exact", "units": "M07 z-units"}},
        "emission": {"operational_use": "RETROSPECTIVE_ONLY", "emittable_from": {"delta": emit_from.isoformat(), "u": emit_from.isoformat()}},
        "selection": {"causal_evidence_level": "ASSOCIATION", "cf_eligible": False, "reason_code": "RUNG2_NOT_IDENTIFIED_DEVELOPMENT_ESTIMATE_ONLY"},
        "limitations": [
            "DEVELOPMENT: the view is a git-pinned local file (no governed availability contract); nothing here is confirmatory",
            "the timestamp semantics of the view are undeclared (FEATURE_DAG.v3); bar-close is assumed, not certified",
            "a derived feature cannot be set without moving prices: do(X_j) is an intervention on the model's input, reported here because the orders ask for it; it is not an intervention on the market",
            "the partially linear model is declared, not tested against a non-linear alternative here",
            "all rows are TRAIN rows [0,13699) of the declared calendar split; validation 2024 and test 2025 were never read",
            f"the horizon {h} bars = {4 * h} h " + ("is the spec's Y_s grid (4 h)" if h == 1 else "lies between the spec's Y_s (1-6 h) and Y_l (24-144 h) grids" if h < 6 else "is the Y_l grid"),
        ],
    }


def validate(doc, schema_path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["JSONSCHEMA_NOT_INSTALLED_VALIDATION_SKIPPED"]
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return [e.message for e in Draft202012Validator(schema).iter_errors(doc)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features", default=None, help="comma list; default = top-20 of --contribution-summary")
    p.add_argument("--contribution-summary", default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--horizons", default="1,2,3,4,5,6")
    p.add_argument("--revision", required=True, help="git revision of the producing tree")
    p.add_argument("--schema", default=None)
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, args.split)
    if args.features:
        feats = args.features.split(",")
    else:
        summ = json.loads(Path(args.contribution_summary).read_text(encoding="utf-8"))
        feats = summ["ranking_by_stable_incremental_utility"][: args.top]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, t0 = [], time.process_time()
    for f in feats:
        for h in [int(v) for v in args.horizons.split(",")]:
            s = study(pop, f, h)
            doc = dossier(pop, s, revision=args.revision)
            errors = validate(doc, args.schema) if args.schema else ["NOT_VALIDATED"]
            path = out / f"{doc['dossier_id']}.json"
            path.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
            index.append({"feature": f, "horizon_bars": h, "file": path.name, "n": s["n"], "theta_z": s["theta"], "theta_se": s["se"],
                          "effect_q25_q75_z": s["effect_q25_to_q75_z"], "effect_q25_q75_log_return": s["effect_q25_to_q75_log_return"],
                          "residual_variance_share": s["residual_variance_share"], "support_state": s["support_state"],
                          "controls_failed_as_required": all(c["failed_as_required"] for c in s["controls"].values()),
                          "scrambled_theta": s["controls"]["scrambled_label"]["theta"],
                          "future_shifted_mae_z": s["controls"]["future_shifted_feature"]["single_feature_mae_z"],
                          "causal_mae_z": s["controls"]["future_shifted_feature"]["causal_single_feature_mae_z"],
                          "naive_zero_mae_z": s["controls"]["future_shifted_feature"]["naive_zero_mae_z"],
                          "rung2_state": doc["rung2"]["state"], "schema_errors": errors, "redundant": s["redundant"]})
            print(f"{f} h{h} n={s['n']} theta={s['theta']:.4f}+-{s['se']:.4f} share={s['residual_variance_share']:.3f} "
                  f"{s['support_state']} controls_ok={index[-1]['controls_failed_as_required']} errors={len(errors)}", flush=True)
    (out / "DOSSIER_INDEX.json").write_text(json.dumps({"label": "DEVELOPMENT", "bindings": pop.bindings, "cpu_seconds": time.process_time() - t0,
                                                        "dossiers": index}, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
