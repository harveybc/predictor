"""Lane C2 deliverable 1: per-feature predictive contribution on the ETH 4h TRAIN split, held out within TRAIN and
blocked by time.

For every horizon h and every held-out split (five contiguous time blocks with an embargo, and lane B's three
expanding inner folds for comparability) the tool fits a ridge on the standardized 83 features (alpha chosen on the
last 20 % of the fit rows from a fixed grid, the same for every arm; lane B's fixed alpha = 1 is reported beside) and
measures, on the identical held-out rows and in two units (M07 z-units and raw log return):

* `full`                 : all 83 features;
* `without:<f>`          : the 82 others — the conditional incremental utility of f is MAE(without f) - MAE(full);
* `only:<f>`             : f alone;
* `naive_zero`, `naive_last_return`, `naive_seasonal_24h`, `naive_train_mean`: reference rows on the same rows;
* `hgb_full`, `hgb_perm:<f>`: a gradient-boosted tree on the same rows with a circular-shift permutation of f on
  the held-out block (non-linear check of the same question; optional, `--skip-hgb`).

Stability: for each feature, the per-block incremental utilities carry a sign-agreement fraction and a rank; the
per-horizon mean pairwise Spearman of the block rank vectors is the rank agreement.

CPU only. One process. The output is a long CSV of records plus a summary JSON; every record carries the population
digest, the split, the rows it was fitted and evaluated on, and the horizon.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import (bind_population, blocked_splits, lane_b_splits, naive_predictions,  # noqa: E402
                               LANE_B_HORIZON_BARS)

ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
LANE_B_ALPHA = 1.0


def _standardize(xfit, xeval):
    mean = xfit.mean(axis=0)
    scale = xfit.std(axis=0)
    scale[scale == 0] = 1.0
    return (xfit - mean) / scale, (xeval - mean) / scale


def ridge_fit(x, y, alpha):
    """Closed-form ridge with an unpenalized intercept on already standardized x."""
    n, p = x.shape
    ym = y.mean()
    a = x.T @ x + alpha * np.eye(p)
    beta = np.linalg.solve(a, x.T @ (y - ym))
    return beta, ym


def ridge_predict(model, x):
    beta, ym = model
    return x @ beta + ym


def choose_alpha(xfit, yfit, grid=ALPHA_GRID, tail=0.2):
    """Alpha by MAE on the last `tail` of the fit rows (time-ordered), the same rule for every arm."""
    cut = int(len(yfit) * (1 - tail))
    xa, xb = _standardize(xfit[:cut], xfit[cut:])
    best, best_mae = None, np.inf
    for alpha in grid:
        mae = np.mean(np.abs(ridge_predict(ridge_fit(xa, yfit[:cut], alpha), xb) - yfit[cut:]))
        if mae < best_mae:
            best, best_mae = alpha, mae
    return float(best)


def losses(pred, y):
    err = pred - y
    return float(np.mean(np.abs(err))), float(np.mean(err ** 2))


def circular_shift_permutation(x_col, rng, min_shift=50):
    n = len(x_col)
    k = int(rng.integers(min_shift, max(min_shift + 1, n - min_shift)))
    return np.roll(x_col, k)


def run(pop, horizons, out_dir, *, protocols=("blocks5", "laneB"), skip_hgb=False, hgb_iters=150, max_features=None,
        seed=20261001, heartbeat=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pop.features if max_features is None else pop.features[:max_features]
    xall = pop.frame[features].to_numpy(dtype=np.float64)
    y_raw_by_h = {h: pop.raw_log_return(h) for h in horizons}
    y_z_by_h = {h: pop.m07_target(h) for h in horizons}
    rng = np.random.default_rng(seed)
    records = []
    t0 = time.process_time()
    wall0 = time.time()

    def rec(**kw):
        base = {"population": pop.bindings["view"]["dataset_id"], "view_sha256": pop.bindings["view"]["sha256"],
                "split": "TRAIN", "train_rows": f"[{pop.train_rows[0]},{pop.train_rows[1]})"}
        base.update(kw)
        records.append(base)

    hgb_cls = None
    if not skip_hgb:
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgb_cls = HistGradientBoostingRegressor

    for h in horizons:
        y_z, y_raw = y_z_by_h[h], y_raw_by_h[h]
        split_sets = []
        if "blocks5" in protocols:
            split_sets += blocked_splits(pop.origins, h)
        if "laneB" in protocols:
            split_sets += lane_b_splits(pop.origins, h, pop.train_end)
        for sp in split_sets:
            fit = sp["fit"][np.isfinite(y_z[sp["fit"]])]
            ev = sp["eval"][np.isfinite(y_z[sp["eval"]])]
            if len(fit) < 100 or len(ev) < 50:
                continue
            common = dict(protocol=sp["protocol"], block=sp["name"], horizon_bars=h, horizon_hours=4 * h,
                          n_fit=int(len(fit)), n_eval=int(len(ev)), fit_rows=f"[{int(fit.min())},{int(fit.max())}]",
                          eval_rows=f"[{sp['eval_rows'][0]},{sp['eval_rows'][1]}]", embargo_rows=int(sp["embargo"]))
            yf, ye = y_z[fit], y_z[ev]
            ye_raw = y_raw[ev]
            sigma = pop.sigma

            def emit(arm, pred_z, alpha=None, n_features=None, model="ridge"):
                mae_z, mse_z = losses(pred_z, ye)
                pred_raw = pop.z_to_log_return(pred_z, h)
                mae_r, mse_r = losses(pred_raw, ye_raw)
                rec(arm=arm, model=model, alpha=alpha, n_features=n_features, mae_z=mae_z, mse_z=mse_z,
                    mae_log_return=mae_r, mse_log_return=mse_r, **common)

            # reference rows (raw log-return predictions converted to z-units for the z columns)
            naives = naive_predictions(pop, y_raw_by_h, h, ev)
            for name, pred_raw in naives.items():
                emit(name, (pred_raw - h * pop.mu) / sigma, model="naive")
            emit("naive_train_mean", np.full(len(ev), yf.mean()), model="naive")

            xf, xe = xall[fit], xall[ev]
            xfs, xes = _standardize(xf, xe)
            alpha = choose_alpha(xf, yf)
            for a, tag in ((alpha, "selected"), (LANE_B_ALPHA, "laneB_alpha1")):
                emit(f"full|{tag}", ridge_predict(ridge_fit(xfs, yf, a), xes), alpha=a, n_features=len(features))
            for j, f in enumerate(features):
                keep = np.ones(len(features), dtype=bool)
                keep[j] = False
                emit(f"without:{f}", ridge_predict(ridge_fit(xfs[:, keep], yf, alpha), xes[:, keep]), alpha=alpha,
                     n_features=len(features) - 1)
                emit(f"only:{f}", ridge_predict(ridge_fit(xfs[:, [j]], yf, alpha), xes[:, [j]]), alpha=alpha,
                     n_features=1)
            if hgb_cls is not None:
                model = hgb_cls(max_iter=hgb_iters, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                early_stopping=False, random_state=seed)
                model.fit(xfs, yf)
                base_pred = model.predict(xes)
                emit("hgb_full", base_pred, n_features=len(features), model="hgb")
                for j, f in enumerate(features):
                    xp = xes.copy()
                    xp[:, j] = circular_shift_permutation(xes[:, j], rng)
                    emit(f"hgb_perm:{f}", model.predict(xp), n_features=len(features), model="hgb")
            if heartbeat:
                heartbeat(f"h={h} {sp['protocol']}/{sp['name']} n_fit={len(fit)} n_eval={len(ev)} alpha={alpha} "
                          f"cpu={time.process_time() - t0:.0f}s wall={time.time() - wall0:.0f}s")

    import pandas as pd
    table = pd.DataFrame(records)
    table.to_csv(out_dir / "contribution_records.csv", index=False)
    summary = summarize(table, features)
    summary["bindings"] = pop.bindings
    summary["cpu_seconds"] = time.process_time() - t0
    summary["wall_seconds"] = time.time() - wall0
    summary["peak_rss_bytes_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summary["label"] = "DEVELOPMENT"
    (out_dir / "contribution_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    return table, summary


def summarize(table, features):
    """Per feature x horizon x protocol: incremental utility per block, sign agreement, rank, and the naive rows."""
    out = {"features": {}, "rank_agreement": {}, "reference": {}}
    for (proto, h), sub in table.groupby(["protocol", "horizon_bars"]):
        blocks = sorted(sub["block"].unique())
        full = sub[sub.arm == "full|selected"].set_index("block")
        naive = sub[sub.arm == "naive_zero"].set_index("block")
        key = f"{proto}|h{h}"
        out["reference"][key] = {
            arm: {"mae_z_mean": float(g["mae_z"].mean()), "mae_log_return_mean": float(g["mae_log_return"].mean()),
                  "mae_z_by_block": {b: float(v) for b, v in zip(g["block"], g["mae_z"])},
                  "n_eval_total": int(g["n_eval"].sum())}
            for arm, g in sub[sub.model.isin(["naive"]) | sub.arm.isin(["full|selected", "full|laneB_alpha1", "hgb_full"])]
            .groupby("arm")}
        rank_vectors = {}
        for f in features:
            wo = sub[sub.arm == f"without:{f}"].set_index("block")
            only = sub[sub.arm == f"only:{f}"].set_index("block")
            perm = sub[sub.arm == f"hgb_perm:{f}"].set_index("block")
            hgb = sub[sub.arm == "hgb_full"].set_index("block")
            delta = {b: float(wo.loc[b, "mae_z"] - full.loc[b, "mae_z"]) for b in blocks if b in wo.index}
            delta_rel = {b: delta[b] / float(naive.loc[b, "mae_z"]) for b in delta}
            vals = np.array(list(delta.values()))
            sign = float(max((vals > 0).sum(), (vals < 0).sum()) / len(vals)) if len(vals) else float("nan")
            entry = {
                "delta_mae_z_by_block": delta,
                "delta_mae_over_naive_zero_by_block": delta_rel,
                "delta_mae_z_median": float(np.median(vals)) if len(vals) else float("nan"),
                "delta_mae_z_min": float(vals.min()) if len(vals) else float("nan"),
                "delta_mae_z_max": float(vals.max()) if len(vals) else float("nan"),
                "sign_agreement": sign,
                "only_mae_z_mean": float(only["mae_z"].mean()) if len(only) else float("nan"),
                "only_beats_naive_zero_blocks": int(sum(only.loc[b, "mae_z"] < naive.loc[b, "mae_z"]
                                                        for b in blocks if b in only.index)),
                "n_blocks": int(len(vals)),
            }
            if len(perm) and len(hgb):
                pd_ = {b: float(perm.loc[b, "mae_z"] - hgb.loc[b, "mae_z"]) for b in blocks if b in perm.index}
                entry["hgb_perm_delta_mae_z_by_block"] = pd_
                entry["hgb_perm_delta_mae_z_median"] = float(np.median(list(pd_.values())))
            out["features"].setdefault(f, {})[key] = entry
            rank_vectors[f] = delta
        # ranks per block (1 = largest incremental utility), rank agreement = mean pairwise Spearman
        ranks = {}
        for b in blocks:
            col = {f: rank_vectors[f].get(b) for f in features if b in rank_vectors[f]}
            if not col:
                continue
            order = sorted(col, key=lambda f: -col[f])
            ranks[b] = {f: i + 1 for i, f in enumerate(order)}
        for f in features:
            rs = [ranks[b][f] for b in ranks if f in ranks[b]]
            out["features"][f][key]["rank_by_block"] = {b: ranks[b][f] for b in ranks if f in ranks[b]}
            out["features"][f][key]["rank_median"] = float(np.median(rs)) if rs else float("nan")
        rhos = []
        bl = list(ranks)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                common = [f for f in features if f in ranks[bl[i]] and f in ranks[bl[j]]]
                if len(common) > 2:
                    rhos.append(spearmanr([ranks[bl[i]][f] for f in common], [ranks[bl[j]][f] for f in common])[0])
        out["rank_agreement"][key] = {"mean_pairwise_spearman": float(np.mean(rhos)) if rhos else float("nan"),
                                      "n_pairs": len(rhos)}
    # one stable score per feature: mean over blocks5 horizons of the median relative incremental utility
    scores = {}
    for f, by_key in out["features"].items():
        vals = [np.median(list(e["delta_mae_over_naive_zero_by_block"].values()))
                for k, e in by_key.items() if k.startswith("blocks5|") and e["delta_mae_over_naive_zero_by_block"]]
        signs = [e["sign_agreement"] for k, e in by_key.items() if k.startswith("blocks5|")]
        scores[f] = {"stable_incremental_utility": float(np.mean(vals)) if vals else float("nan"),
                     "mean_sign_agreement": float(np.mean(signs)) if signs else float("nan")}
    out["scores"] = scores
    out["ranking_by_stable_incremental_utility"] = sorted(scores, key=lambda f: -scores[f]["stable_incremental_utility"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", default="1,2,3,4,5,6,36")
    p.add_argument("--protocols", default="blocks5,laneB")
    p.add_argument("--skip-hgb", action="store_true")
    p.add_argument("--hgb-iters", type=int, default=150)
    p.add_argument("--max-features", type=int, default=None, help="pilot only: first n features")
    p.add_argument("--heartbeat-file", default=None)
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, args.split)
    horizons = [int(v) for v in args.horizons.split(",")]
    hb_path = Path(args.heartbeat_file) if args.heartbeat_file else None

    def heartbeat(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid={os.getpid()} {msg}"
        print(line, flush=True)
        if hb_path:
            hb_path.write_text(line + "\n", encoding="utf-8")

    heartbeat(f"bound view={pop.bindings['view']['sha256'][:8]} split_sha={pop.bindings['split']['file_sha256'][:8]} "
              f"origins={len(pop.origins)} features={len(pop.features)}")
    table, summary = run(pop, horizons, args.out, protocols=tuple(args.protocols.split(",")), skip_hgb=args.skip_hgb,
                         hgb_iters=args.hgb_iters, max_features=args.max_features, heartbeat=heartbeat)
    heartbeat(f"done records={len(table)} cpu={summary['cpu_seconds']:.1f}s peak_rss={summary['peak_rss_bytes_self']}")
    for key, ref in summary["reference"].items():
        if key.startswith("blocks5|"):
            print(key, {k: round(v["mae_z_mean"], 5) for k, v in ref.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
