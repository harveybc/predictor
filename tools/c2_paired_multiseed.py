"""Lane C2: four-seed paired inference for M07's EURUSD lake vDh cells (grouped_all vs control_mlp, mae/adamw), DEVELOPMENT.

Per cell the checks and per-seed intervals of `c2_paired_loss_inference.analyse` run unchanged (each cell is refused unless the evidence MAEs are
reproduced). Added here:
* pooled over seeds, per horizon, loss in {abs, sq}, control in {zero-return, intercept-only}: the mean over seeds of the per-seed mean paired difference, with a
  circular block bootstrap in which every seed's row series is resampled INDEPENDENTLY (block length L = max over seeds of the cell's L) and the seed means are
  averaged; 95 % percentile interval; the seed range (min, max of the per-seed mean differences); how many seeds exclude 0 (per-seed intervals);
* architecture effect: per seed, d_arch = loss(grouped_all) - loss(control_mlp) row by row against the SAME rows, per-seed intervals and the pooled version.
The pooled interval treats seeds as independent replicates of one training procedure on one validation sample; the validation rows are shared, so it does
NOT account for between-sample variation: it is a statement about this validation sample, not about new data.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import EURUSD_LAKE_A_S1_SPEC, bind_population  # noqa: E402
from c2_battery_calibration import integrated_autocorr_length  # noqa: E402
from c2_paired_loss_inference import (analyse, load_predictions, m07_fin_scaler, paired_row, sha256_file, validation_targets)  # noqa: E402


def pooled_bootstrap(series_list, L, b, rng):
    """Mean over seeds of the seed means, each seed's series resampled independently by circular blocks; (se, lo, hi)."""
    n = len(series_list[0])
    L = max(1, min(int(L), n // 2))
    nblocks = int(np.ceil(n / L))
    step = max(1, 2_000_000 // n)
    out = []
    for lo in range(0, b, step):
        k = min(step, b - lo)
        acc = np.zeros(k)
        for d in series_list:
            starts = rng.integers(0, n, size=(k, nblocks))
            idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(k, -1)[:, :n] % n
            acc += d[idx].mean(axis=1)
        out.append(acc / len(series_list))
    m = np.concatenate(out)
    return float(np.std(m, ddof=1)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pooled_row(series_list, L, b, rng):
    se, lo, hi = pooled_bootstrap(series_list, L, b, rng)
    means = [float(d.mean()) for d in series_list]
    mean = float(np.mean(means))
    return {"mean_diff": mean, "se": se, "ci95": [lo, hi], "block_length": int(L), "seeds": len(series_list), "n_rows_each": int(len(series_list[0])),
            "seed_range": [min(means), max(means)], "seed_means": means, "excludes_zero": bool(lo > 0 or hi < 0),
            "side": "MODEL_BETTER" if hi < 0 else ("MODEL_WORSE" if lo > 0 else "INCLUDES_ZERO")}


def cell_series(pop, rows, preds, horizons):
    """Per horizon: errors of the model, zero-return and intercept-only forecasts in z-units (same scaler rule as analyse(scaler='m07_fin'))."""
    lr1 = pop.frame["log_return_1"].to_numpy(dtype=np.float64)
    out = {}
    for h in horizons:
        y = validation_targets(lr1, pop.mu, pop.sigma, pop.times, rows, h, pop.bar_seconds)
        tr = pop.m07_target(h)[pop.origins]
        out[h] = {"e_model": preds[h] - y, "e_zero": (-h * pop.mu / pop.sigma) - y, "e_int": float(np.mean(tr[np.isfinite(tr)])) - y, "y": y}
    return out


def run(pop, cells, horizons, *, boot=2000, seed=20261001, work=None):
    """cells: {(label, seed): (predictions_path, evidence_path)}; returns the result document."""
    rng = np.random.default_rng(seed)
    per_cell, series = {}, {}
    for (label, sd), (pp, ep) in sorted(cells.items()):
        rows, preds = load_predictions(pp, horizons)
        evidence = json.loads(Path(ep).read_text())
        res = analyse(pop, rows, preds, evidence, boot=boot, scaler="m07_fin")        # refuses on irreproducible evidence
        res["inputs"] = {"predictions_sha256": sha256_file(pp), "evidence_sha256_file": sha256_file(ep), "candidate": evidence["artifact"]["candidate_cid"]}
        res.pop("bindings", None)
        per_cell[f"{label}|{sd}"] = res
        series[(label, sd)] = (rows, cell_series(pop, rows, preds, horizons))
    labels = sorted({k[0] for k in series})
    seeds = sorted({k[1] for k in series})
    pooled = {}
    for label in labels:
        pooled[label] = {}
        for h in horizons:
            pooled[label][h] = {}
            for ctrl in ("zero", "int"):
                for loss in ("abs", "sq"):
                    ds = []
                    for sd in seeds:
                        s = series[(label, sd)][1][h]
                        e, c = s["e_model"], s["e_" + ctrl]
                        ds.append(np.abs(e) - np.abs(c) if loss == "abs" else e ** 2 - c ** 2)
                    L = int(max(max(per_cell[f"{label}|{sd}"]["horizons"][h][f"{loss}_{ctrl}"]["block_length"] for sd in seeds), 1))
                    pooled[label][h][f"{loss}_{ctrl}"] = pooled_row(ds, L, boot, rng)
                    pooled[label][h][f"{loss}_{ctrl}"]["seeds_excluding_zero"] = [sd for sd in seeds if per_cell[f"{label}|{sd}"]["horizons"][h][f"{loss}_{ctrl}"]["excludes_zero"]]
    arch = {}
    if len(labels) == 2:
        a, b_ = labels  # control_mlp..., grouped_all... (sorted); d = first - second
        g, c = ("grouped_all_mae_adamw", "control_mlp_mae_adamw")
        for h in horizons:
            arch[h] = {"per_seed": {}, "pooled": {}}
            for loss in ("abs", "sq"):
                ds = []
                for sd in seeds:
                    if not np.array_equal(series[(g, sd)][0], series[(c, sd)][0]):
                        raise ValueError("ARCHITECTURE_COMPARISON_ON_DIFFERENT_ROWS")
                    eg, ec = series[(g, sd)][1][h]["e_model"], series[(c, sd)][1][h]["e_model"]
                    d = np.abs(eg) - np.abs(ec) if loss == "abs" else eg ** 2 - ec ** 2
                    ds.append(d)
                    L = int(max(integrated_autocorr_length(d, len(d) // 20), h + 6))
                    arch[h]["per_seed"].setdefault(loss, {})[sd] = paired_row(d, L, boot, rng)
                Lp = max(r["block_length"] for r in arch[h]["per_seed"][loss].values())
                arch[h]["pooled"][loss] = pooled_row(ds, Lp, boot, rng)
    return {"schema": "c2_paired_multiseed.v1", "label": "DEVELOPMENT", "labels": labels, "seeds": seeds, "horizons": horizons, "boot": boot,
            "per_cell": per_cell, "pooled_vs_controls": pooled, "architecture_grouped_minus_control": arch,
            "note": "d = first minus second; negative = the model (or grouped_all) is better. Seeds are replicates on one shared validation sample."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifacts", required=True, help="dir holding ARTIFACTS_INDEX.json, PREDICTIONS_*.csv, EVIDENCE_*.json (read-only copy)")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default="grouped_all_mae_adamw,control_mlp_mae_adamw")
    p.add_argument("--seeds", default="2021,2022,2023,2024")
    p.add_argument("--horizons", default="1,2,3,4")
    p.add_argument("--boot", type=int, default=2000)
    args = p.parse_args(argv)
    art = Path(args.artifacts)
    index = json.loads((art / "ARTIFACTS_INDEX.json").read_text())["cells"]
    want_labels, want_seeds = args.labels.split(","), {int(v) for v in args.seeds.split(",")}
    cells = {}
    for cid, c in index.items():
        if c["label"] in want_labels and int(c["seed"]) in want_seeds:
            pp, ep = art / c["predictions"]["path"], art / c["evidence"]["path"]
            if sha256_file(pp) != c["predictions"]["sha256"] or sha256_file(ep) != c["evidence"]["sha256"]:
                raise ValueError(f"ARTIFACT_SHA_MISMATCH {cid}")
            cells[(c["label"], int(c["seed"]))] = (pp, ep)
    if len(cells) != len(want_labels) * len(want_seeds):
        raise ValueError(f"EXPECTED_{len(want_labels) * len(want_seeds)}_CELLS_FOUND_{len(cells)}")
    pop = bind_population(args.view, args.manifest, spec=EURUSD_LAKE_A_S1_SPEC)
    t0 = time.process_time()
    res = run(pop, cells, [int(v) for v in args.horizons.split(",")], boot=args.boot)
    res.update({"bindings": pop.bindings, "cells_used": {f"{k[0]}|{k[1]}": {"predictions": str(v[0].name), "evidence": str(v[1].name)} for k, v in sorted(cells.items())},
                "cpu_seconds": time.process_time() - t0, "peak_rss_bytes_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({"cpu": res["cpu_seconds"], "peak": res["peak_rss_bytes_self"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
