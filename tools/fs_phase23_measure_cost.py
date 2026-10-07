#!/usr/bin/env python3
"""Measure the worker's memory and time per 1,000 pairs on a real-sized synthetic block.

Builds a synthetic TRAIN population with the shape of the real one (rows, features,
folds, missingness), freezes a manifest, plans shards of about ``--pairs-per-shard``
pairs and runs exactly one shard through the real worker path (same code, same
parameters).  Optionally times phase 3 for one target on synthetic phase-2 matrices.
The report is what the INTEGRATION agent needs to size memory caps and shard counts.

    fs_phase23_measure_cost.py --out DIR [--rows 71734 --features 366 --folds 5 --pairs-per-shard 1000] [--phase3]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import feature_pairwise_campaign as camp  # noqa: E402
from tools import fs_phase23_manifest as man  # noqa: E402


def build_population(out: Path, *, rows: int, features: int, folds: int, nan_fraction: float, seed: int, bar_hours: int) -> tuple[Path, Path]:
    import pandas as pd
    rng = np.random.default_rng(seed)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2012-05-01", periods=rows, freq=f"{bar_hours}h", tz="UTC")
    names = [f"syn.f{i:04d}" for i in range(features)]
    base = rng.normal(size=(rows, 8))
    cols = {}
    for i, name in enumerate(names):
        k = i % 8
        v = base[:, k] * rng.uniform(0.5, 2.0) + rng.normal(size=rows) * rng.uniform(0.2, 1.5)
        if i % 10 == 3:
            v = v ** 2
        if rng.uniform() < nan_fraction * 3:
            lead = int(rng.integers(1, int(rows * nan_fraction * 3) + 2))
            v[:lead] = np.nan
        cols[name] = v
    frame = pd.DataFrame({"t_decision_utc": ts, "row_id": np.arange(rows), **cols})
    frame.to_parquet(data / "features_train.parquet", index=False)
    targets = pd.DataFrame({"t_decision_utc": ts, "row_id": np.arange(rows), "Y_syn_1h": np.roll(base[:, 0], -1) + rng.normal(size=rows)})
    targets.to_parquet(data / "targets_train.parquet", index=False)
    fold_rows = [int(rows * (0.55 + 0.09 * k)) for k in range(folds)]
    meta = [{"feature_id": n, "family": "syn", "source": f"lake:syn/{i % 5}.parquet", "transform": "t", "unit": "u", "availability_time": "t",
             "clock": "OBSERVED", "support_h": 1.0, "train_coverage": float(np.isfinite(cols[n]).mean()), "source_bytes": rows * 8}
            for i, n in enumerate(names)]
    manifest = man.build_manifest(
        population_id="SYNBIG", identity="phase1-final:SYNBIG:" + "ee" * 8, identity_source={"file": "synthetic", "field": "synthetic"},
        phase1={"plan_sha256": "0" * 64},
        data={"features_file": "features_train.parquet", "features_sha256": man.sha256_file(data / "features_train.parquet"),
              "targets_file": "targets_train.parquet", "targets_sha256": man.sha256_file(data / "targets_train.parquet"),
              "timestamp_column": "t_decision_utc", "row_id_column": "row_id", "bar_hours": bar_hours, "train_rows": rows},
        features=meta, targets=[{"target_id": "Y_syn_1h", "column": "Y_syn_1h", "family": "Y", "head": "short", "horizon_hours": 1}],
        folds=[{"fold_id": f"inner_{k}", "train_rows": [0, e]} for k, e in enumerate(fold_rows)],
        causal_supported=[{"feature_id": names[0], "target_id": "Y_syn_1h", "horizon_hours": 1}], contract_source="synthetic_measurement",
        data_root=data)
    mpath = out / "MANIFEST.json"
    man.write_manifest(manifest, mpath)
    return mpath, data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rows", type=int, default=71734)
    p.add_argument("--features", type=int, default=366)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bar-hours", type=int, default=1)
    p.add_argument("--nan-fraction", type=float, default=0.09)
    p.add_argument("--pairs-per-shard", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase3", action="store_true")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mpath, data = build_population(out, rows=args.rows, features=args.features, folds=args.folds, nan_fraction=args.nan_fraction,
                                   seed=args.seed, bar_hours=args.bar_hours)
    build_s = time.time() - t0
    pairs = args.features * (args.features - 1) // 2
    n_shards = max(1, round(pairs / args.pairs_per_shard))
    state = out / "state"
    plan = camp.plan(manifest_path=mpath, state_root=state, n_shards=n_shards, hosts=[{"host_id": "measure", "size_class": "large"}])
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t1 = time.time()
    summary = camp.run_worker(plan_path=state / "PLAN.json", state_root=state, data_root=data, host_id="measure", max_shards=1)
    wall = time.time() - t1
    unit = summary["units"][0] if summary["units"] else None
    report = {
        "schema": "fs_phase23.cost_measurement.v1", "rows": args.rows, "features": args.features, "folds": args.folds,
        "expected_pairs": pairs, "n_shards": n_shards, "build_seconds": build_s, "rss_before_worker_bytes": rss_before,
        "worker": summary, "shard_pairs": unit["pairs"] if unit else 0, "shard_wall_seconds": unit["wall_seconds"] if unit else None,
        "worker_wall_seconds_including_load": wall,
        "seconds_per_1000_pairs": (unit["wall_seconds"] / unit["pairs"] * 1000.0) if unit and unit["pairs"] else None,
        "peak_rss_bytes": summary["peak_rss_bytes"], "peak_rss_gib": summary["peak_rss_bytes"] / 2**30,
    }
    if report["seconds_per_1000_pairs"]:
        report["estimated_core_hours_all_pairs"] = report["seconds_per_1000_pairs"] * pairs / 1000.0 / 3600.0
    if args.phase3:
        from tools import feature_filter_selection as sel
        manifest = man.load_manifest(mpath)
        feats = [f["feature_id"] for f in manifest["features"]]
        rng = np.random.default_rng(1)
        pf = len(feats)
        sp = rng.uniform(-0.6, 0.6, size=(pf, pf))
        sp = (sp + sp.T) / 2
        np.fill_diagonal(sp, 1.0)
        mi = np.abs(rng.normal(0.05, 0.03, size=(pf, pf)))
        mi = (mi + mi.T) / 2
        mats = {"features": feats, "spearman": sp, "mi": mi, "admissible": np.ones(pf, dtype=bool), "population_id": "SYNBIG",
                "identity": manifest["identity"], "alias_representative": feats, "params_sha256": "synthetic"}
        t2 = time.time()
        fl, X, y = sel.load_train_matrix(manifest, data, "Y_syn_1h")
        res = sel.run_filter_methods(manifest, mats, fl, X, y, target_id="Y_syn_1h", seed=1)
        report["phase3_one_target_seconds"] = time.time() - t2
        report["phase3_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["phase3_rows"] = {t: len(r) for t, r in res["rows"].items()}
    (out / "COST_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "worker"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
