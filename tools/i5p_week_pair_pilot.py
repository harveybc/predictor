"""One TRAIN-year weekly RAW/reconstructed predictor pilot, never submitted to FS4."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from tools import fs4_temporal_predictor as P
from tools import fs4_weekly_wrapper as W
from tools.i5p_reconstructed_inputs import (
    PINNED_EXTRACTOR_ROOT, data_sha256, load_pinned_extractor,
    reconstruct_selected_inputs, sha256_file,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--feature-parquet", action="append", required=True)
    ap.add_argument("--target-parquet", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--target", default="Y_s_1h")
    ap.add_argument("--week-ordinal", type=int, default=0)
    ap.add_argument("--extractor-root", default=str(PINNED_EXTRACTOR_ROOT))
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.week_ordinal < 0:
        raise ValueError("WEEK_ORDINAL_INVALID")
    X, _ = load_pinned_extractor(args.extractor_root)
    frozen = json.loads(Path(args.freeze).read_text())
    if X.digest({k: v for k, v in frozen.items() if k not in ("freeze_sha256", "frozen_utc")}) != frozen.get("freeze_sha256"):
        raise ValueError("FREEZE_DIGEST_MISMATCH")
    winner = frozen["winners"]["EURUSD"][args.target]["RAW"]
    members = tuple(P.canonical_features(winner["members"]))
    identity = "phase1-eurusd-final:94d20c038d55e152"
    expected = X.CORPORA[identity]["files"]
    role_files = {f"eurusd_ps1_{Path(p).parent.name}": Path(p) for p in args.feature_parquet}
    if set(role_files) != set(expected) or any(sha256_file(role_files[r]) != h for r, h in expected.items()):
        raise ValueError("FEATURE_CORPUS_IDENTITY_MISMATCH")
    store = W.DataStore.from_train_only("EURUSD", list(role_files.values()), args.target_parquet, bar_hours=1)
    if args.target not in store.targets or any(m not in store.col for m in members):
        raise ValueError("TARGET_OR_MEMBER_MISSING")
    plan_sha = W.digest({"schema": "i5p.train_week_pair.v1", "freeze_sha256": frozen["freeze_sha256"],
                         "target": args.target, "week_ordinal": args.week_ordinal, "members": members})
    protocol = W.W.build_protocol(2023, plan_sha)
    weeks = [w for w in protocol.weeks() if w.split is W.EvaluationSplit.VALIDATION]
    if args.week_ordinal >= len(weeks):
        raise ValueError("WEEK_ORDINAL_OUT_OF_RANGE")
    week = weeks[args.week_ordinal]
    if week.ordinal != args.week_ordinal:
        raise ValueError("WEEK_ORDINAL_MISMATCH")
    week_data = W._week_dict(week)
    task = {"schema": W.TASK_SCHEMA, "plan_sha256": plan_sha, "population_id": "EURUSD",
            "identity": identity, "set_id": winner["set_id"], "target_id": args.target,
            "horizon_hours": 1, "members": list(members), "n_features": len(members),
            "input_mode": "RAW", "stage": 1, "split": "validation", "validation_year": 2023,
            "week": {k: v for k, v in week_data.items() if k != "split"}, "seed": W.SEED,
            "predictor_spec_sha256": P.PredictorSpec().sha256(),
            "encoder_spec_sha256": P.EncoderSpec().sha256()}
    task["task_id"] = W.digest(task)
    cols = [store.col[m] for m in members]
    raw = store.X[:, cols]
    y = store.targets[args.target]
    ts = store.ts
    fit_start = int(week.fit_start.timestamp())
    end = int(week.end.timestamp())
    eligible = np.flatnonzero((ts >= fit_start) & (ts < end) & np.isfinite(y) & np.isfinite(raw).all(axis=1))
    eligible = eligible[eligible >= P.PredictorSpec().window - 1]
    if not eligible.size:
        raise ValueError("NO_COMPLETE_INPUT_ORIGINS")
    refs = {}
    root = Path(args.results_root)
    for feature in members:
        claim = {"schema": "fs4.extractibility.task.v1", "population_id": "EURUSD", "identity": identity,
                 "feature_id": feature, "fold_id": "inner_2023", "arm": "TRAINED_ENCODER_V2", "seed": 0}
        terminal = root / X.task_digest(claim) / "result.json"
        if not terminal.is_file():
            raise ValueError(f"DONOR_MISSING: {feature}")
        rec = json.loads(terminal.read_text())
        if rec.get("fold", {}).get("fit", [None, None])[1] > int(week.start.timestamp()):
            raise ValueError(f"DONOR_FUTURE_FIT: {feature}")
        refs[feature] = {"terminal": terminal, "sha256": sha256_file(terminal)}
    naive = np.zeros(eligible.size, dtype="float64")
    reconstructed = reconstruct_selected_inputs(
        timestamps=ts, row_ids=store.row_ids, values=raw, columns=members,
        scored_rows=eligible, target=y[eligible], naive=naive, checkpoints=refs,
        expected_data_sha256=data_sha256(ts, store.row_ids, raw, members, eligible, y[eligible], naive),
        extractor_root=args.extractor_root)
    if not np.array_equal(reconstructed.row_ids, store.row_ids[eligible]) or \
            not np.array_equal(reconstructed.target, y[eligible]) or not np.array_equal(reconstructed.naive, naive):
        raise ValueError("TARGET_OR_SUPPORT_CHANGED")
    altered = copy.copy(store)
    altered.X = store.X.copy()
    altered.X[np.ix_(eligible, cols)] = reconstructed.values
    transform_sha = data_sha256(ts, store.row_ids, altered.X[:, cols], members, eligible, y[eligible], naive)
    altered.digests = {**store.digests, "i5p_reconstruction_v2": transform_sha}
    raw_result = W.run_task(task, store)
    reconstructed_result = W.run_task(task, altered)
    for key in ("rows_sha256", "n_scored", "fit_population_digest", "inner_population_digest",
                "fit_rows", "inner_rows", "predictor_spec_sha256", "budget_sha256"):
        if raw_result.get(key) != reconstructed_result.get(key):
            raise ValueError(f"PAIRED_SUPPORT_MISMATCH: {key}")
    if raw_result.get("naive") != reconstructed_result.get("naive") or \
            raw_result.get("metrics", {}).get("naive_mae") != reconstructed_result.get("metrics", {}).get("naive_mae"):
        raise ValueError("PAIRED_NAIVE_MISMATCH")
    if raw_result.get("disposition") != "COMPLETED" or reconstructed_result.get("disposition") != "COMPLETED":
        raise ValueError(f"WEEKLY_FIT_FAILED: RAW {raw_result.get('reason')} RECON {reconstructed_result.get('reason')}")
    report = {"schema": "i5p.train_week_pair_result.v1", "diagnostic_only": True,
              "population": "EURUSD", "target": args.target, "year": 2023, "week_ordinal": args.week_ordinal,
              "members": list(members), "n_reconstructed_origins": int(eligible.size),
              "n_missing_origin_observations": int(np.sum((ts >= fit_start) & (ts < end) & ~np.isfinite(raw).all(axis=1))),
              "transform_sha256": transform_sha, "reconstruction_receipt": reconstructed.receipt,
              "raw": raw_result, "reconstructed": reconstructed_result,
              "delta_mae_reconstructed_minus_raw": reconstructed_result["metrics"]["mae"] - raw_result["metrics"]["mae"]}
    report["result_sha256"] = W.digest(report)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({"status": "COMPLETE", "result_sha256": report["result_sha256"],
                      "week_start": week_data["start"], "n_scored": raw_result["n_scored"],
                      "raw_mae": raw_result["metrics"]["mae"],
                      "reconstructed_mae": reconstructed_result["metrics"]["mae"],
                      "naive_mae": raw_result["metrics"]["naive_mae"],
                      "delta_mae": report["delta_mae_reconstructed_minus_raw"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
