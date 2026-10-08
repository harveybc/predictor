"""One matched TRAIN-era weekly architecture comparison; never opens TEST."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools import fs4_temporal_predictor as P
from tools import fs4_weekly_wrapper as W
from tools.i6a_architectures import ARMS, build_model


def make_task(freeze, target, week_ordinal, validation_year):
    if W.digest({k: v for k, v in freeze.items() if k not in ("freeze_sha256", "frozen_utc")}) != freeze.get("freeze_sha256"):
        raise ValueError("FREEZE_DIGEST_MISMATCH")
    winner = freeze["winners"]["EURUSD"][target]["RAW"]
    members = P.canonical_features(winner["members"])
    plan_sha = W.digest({"schema": "i6a.train_week_arch.v1", "freeze_sha256": freeze["freeze_sha256"],
                         "target": target, "week_ordinal": week_ordinal, "members": members,
                         "validation_year": validation_year})
    weeks = [w for w in W.W.build_protocol(validation_year, plan_sha).weeks()
             if w.split is W.EvaluationSplit.VALIDATION]
    if not 0 <= week_ordinal < len(weeks):
        raise ValueError("WEEK_ORDINAL_OUT_OF_RANGE")
    week = weeks[week_ordinal]
    task = {"schema": W.TASK_SCHEMA, "plan_sha256": plan_sha, "population_id": "EURUSD",
            "identity": "phase1-eurusd-final:94d20c038d55e152", "set_id": winner["set_id"],
            "target_id": target, "horizon_hours": 1, "members": list(members), "n_features": len(members),
            "input_mode": "RAW", "stage": 1, "split": "validation", "validation_year": validation_year,
            "week": {k: v for k, v in W._week_dict(week).items() if k != "split"}, "seed": W.SEED,
            "predictor_spec_sha256": P.PredictorSpec().sha256(), "encoder_spec_sha256": P.EncoderSpec().sha256()}
    task["task_id"] = W.digest(task)
    return task


def run(arm, task, store):
    members = P.canonical_features(task["members"])

    def trainer(spec, X, y, fit_idx, inner_idx, input_mode, encoder, seed):
        return P.fit_named(spec, X, members, y, fit_idx, inner_idx, input_mode=input_mode,
                           encoder=encoder, seed=seed, timestamps=store.ts,
                           min_timestamp=int(W._parse(task["week"]["fit_start"]).timestamp()),
                           model_builder=lambda s, f, m, d: build_model(s, f, m, d, arm))

    result = W.run_task(task, store, trainer=trainer)
    if result["disposition"] != "COMPLETED":
        raise ValueError(f"ARCHITECTURE_FIT_FAILED: {arm}: {result.get('reason')}")
    result["trainer"] = "I6A_REAL_KERAS_ARCHITECTURE"
    result["architecture_arm"] = arm
    result["result_sha256"] = W.digest({k: v for k, v in result.items() if k != "result_sha256"})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--feature-parquet", action="append", required=True)
    ap.add_argument("--target-parquet", required=True)
    ap.add_argument("--val-feature-parquet", action="append")
    ap.add_argument("--val-target-parquet")
    ap.add_argument("--validation-year", type=int, default=2023)
    ap.add_argument("--week-ordinal", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.validation_year not in (2023, 2024):
        raise ValueError("ONLY_TRAIN_2023_OR_VALIDATION_2024; TEST_IS_SEALED")
    task = make_task(json.loads(Path(args.freeze).read_text()), "Y_s_1h", args.week_ordinal, args.validation_year)
    if args.validation_year == 2023:
        if args.val_feature_parquet or args.val_target_parquet:
            raise ValueError("TRAIN_PILOT_MUST_NOT_READ_VALIDATION")
        store = W.DataStore.from_train_only("EURUSD", args.feature_parquet, args.target_parquet, bar_hours=1)
    else:
        if not args.val_feature_parquet or not args.val_target_parquet:
            raise ValueError("VALIDATION_INPUTS_REQUIRED")
        store = W.DataStore.from_paths("EURUSD", args.feature_parquet, args.target_parquet,
                                      args.val_feature_parquet, args.val_target_parquet, bar_hours=1)
    result = run(args.arm, task, store)
    out = {"schema": "i6a.weekly_arch_cell.v1", "diagnostic_only": args.validation_year == 2023,
           "selection_conditioned_on_validation_2024": True, "arm": args.arm, "task": task, "result": result}
    out["sha256"] = W.digest(out)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(json.dumps({"status": "COMPLETE", "arm": args.arm, "mae": result["metrics"]["mae"],
                      "naive_mae": result["metrics"]["naive_mae"], "n_scored": result["n_scored"],
                      "n_params": result["cost"]["n_params"], "sha256": out["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
