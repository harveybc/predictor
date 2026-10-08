"""Close the complete matched 2024 architecture campaign; refuse partial means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import fs4_weekly_wrapper as W
from tools.i6a_architectures import ARMS
from tools.i6a_campaign import atomic_json, paired_week, verify_result
from tools.i6a_weekly_arch_pilot import make_task


def close(freeze: dict, roots):
    first = make_task(freeze, "Y_s_1h", 0, 2024)
    weeks = [w for w in W.W.build_protocol(2024, first["plan_sha256"]).weeks()
             if w.split is W.EvaluationSplit.VALIDATION]
    records = {}
    problems = []
    for i in range(len(weeks)):
        task = make_task(freeze, "Y_s_1h", i, 2024)
        for arm in ARMS:
            paths = [root / f"{arm}_val2024_week{i}.json" for root in roots]
            present = [p for p in paths if p.is_file()]
            if len(present) != 1:
                problems.append({"week": i, "arm": arm, "reason": "MISSING" if not present else "DUPLICATE"})
                continue
            try:
                records[(i, arm)] = verify_result(present[0], arm, task)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                problems.append({"week": i, "arm": arm, "reason": str(exc)})
        if all((i, arm) in records for arm in ARMS):
            try:
                paired_week([records[(i, arm)] for arm in ARMS])
                inputs = [records[(i, arm)]["result"]["input_sha256"] for arm in ARMS]
                if len(set(inputs)) != 1:
                    raise ValueError("PAIRED_WEEK_INPUT_MISMATCH")
            except ValueError as exc:
                problems.append({"week": i, "reason": str(exc)})
    base = {"schema": "i6a.architecture_closure.v1", "year": 2024, "split": "validation",
            "expected_weeks": len(weeks), "expected_cells": len(weeks) * len(ARMS),
            "verified_cells": len(records), "problems": problems, "test_read": False,
            "selected_set_conditioned_on_validation": True, "seed": W.SEED}
    if problems:
        base["state"] = "INCOMPLETE_EVIDENCE"
        base["sha256"] = W.digest(base)
        return base
    core_code = {r["result"]["code_sha256"] for r in records.values()}
    if len(core_code) != 1:
        base["state"] = "INCOMPLETE_EVIDENCE"
        base["problems"] = [{"reason": "MIXED_CORE_CODE_IDENTITIES"}]
        base["sha256"] = W.digest(base)
        return base
    base["core_code_sha256"] = next(iter(core_code))
    by_arm = {}
    ref = np.array([records[(i, "ARCH_A")]["result"]["metrics"]["mae"] for i in range(len(weeks))])
    for arm in ARMS:
        results = [records[(i, arm)]["result"] for i in range(len(weeks))]
        mae = np.array([r["metrics"]["mae"] for r in results])
        naive = np.array([r["metrics"]["naive_mae"] for r in results])
        params = sorted(set(r["cost"]["n_params"] for r in results))
        architecture_ids = sorted(set(r["architecture_sha256"] for r in results))
        if len(params) != 1 or len(architecture_ids) != 1 or not np.isfinite(mae).all() or not np.isfinite(naive).all() or not np.all(naive > 0):
            base["state"] = "INCOMPLETE_EVIDENCE"
            base["problems"] = [{"arm": arm, "reason": "INVALID_METRIC_OR_PARAMETER_CONTRACT"}]
            base["sha256"] = W.digest(base)
            return base
        by_arm[arm] = {"mean_weekly_mae": float(mae.mean()), "mean_weekly_naive_mae": float(naive.mean()),
                       "mean_weekly_skill_mae": float(np.mean((naive - mae) / naive)),
                       "weeks_better_than_naive": int(np.sum(mae < naive)),
                       "mean_paired_delta_mae_vs_arch_a": float(np.mean(mae - ref)),
                       "n_params": params[0],
                       "architecture_sha256": architecture_ids[0],
                       "fit_seconds_total": float(sum(r["cost"]["fit_seconds"] for r in results)),
                       "cell_digests": [records[(i, arm)]["sha256"] for i in range(len(weeks))]}
    base.update({"state": "COMPLETE", "arms": by_arm,
                 "paired_naive_row_population": [records[(i, "ARCH_A")]["result"]["rows_sha256"]
                                                for i in range(len(weeks))]})
    base["sha256"] = W.digest(base)
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--result-dir", action="append", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    result = close(json.loads(Path(args.freeze).read_text()), args.result_dir)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("state", "verified_cells", "expected_cells", "sha256")}))
    return 0 if result["state"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
