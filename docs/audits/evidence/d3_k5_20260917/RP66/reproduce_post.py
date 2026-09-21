"""POST: Musashi's POST_HUBER_REVIEW probes re-run against the repaired governing code (RP66/RP67).

His script is left untouched; this harness calls the same probes and records a refusal where the
repaired code refuses (a TypeError on the removed boolean is the intended outcome, not a crash).
Non-governing: only fabricated temporary data are written; no training, no service writes.
"""
import argparse
import copy
import json
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/"tools"))
import df_benchmark_contract as B
import df_closure_table as C
import df_fin_loss_opt_design as F


def refused(fn):
    try:
        return {"outcome": "ACCEPTED", "value": fn()}
    except BaseException as exc:                      # ContractRefusal is a SystemExit on purpose
        return {"outcome": "REFUSED", "exception": type(exc).__name__, "message": str(exc)[:300]}


def probes():
    out = {"scope": "NON_GOVERNING_SYNTHETIC_AUDIT_NO_TRAINING_NO_SERVICE_WRITES", "base": "repaired tree (RP66/RP67)"}
    ours = B.household_ours()
    out["scale_change"] = B.decide(ours, replace(ours, metric_scale="USD"))["mode"]           # 'USD' is not a scale -> invalid
    out["scale_change_valid_native"] = B.decide(ours, replace(ours, metric_scale="native"))["mode"]
    out["seconds_change"] = B.decide(ours, replace(ours, horizon_seconds=72*3600))["mode"]
    out["boolean_reexecution"] = refused(lambda: B.decide(ours, B.gasparin_2019(), reexecuted_on_our_rows=True)["mode"])
    empty = {k: None for k in ("task_id", "dataset_id", "target", "horizon_steps", "split_rule",
                              "target_transform", "scaler_fit_population", "metric_formula", "naive_baseline")}
    empty.update(schema=B.SCHEMA, comparability={"mode": "REPRODUCTION"})
    out["null_contract"] = refused(lambda: B.require({"benchmark_contract": empty}))
    out["unknown_vs_unknown"] = B.decide(B.kim_cho_2019(), replace(B.kim_cho_2019(), task_id="other"))["mode"]

    import df_e1_huber as H
    source = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()
    if (source/"DESIGN.json").is_file():
        d = H.seal(source)
        H.validate(d)
        sha_obj = H.P._module("df_mod_e0").sha_obj
        foreign = copy.deepcopy(d)                                   # Musashi's probe: outer digest re-digested only
        foreign["benchmark_contract"]["target"] = "unrelated_target"
        foreign["benchmark_contract"]["horizon_steps"] = 72
        foreign["design_sha256"] = sha_obj({k: v for k, v in foreign.items() if k != "design_sha256"})
        out["real_factorial_foreign_task"] = refused(lambda: H.validate(foreign))
        consistent = copy.deepcopy(d)                                # inner AND outer digests consistent, still foreign
        consistent["benchmark_contract"]["target"] = "unrelated_target"
        consistent["benchmark_contract"]["horizon_steps"] = 72
        consistent["benchmark_contract"]["horizon_seconds"] = 72*60
        consistent["benchmark_contract"]["contract_sha256"] = B.BenchmarkContract.from_block(consistent["benchmark_contract"]).sha256()
        consistent["benchmark_contract"]["comparability"]["ours_sha256"] = consistent["benchmark_contract"]["contract_sha256"]
        consistent["design_sha256"] = sha_obj({k: v for k, v in consistent.items() if k != "design_sha256"})
        out["real_factorial_foreign_task_consistently_redigested"] = refused(lambda: H.validate(consistent))
    else:
        out["real_factorial_foreign_task"] = "NOT_RUN_SOURCE_UNAVAILABLE"

    # Musashi's closure fixture, verbatim (unregistered unit, no design roles)
    with tempfile.TemporaryDirectory(prefix="post-huber-audit-") as directory:
        root = Path(directory)
        unit = "arm_s1"
        a = root/"attempts"/unit
        a.mkdir(parents=True)
        receipt = {"campaign_sha256": "a"*64, "terminal_sha256": "b"*64}
        (root/"DESIGN.json").write_text(json.dumps({"purpose": "AUDIT_FIXTURE"}))
        (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {unit: receipt}}))
        origins = np.arange(4)
        Y = np.arange(70, dtype=float)
        truth = Y[origins+60]
        np.savez(root/"DATA.npz", Y=Y, horizon=[60], target_channel=[0], scaler_sd=[2.], eval_origins=origins)

        def arrays(pred):
            np.savez(a/"arrays.npz", pred=pred, y=truth, naive=Y[origins], origins=origins)

        def wh(_campaign):
            return {"current": {unit: {"terminal_sha256": receipt["terminal_sha256"]}}}

        def table(warehouse=wh):
            return C.build([str(root)], registry=B.registry(), warehouse=warehouse, no_new_measurement=True)
        arrays(truth+2.)
        first = table()
        arrays(truth)
        second = table()
        out["musashi_fixture_unregistered_unit"] = {
            "mae_before": first["rows"][0]["model_error"], "mae_after": second["rows"][0]["model_error"],
            "problems_before": first["problems"], "problems_after": second["problems"],
            "verified_rows": [first["verified_rows"], second["verified_rows"]]}
        # the same fixture with the unit REGISTERED in the design (arm), the terminal payload binding the arrays
        (root/"DESIGN.json").write_text(json.dumps({"purpose": "AUDIT_FIXTURE", "cells": [{"cell_id": unit, "arm": "arm", "seed": 1}]}))
        (root/"TERMINALS").mkdir()

        def bind(sha):
            (root/"TERMINALS"/f"{unit}.json").write_text(json.dumps({"artifacts": [{"role": "predictions", "sha256": sha}]}))

        def wh_full(sha, present=True):
            return lambda _c: {"current": {unit: {"terminal_sha256": receipt["terminal_sha256"], "status": "COMPLETED",
                                                  "artifacts": [{"role": "predictions", "sha256": sha}]}}} if present else {"current": {}}
        arrays(truth+2.)
        sha = C.sha_file(a/"arrays.npz"); bind(sha)
        first = table(wh_full(sha))
        arrays(truth)                                                # changed under the same receipt/terminal
        second = table(wh_full(sha))
        out["changed_predictions_same_receipt"] = {
            "mae_before": first["rows"][0]["model_error"], "mae_after": second["rows"][0]["model_error"],
            "problems_before": first["problems"], "problems_after": second["problems"],
            "binding": [first["rows"][0]["binding"]["level"], second["rows"][0]["binding"]["level"]],
            "verified_rows": [first["verified_rows"], second["verified_rows"]]}
        arrays(truth+2.)
        out["warehouse_missing_terminal_problems"] = table(wh_full(sha, present=False))["problems"]
        arrays(np.full(4, np.nan))
        sha = C.sha_file(a/"arrays.npz"); bind(sha)
        out["nonfinite_prediction_problems"] = table(wh_full(sha))["problems"]
        (a/"arrays.npz").unlink()
        t = table(wh_full(sha))
        out["missing_prediction_file"] = {"rows": len(t["rows"]), "problems": t["problems"], "verified_rows": t["verified_rows"]}

    # RP70 probes (financial): recorded as they stand on this tree; repaired in the RP70 block
    y = np.r_[np.zeros(900), np.ones(100)]
    try:
        delta = F.delta_candidates(y, train_end=999, horizon=6)
        out["positive_sd_zero_residual_scale"] = {"status": delta["status"], "sd_train": delta["sd_train"],
                                                  "candidate_deltas": [c["delta_z"] for c in delta["candidates"]]}
    except Exception as exc:
        out["positive_sd_zero_residual_scale"] = {"exception": type(exc).__name__, "message": str(exc)[:300]}
    tf = F.task_freeze()
    out["row_horizon_is_not_elapsed_time"] = {"short_horizon_declaration": tf["horizons"].get("short")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = probes()
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=str)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False, default=str))
