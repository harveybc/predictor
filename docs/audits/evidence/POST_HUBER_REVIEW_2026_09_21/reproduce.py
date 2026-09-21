"""Non-governing audit probes. Only temporary fabricated data are modified."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import df_benchmark_contract as B
import df_closure_table as C
import df_fin_loss_opt_design as F


def probes():
    out = {"scope": "NON_GOVERNING_SYNTHETIC_AUDIT_NO_TRAINING_NO_SERVICE_WRITES"}
    ours = B.household_ours()
    out["scale_change"] = B.decide(ours, replace(ours, metric_scale="USD"))["mode"]
    out["seconds_change"] = B.decide(ours, replace(ours, horizon_seconds=72 * 3600))["mode"]
    out["boolean_reexecution"] = B.decide(ours, B.gasparin_2019(), reexecuted_on_our_rows=True)["mode"]
    empty = {k: None for k in ("task_id", "dataset_id", "target", "horizon_steps", "split_rule",
                              "target_transform", "scaler_fit_population", "metric_formula", "naive_baseline")}
    empty.update(schema=B.SCHEMA, comparability={"mode": "REPRODUCTION"})
    try:
        B.require({"benchmark_contract": empty})
        out["null_contract"] = "ACCEPTED"
    except (B.ContractRefusal, ValueError, TypeError) as exc:
        out["null_contract"] = type(exc).__name__

    import df_e1_huber as H
    source = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()
    if (source / "DESIGN.json").is_file():
        d = H.seal(source)
        H.validate(d)
        foreign = copy.deepcopy(d)
        foreign["benchmark_contract"]["target"] = "unrelated_target"
        foreign["benchmark_contract"]["horizon_steps"] = 72
        foreign["design_sha256"] = H.P._module("df_mod_e0").sha_obj(
            {k: v for k, v in foreign.items() if k != "design_sha256"})
        try:
            H.validate(foreign)
            out["real_factorial_foreign_task"] = "ACCEPTED"
        except (B.ContractRefusal, ValueError) as exc:
            out["real_factorial_foreign_task"] = type(exc).__name__
    else:
        out["real_factorial_foreign_task"] = "NOT_RUN_SOURCE_UNAVAILABLE"

    with tempfile.TemporaryDirectory(prefix="post-huber-audit-") as directory:
        root = Path(directory)
        unit = "arm_s1"
        a = root / "attempts" / unit
        a.mkdir(parents=True)
        receipt = {"campaign_sha256": "a" * 64, "terminal_sha256": "b" * 64}
        (root / "DESIGN.json").write_text(json.dumps({"purpose": "AUDIT_FIXTURE"}))
        (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {unit: receipt}}))
        origins = np.arange(4)
        Y = np.arange(70, dtype=float)
        truth = Y[origins + 60]
        np.savez(root / "DATA.npz", Y=Y, horizon=[60], target_channel=[0], scaler_sd=[2.])
        def arrays(pred):
            np.savez(a / "arrays.npz", pred=pred, y=truth, naive=Y[origins], origins=origins)
        def wh(_campaign):
            return {"current": {unit: {"terminal_sha256": receipt["terminal_sha256"]}}}
        def table(warehouse=wh):
            return C.build([str(root)], registry=B.registry(), warehouse=warehouse, no_new_measurement=True)
        arrays(truth + 2.)
        first = table()
        arrays(truth)
        second = table()
        out["changed_predictions_same_receipt"] = {
            "mae_before": first["rows"][0]["model_error"],
            "mae_after": second["rows"][0]["model_error"],
            "problems_before": first["problems"], "problems_after": second["problems"],
            "warehouse_digest_matches_both": all(t["rows"][0]["warehouse"]["digest_matches_receipt"]
                                                  for t in (first, second)),
        }
        out["warehouse_missing_terminal_problems"] = table(lambda _: {"current": {}})["problems"]
        arrays(np.full(4, np.nan))
        out["nonfinite_prediction_problems"] = table()["problems"]
        (a / "arrays.npz").unlink()
        out["missing_prediction_file"] = {k: table()[k] for k in ("rows", "problems")}

    y = np.r_[np.zeros(900), np.ones(100)]
    delta = F.delta_candidates(y, train_end=999, horizon=6)
    out["positive_sd_zero_residual_scale"] = {
        "status": delta["status"], "sd_train": delta["sd_train"],
        "candidate_deltas": [c["delta_z"] for c in delta["candidates"]],
    }
    out["row_horizon_is_not_elapsed_time"] = {
        "fixture": "hourly weekday bars, weekend absent",
        "last_friday": "2024-01-05T23:00", "sixth_next_bar": "2024-01-08T05:00",
        "elapsed_hours": int((np.datetime64("2024-01-08T05:00") -
                              np.datetime64("2024-01-05T23:00")) / np.timedelta64(1, "h")),
        "claimed_elapsed_hours_for_six_rows": F.task_freeze()["horizons"]["short"]["hours"],
    }
    out["volume_validation_cadence"] = {
        "assumption": "gap-free extension; actual row enumeration must replace this illustration",
        "batch": 64, "max_updates": 4000,
        "steps_per_epoch": {str(days): int(np.ceil((40080 + (days - 28) * 1440) / 64))
                            for days in (28, 56, 112)},
    }
    return out


def reach_probe():
    import df_e1_pilot as P
    tf = P.E._tf()
    original = P.core_dilations
    try:
        P.core_dilations = lambda _: [1, 2, 4, 8, 16]
        model = P._model_for_target([0], 1440, 1, 0, 1, core="tcn_w")
    finally:
        P.core_dilations = original
    x = tf.Variable(np.random.default_rng(7).normal(0, .1, (1, 1440, 1)).astype("float32"))
    with tf.GradientTape() as tape:
        value = model(x, training=False)[0, 0]
    grad = tape.gradient(value, x).numpy()[0, :, 0]
    indices = np.flatnonzero(grad != 0)
    return {"clamped_core_declared_reach": 60,
            "actual_earliest_nonzero_gradient_lag": int(1439 - indices.min()),
            "actual_support_samples": int(1440 - indices.min()),
            "nonzero_older_than_last_60": int(np.count_nonzero(grad[:-60])),
            "derivation": "branch 5 + core (1 + 2*(1+2+4+8+16)) - 1 = 67"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-reach", action="store_true")
    args = ap.parse_args()
    result = probes()
    if args.with_reach:
        result["measured_clamped_reach"] = reach_probe()
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
