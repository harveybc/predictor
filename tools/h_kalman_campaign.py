"""Lane H: shared variant/arms campaign loop (ETH and EURUSD runners)."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pipe = _load("h_kalman_pipeline")
kf = pipe.kf
arms_lib = pipe.arms_lib


def run_variants(d, groups, variants, heavy_variant, skip_heavy, hb, role, L, replay_inputs, heavy_lags=24):
    """Returns (variants_results, replay) for the declared variants; arms A, B, C and the controls."""
    replay = {"inputs": replay_inputs, "kalman": {}, "arms_exact_inputs": {}, "arms_numeric_predictions": {}}
    out = {}
    for vname in variants:
        hb.stage(f"variant:{vname}:kalman")
        variant = pipe.VARIANTS[vname]
        t0, c0 = time.time(), time.process_time()
        kal = pipe.build_kalman(d, groups, variant, host_role=role)
        kal_cost = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        diag = pipe.kalman_diagnostics(d, kal)
        cost = {g: kf.measure_cost(k["artifact"], d["Z"][:d["val_rows"][1]][:, k["idx"]], repeats=1) for g, k in kal.items()}
        replay["kalman"][vname] = {g: {"artifact_sha256": v["artifact_sha256"], "fitted_state_digest": v["fitted_state_digest"],
                                       "output_digest": v["output_digest"]} for g, v in diag.items()}
        vres = {"kalman_fit_and_transform_cost": kal_cost, "kalman_cost": cost, "diagnostics": diag, "arms": {}}
        for lags in ([1, heavy_lags] if (vname == heavy_variant and not skip_heavy) else [1]):
            hb.stage(f"variant:{vname}:arms:lags{lags}")
            use = ["A", "B", "C", "IDENTITY", "B_PERMUTED", "B_NOISE", "C_EWMA", "C_SMOOTHER_NONCAUSAL"] if lags == 1 \
                else ["A", "B", "C", "C_EWMA"]
            # lag-1 matrices are small and built together; wide (lagged) matrices are built one arm at a time
            arms_all = pipe.arm_matrices(d, kal, lags=lags, controls=True, only=set(use)) if lags == 1 else None
            evs = {}
            for a in use:
                hb.stage(f"variant:{vname}:lags{lags}:arm:{a}")
                t0, c0 = time.time(), time.process_time()
                arm = arms_all[a] if arms_all is not None else \
                    pipe.arm_matrices(d, kal, lags=lags, controls=True, only={a})[a]
                ev = pipe.evaluate_arm(d, arm)
                ev["cost"] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
                ev["input_matrix_validation_sha256"] = arms_lib.sha_array(arm["validation"])
                ev["input_matrix_train_sha256"] = arms_lib.sha_array(arm["train"])
                del arm
                evs[a] = ev
                replay["arms_exact_inputs"][f"{vname}|lags{lags}|{a}"] = {
                    "train": ev["input_matrix_train_sha256"], "validation": ev["input_matrix_validation_sha256"]}
                replay["arms_numeric_predictions"][f"{vname}|lags{lags}|{a}"] = ev["prediction_sha256"]
            if lags == 1:
                assert evs["IDENTITY"]["input_matrix_validation_sha256"] == evs["A"]["input_matrix_validation_sha256"]
            paired = {a: pipe.paired_against(d, evs["A"], evs[a], L) for a in use if a not in ("A", "IDENTITY")}
            vres["arms"][f"lags{lags}"] = {"evaluations": pipe.public(evs), "paired_vs_A": paired}
            del arms_all
        out[vname] = vres
    return out, replay
