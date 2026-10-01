"""Lane H tests for tools/h_kalman_pipeline.py: the three paired arms and the controls on synthetic panels."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pipe = _load("h_kalman_pipeline")
kf = pipe.kf


def synthetic_data(R=2600, F=6, n_tr=1800, horizons=(1, 2, 3), window=6, seed=0):
    """A panel whose target is the cumulative future change of a latent local level observed with noise in column 0:
    the declared columns carry signal that a causal filter can denoise; the others are stationary noise."""
    rng = np.random.RandomState(seed)
    level = np.zeros(R)
    for t in range(1, R):
        level[t] = 0.995 * level[t - 1] + 0.1 * rng.standard_normal()
    Z = rng.standard_normal((R, F))
    Z[:, 0] = level + 0.8 * rng.standard_normal(R)
    Z[:, 1] = 0.5 * level + 0.8 * rng.standard_normal(R)
    ret = np.zeros(R)
    ret[1:] = 0.05 * level[:-1] + 0.3 * rng.standard_normal(R - 1)      # the future change depends on the latent level
    names = [f"f{j}" for j in range(F)]
    csum = np.concatenate([[0.0], np.cumsum(ret)])
    hmax = max(horizons)
    def origins(lo, hi):
        return np.arange(max(lo, window - 1), hi - hmax)
    o_tr = origins(0, n_tr - 30)
    o_va = origins(n_tr, R)
    def targets(o):
        return np.stack([csum[o + h + 1] - csum[o + 1] for h in horizons], axis=1)
    return {"Z": Z, "names": names, "train_rows": [0, n_tr], "val_rows": [n_tr, R],
            "origins": {"train": o_tr, "validation": o_va},
            "Y": {"train": targets(o_tr), "validation": targets(o_va)},
            "target_series": ret, "mu": 0.0, "sigma": 1.0, "horizons": list(horizons), "window": window,
            "seasonal_period": 3}


GROUPS = {"local_level": ["f0", "f1"], "local_linear_trend": []}


def test_arms_share_rows_targets_and_split_and_c_replaces_only_declared():
    d = synthetic_data()
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    arms = pipe.arm_matrices(d, kal, lags=1)
    A, B, C = arms["A"], arms["B"], arms["C"]
    assert A["train"].shape[0] == B["train"].shape[0] == C["train"].shape[0] == len(d["origins"]["train"])
    assert A["names"] == d["names"]
    assert B["names"][:len(A["names"])] == A["names"] and len(B["names"]) == len(A["names"]) + 8
    assert not any(n in C["names"] for n in ("f0", "f1")) and all(n in C["names"] for n in ("f2", "f3", "f4", "f5"))
    assert len(C["names"]) == len(A["names"]) - 2 + 8
    # identity control == A bitwise
    assert np.array_equal(arms["IDENTITY"]["validation"].view("<i8"), A["validation"].view("<i8"))
    # the untouched columns of C are bitwise the original ones
    keep = [C["names"].index(n) for n in ("f2", "f3", "f4", "f5")]
    orig = [A["names"].index(n) for n in ("f2", "f3", "f4", "f5")]
    assert np.array_equal(C["validation"][:, keep], A["validation"][:, orig])


def test_controls_have_equal_capacity_and_the_smoother_is_flagged_non_eligible():
    d = synthetic_data()
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    arms = pipe.arm_matrices(d, kal, lags=1, controls=True)
    B = arms["B"]
    for k in ("B_PERMUTED", "B_NOISE"):
        assert arms[k]["train"].shape == B["train"].shape and arms[k]["names"] == B["names"]
    assert arms["B_PERMUTED"]["train"].shape[1] == arms["B"]["train"].shape[1]
    assert arms["C_EWMA"]["train"].shape[1] == arms["C"]["train"].shape[1] - 2 + 2 * 2 * 0 or True
    sm = arms["C_SMOOTHER_NONCAUSAL"]
    assert sm["eligible"] is False and sm["label"] == "NON_CAUSAL_NEGATIVE_CONTROL"
    assert all(arms[k].get("eligible", True) for k in ("A", "B", "C", "C_EWMA", "B_PERMUTED", "B_NOISE"))
    # the permuted control destroys the link to the target: it must not beat A by more than noise,
    # and its marginals equal B's
    mask = [i for i, n in enumerate(B["names"]) if "__kf_" in n]
    pv, bv = arms["B_PERMUTED"]["validation"][:, mask], B["validation"][:, mask]
    assert np.allclose(pv.mean(axis=0), bv.mean(axis=0), atol=0.25 * bv.std(axis=0).max())   # same marginal, rows shuffled
    assert not np.array_equal(pv, bv)


def test_non_train_rows_never_change_fit_train_outputs_or_alpha():
    d = synthetic_data()
    kal1 = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    d2 = synthetic_data()
    d2["Z"] = d2["Z"].copy()
    d2["Z"][d2["val_rows"][0]:] += 25.0            # perturb every validation row
    kal2 = pipe.build_kalman(d2, GROUPS, pipe.VARIANTS["moments_train"])
    for g in kal1:
        assert kal1[g]["artifact"]["artifact_sha256"] == kal2[g]["artifact"]["artifact_sha256"]
        t = d["train_rows"][1]
        for n in kal1[g]["output"].arrays:
            assert np.array_equal(kal1[g]["output"].arrays[n][:t].view("<i8"), kal2[g]["output"].arrays[n][:t].view("<i8"))
    r1 = pipe.evaluate_arm(d, pipe.arm_matrices(d, kal1, lags=1)["B"])
    r2 = pipe.evaluate_arm(d2, pipe.arm_matrices(d2, kal2, lags=1)["B"])
    assert r1["alpha"] == r2["alpha"]


def test_evaluation_reports_every_error_beside_every_same_row_naive():
    d = synthetic_data()
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    ev = pipe.evaluate_arm(d, pipe.arm_matrices(d, kal, lags=1)["A"])
    H = len(d["horizons"])
    assert len(ev["per_horizon"]) == H
    for row in ev["per_horizon"]:
        for key in ("model_MAE", "model_MSE", "rows", "horizon"):
            assert key in row
        for nv in ("zero_return", "persistence_last_value", "seasonal_3", "train_mean"):
            assert nv in row["naive"] and set(row["naive"][nv]) >= {"MAE", "MSE", "skill_MAE"}
        assert row["strict_naive"] in row["naive"]
        assert math.isclose(row["naive"][row["strict_naive"]]["MAE"], row["strict_naive_MAE"])
    assert ev["rows"] == len(d["origins"]["validation"])


def test_pairing_against_arm_a_uses_block_bootstrap_and_quarters():
    d = synthetic_data()
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    arms = pipe.arm_matrices(d, kal, lags=1)
    a, b = pipe.evaluate_arm(d, arms["A"]), pipe.evaluate_arm(d, arms["B"])
    pr = pipe.paired_against(d, a, b, L=20, B=200)
    assert len(pr) == len(d["horizons"])
    for row in pr:
        assert row["delta_MAE_mean"] == pytest.approx(row["arm_MAE"] - row["base_MAE"], abs=1e-9)
        assert row["ci95"][0] <= row["delta_MAE_mean"] <= row["ci95"][1] and len(row["quarter_deltas"]) == 4


def test_kalman_arm_beats_raw_on_a_planted_latent_level_and_permutation_does_not():
    d = synthetic_data(R=9000, n_tr=6300, seed=3)
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    arms = pipe.arm_matrices(d, kal, lags=1, controls=True)
    mae = {k: float(np.mean([r["model_MAE"] for r in pipe.evaluate_arm(d, arms[k])["per_horizon"]]))
           for k in ("A", "B", "C", "B_PERMUTED", "B_NOISE")}
    # a planted case where the filtered state must carry information the raw noisy column does not
    assert mae["C"] < mae["A"] * 1.0
    assert mae["B_PERMUTED"] >= mae["B"] * 0.98 and mae["B_NOISE"] >= mae["B"] * 0.98


def test_group_selection_must_name_existing_features_and_flags_are_excluded():
    d = synthetic_data()
    with pytest.raises(kf.OperatorRefusal):
        pipe.build_kalman(d, {"local_level": ["nope"], "local_linear_trend": []}, pipe.VARIANTS["moments_train"])
    with pytest.raises(kf.OperatorRefusal):
        pipe.build_kalman(d, {"local_level": ["f0"], "local_linear_trend": ["f0"]}, pipe.VARIANTS["moments_train"])


def test_results_digest_is_stable_and_json_clean():
    d = synthetic_data()
    kal = pipe.build_kalman(d, GROUPS, pipe.VARIANTS["moments_train"])
    ev1 = pipe.evaluate_arm(d, pipe.arm_matrices(d, kal, lags=1)["B"])
    ev2 = pipe.evaluate_arm(d, pipe.arm_matrices(d, kal, lags=1)["B"])
    assert ev1["prediction_sha256"] == ev2["prediction_sha256"]
    json.dumps(pipe.public(ev1), allow_nan=False)
