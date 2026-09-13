"""C131: operational information and compression descriptors under train-frozen
quantization. Synthetic arrays only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("test_df_profile_univariate_helpers", ROOT / "tests/test_df_profile_univariate.py")
I = _load("df_profile_information", ROOT / "tools/df_profile_information.py")
MOD = ROOT / "tools/df_profile_information.py"


def test_permutation_entropy_white_noise_near_one_and_sine_low():
    rng = np.random.default_rng(10)
    T = 12000
    noise = rng.standard_normal(T)
    sine = np.sin(2 * np.pi * np.arange(T) / 200.0)
    ct = H.contract(T, 2)
    rows = I.run_information(ct, np.column_stack([noise, sine]))
    vn, vs = (v["variable_id"] for v in ct["variables"])
    for p in ("train", "calibration", "confirmation"):
        for m in (3, 4, 5):
            assert H.pick(rows, f"permutation_entropy_order_{m}", p, vn)["value"] > 0.97
            assert H.pick(rows, f"permutation_entropy_order_{m}", p, vs)["value"] < 0.45
    H.check_rows(rows, MOD)


def test_compression_temporal_gain_positive_for_structure_and_near_zero_for_noise():
    rng = np.random.default_rng(11)
    T = 6000
    noise = rng.standard_normal(T)
    walk = np.cumsum(rng.standard_normal(T))
    ct = H.contract(T, 2)
    rows = I.run_information(ct, np.column_stack([noise, walk]))
    vn, vw = (v["variable_id"] for v in ct["variables"])
    for c in ("zlib9", "lzma6"):
        g_noise = H.pick(rows, f"temporal_structure_gain_{c}_symbols", "train", vn)["value"]
        g_walk = H.pick(rows, f"temporal_structure_gain_{c}_symbols", "train", vw)["value"]
        assert abs(g_noise) < 0.05 and g_walk > 0.3
        assert H.pick(rows, f"compressed_bits_per_sample_{c}_raw_float64", "train", vn)["value"] > 0
    # a declared entropy bound: 16 bins on train quantiles of noise gives about 4 bits on train
    assert abs(H.pick(rows, "discrete_entropy_bits", "train", vn)["value"] - 4.0) < 0.01


def test_conditional_redundancy_and_insufficient_sample():
    rng = np.random.default_rng(12)
    T = 20000
    ar = H.ar1(T, 0.9, rng)
    noise = rng.standard_normal(T)
    ct = H.contract(T, 2)
    rows = I.run_information(ct, np.column_stack([ar, noise]))
    va, vn = (v["variable_id"] for v in ct["variables"])
    assert H.pick(rows, "conditional_redundancy_bits_lag1", "train", va)["value"] > 0.8
    assert H.pick(rows, "conditional_redundancy_bits_lag1", "train", vn)["value"] < 0.05
    small = H.contract(400, 1)
    r2 = I.run_information(small, rng.standard_normal((400, 1)))
    row = H.pick(r2, "conditional_redundancy_bits_lag1", "confirmation")
    assert row["status"] == "INCONCLUSIVE" and row["reason"] == "INSUFFICIENT_SAMPLE" and row["value"] is None


def test_train_frozen_bins_confirmation_changes_do_not_touch_train_rows():
    rng = np.random.default_rng(13)
    T = 4000
    X = rng.standard_normal((T, 2))
    ct = H.contract(T, 2)
    s = ct["partitions"]["boundaries"]["confirmation"][0]
    Y = X.copy()
    Y[s:] = Y[s:] * 5 - 2
    strip = lambda rows, p: [{k: v for k, v in r.items() if k != "cpu_seconds"} for r in rows if r["partition"] == p]
    rx, ry = I.run_information(ct, X), I.run_information(ct, Y)
    assert strip(rx, "train") == strip(ry, "train")
    assert strip(rx, "calibration") == strip(ry, "calibration")
    # the confirmation entropy moved because the bins stayed frozen at train quantiles
    vid = ct["variables"][0]["variable_id"]
    assert H.pick(rx, "discrete_entropy_bits", "confirmation", vid)["value"] != \
        H.pick(ry, "discrete_entropy_bits", "confirmation", vid)["value"]


def test_effective_rank_on_rank_two_matrices():
    T = 4000
    t = np.arange(T)
    s, c = np.sin(2 * np.pi * t / 100), np.cos(2 * np.pi * t / 100)
    er, nr, _ = I.effective_rank(np.column_stack([s, c]) / np.sqrt(T / 2))
    assert abs(er - 2.0) < 1e-6 and nr == 2
    rng = np.random.default_rng(14)
    A = rng.standard_normal((T, 2))
    B = rng.standard_normal((2, 5))
    ct = H.contract(T, 5)
    rows = I.run_information(ct, A @ B)
    row = H.pick(rows, "effective_rank", "train")
    assert row["group_id"] == "train_matrix_all_variables"
    assert 1.0 < row["value"] <= 2.0 + 1e-6
    assert H.pick(rows, "numerical_rank", "train")["value"] == 2.0
    assert "Roy & Vetterli" in row["estimator"]["params"]["reference"]
    assert not [r for r in rows if r["metric"] == "effective_rank" and r["partition"] != "train"]
    H.check_rows(rows, MOD)


def test_spectral_entropy_orders_noise_above_sine():
    rng = np.random.default_rng(15)
    T = 5000
    ct = H.contract(T, 2)
    X = np.column_stack([rng.standard_normal(T), np.sin(2 * np.pi * np.arange(T) / 16)])
    rows = I.run_information(ct, X)
    vn, vs = (v["variable_id"] for v in ct["variables"])
    assert H.pick(rows, "spectral_entropy_median", "train", vn)["value"] > 0.9
    assert H.pick(rows, "spectral_entropy_median", "train", vs)["value"] < 0.5
