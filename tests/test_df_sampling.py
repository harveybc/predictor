"""C133: sampling regularity, gaps, Nyquist, decimation sensitivity and the
control-only aliasing rule. Synthetic arrays only."""
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


H = _load("test_df_profile_univariate_helpers_s", ROOT / "tests/test_df_profile_univariate.py")
S = _load("df_sampling", ROOT / "tools/df_sampling.py")
MOD = ROOT / "tools/df_sampling.py"
MIN = 60_000_000_000


def test_regular_segments_and_gap_detection_with_an_inserted_gap():
    T = 3000
    ts = np.arange(T, dtype=np.int64) * MIN
    ts[500:] += 30 * MIN                    # a 31-minute step inside train
    ts[2000:] += 5 * MIN                    # a 6-minute step inside calibration
    ts[2500:] += 7_000_000_000              # 67 s: 7 s off nominal, beyond the 1% tolerance but not a gap
    ct = H.contract(T, 1, meaning="PERIOD_START", period=60)
    rows = S.run_sampling(ct, np.random.default_rng(30).standard_normal((T, 1)), ts)
    g = lambda m, p: H.pick(rows, m, p)
    assert g("nominal_period_seconds", "train")["value"] == 60.0
    assert g("observed_median_period_seconds", "train")["value"] == 60.0
    assert g("gap_count", "train")["value"] == 1.0
    assert g("longest_gap_seconds", "train")["value"] == 31 * 60.0
    assert g("regular_segment_count", "train")["value"] == 2.0
    assert g("longest_regular_segment_samples", "train")["value"] == 1300.0
    assert g("gap_count", "calibration")["value"] == 1.0
    assert g("regular_segment_count", "calibration")["value"] == 2.0
    assert g("gap_count", "confirmation")["value"] == 0.0
    assert g("regular_segment_count", "confirmation")["value"] == 2.0
    assert g("longest_regular_segment_samples", "confirmation")["value"] == 500.0
    assert g("nyquist_frequency", "train")["value"] == 1 / 120.0
    assert g("jitter_mad_seconds", "train")["value"] == 0.0
    assert abs(g("coverage_vs_grid", "train")["value"] - 1800 / 1830) < 1e-12
    H.check_rows(rows, MOD)


def test_nyquist_needs_a_regular_segment():
    T = 400
    rng = np.random.default_rng(31)
    ts = np.cumsum(rng.choice([31, 47, 83, 130], size=T)).astype(np.int64) * 1_000_000_000  # never 60 s
    ct = H.contract(T, 1, meaning="INSTANT", period=60)
    rows = S.run_sampling(ct, rng.standard_normal((T, 1)), ts)
    row = H.pick(rows, "nyquist_frequency", "train")
    assert row["status"] == "INCONCLUSIVE" and row["reason"] == "NO_REGULAR_SEGMENT"


def test_single_frequency_without_control_is_not_identifiable_even_with_high_near_nyquist_energy():
    T = 20000
    rng = np.random.default_rng(32)
    x = np.sin(2 * np.pi * 0.47 * np.arange(T)) + 0.01 * rng.standard_normal(T)
    ct = H.contract(T, 1)
    rows = S.run_sampling(ct, x[:, None])
    for p in ("train", "calibration", "confirmation"):
        assert H.pick(rows, "near_nyquist_energy_fraction", p)["value"] > 0.9
        al = H.pick(rows, "aliasing_assessment", p)
        assert al["status"] == "INCONCLUSIVE" and al["value"] is None
        assert al["reason"] == "ALIASING_NOT_IDENTIFIABLE_WITHOUT_CONTROL"
    assert H.pick(rows, "decimation_variance_ratio", "train")["value"] > 10
    H.check_rows(rows, MOD)


def test_synthetic_sine_at_045_fs_folds_without_filter():
    rows = S.synthetic_aliasing_control(f0_fraction_of_fs=0.45, factor=2)
    folded = H.pick(rows, "folded_energy_fraction_unfiltered", "synthetic_control")
    assert folded["value"] > 0.9 and folded["reason"] == "FOLDED_ENERGY_AT_PREDICTED_FREQUENCY"
    assert abs(folded["estimator"]["params"]["predicted_alias_fraction_of_new_fs"] - 0.1) < 1e-12
    assert H.pick(rows, "filtered_over_unfiltered_alias_band_energy", "synthetic_control")["value"] < 0.01
    H.check_rows(rows, MOD)


def test_higher_frequency_source_control_shows_folded_energy():
    rng = np.random.default_rng(33)
    T = 12000
    high = np.sin(2 * np.pi * 0.45 * np.arange(2 * T)) + 0.05 * rng.standard_normal(2 * T)
    low = high[::2]                              # decimated by 2 without a filter
    ct = H.contract(T, 1)
    vid = ct["variables"][0]["variable_id"]
    rows = S.run_sampling(ct, low[:, None], controls={vid: {"x_high": high, "factor": 2}})
    for p in ("train", "calibration", "confirmation"):
        al = H.pick(rows, "aliasing_assessment", p)
        assert al["status"] == "COMPLETED" and al["reason"] == "ALIASING_CONSISTENT_WITH_CONTROL"
        assert al["value"] > 0.5
        assert H.pick(rows, "control_folded_energy_fraction", p)["value"] > 0.9
    # the same control with a properly low-pass source reports no folding
    smooth = np.convolve(rng.standard_normal(2 * T), np.ones(16) / 16, mode="same")
    rows2 = S.run_sampling(ct, smooth[::2][:, None], controls={vid: {"x_high": smooth, "factor": 2}})
    al2 = H.pick(rows2, "aliasing_assessment", "train")
    assert al2["reason"] == "NO_FOLDED_ENERGY_DETECTED_WITH_CONTROL"
    H.check_rows(rows + rows2, MOD)


def test_runs_helper():
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    reg = np.array([1, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert S.runs(valid, reg) == [(0, 3), (4, 5), (5, 8)]
