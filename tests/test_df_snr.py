"""C134 tests: noise / SNR estimators calibrated against known truth.

Self-contained: a tiny generator lives in this file; tools/df_synthetic_bank.py
is NOT imported and no real data root is read.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
import df_snr as d  # noqa: E402

SMALL_BS = dict(d.DEFAULT_BOOTSTRAP, B=20)


def _ar1(n, phi, rng):
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0] / math.sqrt(1 - phi ** 2)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y * math.sqrt(1 - phi ** 2)  # unit marginal variance


def _sine(n, period=500.0):
    return np.sin(2 * np.pi * np.arange(n) / period)  # variance 0.5


def _noise_at_snr(clean, snr_db, rng, kind="white"):
    var = np.var(clean) / 10 ** (snr_db / 10)
    base = rng.standard_normal(len(clean)) if kind == "white" else _ar1(len(clean), 0.8, rng)
    return base * math.sqrt(var)


# --------------------------------------------------------------------------
def test_declarations_are_data():
    decl = d.estimator_declarations()
    assert set(decl) == {"mad_first_difference", "wavelet_mad", "ar_residual", "spectral_floor",
                         "local_level_kalman", "trailing_median_residual"}
    for name, spec in decl.items():
        for k in ("name", "parameters", "decomposition_model", "assumptions", "bootstrap", "fitted_on"):
            assert k in spec, (name, k)
        assert spec["bootstrap"]["scheme"] == "moving_block"
        json.dumps(spec)  # serializable
    assert "MODEL-CONDITIONAL" in " ".join(decl["ar_residual"]["assumptions"])


@pytest.mark.parametrize("name", ["mad_first_difference", "wavelet_mad"])
def test_sine_white_known_snr_long_n(name):
    rng = np.random.default_rng(1)
    n = 20000
    clean = _sine(n)
    noise = _noise_at_snr(clean, 0.0, rng)
    x = clean + noise
    true_snr = 10 * math.log10(np.var(clean) / np.var(noise))
    r = d.estimate(x, (0, n), name, do_bootstrap=False)
    assert r["status"] == d.STATUS_OK
    assert abs(math.log(r["noise_variance"] / np.var(noise))) < 0.05
    assert abs(r["snr_db"] - true_snr) < 0.5


def test_ar1_colored_noise_biases_mad_first_difference_low():
    """Documented direction: with positively autocorrelated noise (phi=0.8),
    var(diff(e)) = 2*sigma^2*(1-phi), so the first-difference MAD estimator
    UNDER-estimates noise variance (by ~ factor 1-phi) and OVER-states SNR."""
    rng = np.random.default_rng(2)
    n = 20000
    clean = _sine(n)
    noise = _noise_at_snr(clean, 0.0, rng, kind="ar1")
    true_snr = 10 * math.log10(np.var(clean) / np.var(noise))
    r = d.estimate(clean + noise, (0, n), "mad_first_difference", do_bootstrap=False)
    assert r["status"] == d.STATUS_OK
    assert r["noise_variance"] < 0.5 * np.var(noise)
    assert r["snr_db"] > true_snr + 3.0


def test_null_noise_no_division_errors():
    n = 4096
    with np.errstate(all="raise"):
        for x in (_sine(n), np.repeat([0.0, 1.0, -1.0, 2.0], n // 4)):
            for name in d.ESTIMATORS:
                if name == "local_level_kalman":
                    continue  # statsmodels internals are not errstate-clean; checked below
                r = d.estimate(x, (0, n), name, bootstrap=SMALL_BS)
                assert r["status"] in (d.STATUS_OK, d.STATUS_NI)
                if r["status"] == d.STATUS_OK:
                    assert r["snr_db"] > 20.0 and math.isfinite(r["snr_db"])
                else:
                    assert r["snr_db"] is None and r["reason"]
    step = d.estimate(np.repeat([0.0, 1.0, -1.0, 2.0], n // 4), (0, n), "mad_first_difference")
    assert step["status"] == d.STATUS_NI
    assert step["reason"].startswith("noise_variance_at_or_below_numerical_floor")
    r = d.estimate(_sine(1024), (0, 1024), "local_level_kalman", do_bootstrap=False)
    assert r["status"] == d.STATUS_NI or r["snr_db"] > 20.0


def test_constant_series_is_not_identifiable():
    r = d.estimate(np.ones(512), (0, 512), "wavelet_mad")
    assert r["status"] == d.STATUS_NI and r["reason"] == "total_variance_nonpositive"


def test_null_signal_very_negative_or_not_identifiable():
    rng = np.random.default_rng(3)
    n = 4096
    x = rng.standard_normal(n)
    for name in d.ESTIMATORS:
        if name == "local_level_kalman":
            x_use, nn = x[:1024], 1024
        else:
            x_use, nn = x, n
        r = d.estimate(x_use, (0, nn), name, do_bootstrap=False)
        if r["status"] == d.STATUS_OK:
            assert r["snr_db"] < -3.0, (name, r["snr_db"])
        else:
            assert r["noise_variance"] is None and r["signal_variance"] is None and r["snr_db"] is None
            assert r["reason"]


def test_signal_variance_nonpositive_rule():
    rng = np.random.default_rng(4)
    # pure white noise: wavelet MAD often exceeds var(x) -> must be NI, never a negative variance value
    hits = 0
    for seed in range(20):
        x = np.random.default_rng(seed).standard_normal(2048)
        r = d.estimate(x, (0, 2048), "wavelet_mad", do_bootstrap=False)
        if r["status"] == d.STATUS_NI:
            assert r["reason"].startswith("signal_variance_nonpositive")
            assert r["signal_variance"] is None
            hits += 1
        else:
            assert r["signal_variance"] > 0
    assert hits > 0
    del rng


def test_bootstrap_coverage_white_noise_loose_band():
    covered = 0
    reps = 30
    for k in range(reps):
        rng = np.random.default_rng(100 + k)
        n = 2048
        clean = np.sin(2 * np.pi * (np.arange(n) + rng.uniform(0, 500)) / 500)
        noise = _noise_at_snr(clean, 3.0, rng)
        true_snr = 10 * math.log10(np.var(clean) / np.var(noise))
        r = d.estimate(clean + noise, (0, n), "mad_first_difference",
                       bootstrap=dict(d.DEFAULT_BOOTSTRAP, B=100, seed=k))
        b = r["bootstrap"]
        assert b["B"] == 100 and b["block_length"] == 50 and b["seed"] == k
        covered += b["ci_low_db"] <= true_snr <= b["ci_high_db"]
    assert 0.6 <= covered / reps <= 1.0


def test_train_only_confirmation_changes_do_not_move_estimates():
    rng = np.random.default_rng(5)
    n = 3000
    x = _sine(n) + rng.standard_normal(n) * 0.5
    y = x.copy()
    y[2000:] = rng.standard_normal(1000) * 50 + 1e3  # rewrite confirmation data
    y[2500] = np.nan
    for name in d.ESTIMATORS:
        a = d.estimate(x, (0, 2000), name, bootstrap=SMALL_BS)
        b = d.estimate(y, (0, 2000), name, bootstrap=SMALL_BS)
        assert a["noise_variance"] == b["noise_variance"], name
        assert a["snr_db"] == b["snr_db"], name
        assert (a["bootstrap"] or {}).get("ci_low_db") == (b["bootstrap"] or {}).get("ci_low_db"), name


def test_real_data_label_and_model_name():
    rng = np.random.default_rng(6)
    x = np.vstack([_sine(1024) + rng.standard_normal(1024) * 0.3, rng.standard_normal(1024)])
    rows = d.estimate_real(x, (0, 700), variable_names=["a", "b"],
                           estimators=["mad_first_difference", "ar_residual"], bootstrap=SMALL_BS)
    assert len(rows) == 4
    for r in rows:
        assert r["label"] == "MODEL_CONDITIONAL_SNR_ESTIMATE"
        assert r["decomposition_model"] == d.ESTIMATORS[r["estimator"]]["decomposition_model"]
        assert r["assumptions"]
        assert "snr_db" not in r and "snr" not in r  # never a bare SNR key
        assert len(r["code_sha256"]) == 64
        json.dumps(r, allow_nan=False)


def test_nan_uses_longest_complete_segment_or_refuses():
    rng = np.random.default_rng(7)
    n = 2000
    x = _sine(n) + rng.standard_normal(n) * 0.5
    xn = x.copy()
    xn[300] = np.nan
    xn[1200:1210] = np.nan
    for name in d.ESTIMATORS:
        r = d.estimate(xn, (0, 1500), name, bootstrap=SMALL_BS)
        assert r["segment_used"] == [301, 1200]  # longest of [0,300), [301,1200), [1210,1500)
        assert r["n_missing_in_train"] == 11
        ref = d.estimate(x[301:1200], (0, 899), name, bootstrap=SMALL_BS)
        assert r["noise_variance"] == ref["noise_variance"], name  # same data, no interpolation
    # mask-based missingness is honoured the same way
    mask = np.zeros(n, bool)
    mask[300] = True
    mask[1200:1210] = True
    rm = d.estimate(x, (0, 1500), "wavelet_mad", missing_mask=mask, do_bootstrap=False)
    assert rm["segment_used"] == [301, 1200]
    # fragmented train: every complete segment too short -> refuse
    xf = x.copy()
    xf[::50] = np.nan
    rf = d.estimate(xf, (0, 1500), "mad_first_difference")
    assert rf["status"] == d.STATUS_NI and rf["reason"].startswith("insufficient_complete_segment")
    # all-NaN train
    ra = d.estimate(np.full(500, np.nan), (0, 500), "spectral_floor")
    assert ra["status"] == d.STATUS_NI


# --------------------------------------------------------------------------
def _write_units(root, n=512):
    """6 units: 3 in the C128 'partitions' layout, 3 in the T1 'temporal_roles' layout."""
    specs = [
        ("partitions", "sine", "white", 10, 1, "VT", 2, False),
        ("partitions", "sine", "colored", 10, 2, "TV", 2, False),
        ("partitions", "sine", "white", 0, 3, "VT", 1, True),
        ("temporal_roles", "sine", "white", 10, 4, "VT", 1, False),
        ("temporal_roles", "sine", "colored", 0, 5, "VT", 1, False),
        ("temporal_roles", "sine", "white", 0, 6, "TV", 3, False),
    ]
    for layout, fam, pert, snr, seed, orient, V, missing in specs:
        rng = np.random.default_rng(seed)
        clean = np.vstack([np.sin(2 * np.pi * (np.arange(n) + 37 * v) / 128) for v in range(V)])
        kind = "white" if pert == "white" else "ar1"
        noise = np.vstack([_noise_at_snr(clean[v], snr, rng, kind) for v in range(V)])
        obs = clean + noise
        uid = f"{fam}__{pert}__snr{snr}__{layout}__seed{seed}"
        ud = os.path.join(root, uid)
        os.makedirs(ud)
        meta = {"unit_id": uid, "family": fam, "perturbation": pert, "declared_snr_db": snr,
                "n_samples": n, "seed": seed,
                layout: {"train": [0, 320], "validation": [320, 416], "confirmation": [416, n]}}
        if missing:
            mask = np.zeros_like(obs, bool)
            mask[:, 100:104] = True
            obs = obs.copy()
            obs[mask] = np.nan
            np.save(os.path.join(ud, "missing_mask.npy"), mask if orient == "VT" else mask.T)
        for k, arr in (("clean_signal", clean), ("additive_noise", noise), ("observed_signal", obs)):
            np.save(os.path.join(ud, k + ".npy"), arr if orient == "VT" else arr.T)
        with open(os.path.join(ud, "UNIT.json"), "w") as fh:
            json.dump(meta, fh)


def test_calibration_runner_both_layouts_write_once(tmp_path, capsys):
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_units(str(bank))
    out = tmp_path / "cal.json"
    rc = d.main(["--bank", str(bank), "--out", str(out), "--bootstrap-b", "20"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == "crispdm.data_foundation.snr_calibration.v1"
    assert doc["unit_count"] == 6
    assert len(doc["code_sha256"]) == 64 and doc["code_sha256"] == d.code_sha256()
    assert set(doc["estimators"]) == set(d.ESTIMATORS)
    assert doc["record_count"] == 10 * 6  # (2+2+1+1+1+3) variables x 6 estimators
    assert doc["least_biased_per_perturbation"]["label"] == "DESCRIPTIVE_ONLY_NOT_A_SELECTION"
    keys = {(g["estimator"], g["perturbation"], g["declared_snr_db"], g["length"]) for g in doc["grouped_table"]}
    assert ("mad_first_difference", "colored", 0, 512) in keys
    for g in doc["grouped_table"]:
        for k in ("bias_log_noise_var", "rmse_log_noise_var", "bias_snr_db", "rmse_snr_db",
                  "ci_coverage_true_snr", "not_identifiable_rate", "failure_rate"):
            assert k in g
    rows_path = d.rows_path_for(str(out))
    rows = [json.loads(line) for line in open(rows_path)]
    assert len(rows) == doc["olap_rows"]["count"]
    for r in rows[:50]:
        assert set(r) == {"unit_id", "variable_index", "estimator", "metric", "value", "status",
                          "reason", "code_sha256"}
    # missing-mask unit: truth and estimate on the post-gap segment [104, 320)
    miss = [r for r in rows if "seed3" in r["unit_id"] and r["metric"] == "snr_db"]
    assert miss
    # white-noise MAD estimators should be close in this smoke (loose)
    white_mad = [g for g in doc["grouped_table"]
                 if g["estimator"] == "mad_first_difference" and g["perturbation"] == "white"]
    assert all(abs(g["bias_snr_db"]) < 3.0 for g in white_mad if g["bias_snr_db"] is not None)
    colored_mad = [g for g in doc["grouped_table"]
                   if g["estimator"] == "mad_first_difference" and g["perturbation"] == "colored"]
    assert all(g["bias_log_noise_var"] < 0 for g in colored_mad)
    # smoke report of bias / RMSE per estimator (visible with pytest -s)
    for est in d.ESTIMATORS:
        gs = [g for g in doc["grouped_table"] if g["estimator"] == est]
        print("CAL_SMOKE", est, json.dumps([(g["perturbation"], g["declared_snr_db"], g["bias_snr_db"],
                                              g["rmse_snr_db"], g["bias_log_noise_var"],
                                              g["rmse_log_noise_var"], g["not_identifiable_rate"])
                                             for g in gs]))
    # write-once: both CLI and API refuse to overwrite
    before = out.read_bytes()
    assert d.main(["--bank", str(bank), "--out", str(out), "--limit", "1"]) == 2
    with pytest.raises(FileExistsError):
        d.write_calibration({"x": 1}, [], str(out))
    assert out.read_bytes() == before
    # --limit smoke
    out2 = tmp_path / "cal2.json"
    assert d.main(["--bank", str(bank), "--out", str(out2), "--limit", "2", "--bootstrap-b", "10"]) == 0
    assert json.loads(out2.read_text())["unit_count"] == 2


def test_orientation_detection_and_ambiguity():
    a = np.zeros((3, 100))
    assert d._orient(a, 100, "x").shape == (3, 100)
    assert d._orient(a.T, 100, "x").shape == (3, 100)
    with pytest.raises(ValueError):
        d._orient(np.zeros((100, 100)), 100, "x")
