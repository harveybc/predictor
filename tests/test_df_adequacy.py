"""S1/S2 acceptance tests, declared before any adequacy measurement: analytic clean-sine
recurrence; label/row/horizon identity; train-only scaling; true forward held-out predictions;
future perturbation and restart parity; noise-only control; actual optimizer updates;
convergence/underfit diagnostics; complete failures and costs through the isolated child;
receptive field; reload parity; equal rows and labels across models."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("df_adequacy_design")
M = _load("df_adequacy_models")
RUN = _load("df_adequacy_run")

BANK = Path.home() / ".local/state/crispdm-data-foundation/synthetic_bank_c128_v1"
UNIT = D.UNITS[0]
HAVE_BANK = (BANK / UNIT / "clean_signal.npy").is_file()
TRAINING_FAST = {**D.TRAINING, "max_epochs": 30, "max_updates": 300}


def _fake_bank(tmp_path, period=40.0, amplitude=1.0, noise=0.3, n=2048, seed=0):
    """A bank unit like the real ones: clean sine + white noise, UNIT.json with the parameters."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    clean = amplitude * np.sin(2 * np.pi * t / period + 0.7)
    observed = clean + rng.normal(0, noise, n)
    d = tmp_path / "bank" / UNIT
    d.mkdir(parents=True)
    np.save(d / "clean_signal.npy", clean)
    np.save(d / "observed_signal.npy", observed)
    (d / "UNIT.json").write_text(json.dumps({"unit_id": UNIT, "n_samples": n, "clean_params": {"per_variable": [{"amplitude": amplitude, "period": period, "phase": 0.7}]},
                                            "noise_model": {"scale_per_variable": [noise]},
                                            "digests": {"observed_signal": hashlib.sha256(observed.tobytes()).hexdigest(),
                                                        "clean_signal": hashlib.sha256(clean.tobytes()).hexdigest()}}))
    return tmp_path / "bank"


def test_S1_the_design_is_sealed_with_boundaries_purge_coverage_and_the_review_table(tmp_path):
    bank = _fake_bank(tmp_path)
    (bank / D.UNITS[1]).mkdir()
    for f in ("clean_signal.npy", "observed_signal.npy", "UNIT.json"):
        (bank / D.UNITS[1] / f).write_bytes((bank / UNIT / f).read_bytes())
    doc = D.build(bank)
    assert doc["schema"] == D.DESIGN_SCHEMA and doc["cells_total"] == 2 * 3 * 3 * 4 * 3 * 1
    assert doc["design_sha256"] == D.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    b = doc["boundaries"]["W256__L768"]
    assert b["test"] == [1664, 2048] and b["purge"] == 257
    assert b["validation"][1] + b["purge"] == 1664 and b["train"][1] + b["purge"] == b["validation"][0]
    assert b["consumed_first_row"] >= 0
    with pytest.raises(ValueError):
        D.boundaries(256, 1024)                                                         # why 1024 is not in the design
    cov = doc["context_coverage"]["seed12__W256"]
    assert cov["covers_two_periods"] is True and cov["conv_receptive_field"] >= 256
    assert doc["context_coverage"]["seed12__W4"]["covers_two_periods"] is False
    assert [r["status"] for r in doc["review_table"]] == ["NOT_TESTED"] * 10
    assert {r["requirement"] for r in doc["review_table"]} >= {"question_estimand", "target_noise", "context_support", "data_sufficiency",
                                                               "model_adequacy", "temporal_validity", "comparison_fairness", "statistics",
                                                               "business_transfer", "independent_evidence"}
    assert doc["training"]["tuning_allowance"].startswith("NONE")


def test_S1_analytic_clean_sine_recurrence_and_label_row_horizon_identity(tmp_path):
    bank = _fake_bank(tmp_path)
    arrays = M.load_unit_arrays(bank, UNIT)
    clean = arrays["clean"]
    P = arrays["meta"]["period"]
    c = 2 * math.cos(2 * math.pi / P)
    assert np.allclose(clean[2:], c * clean[1:-1] - clean[:-2], atol=1e-9)             # the recurrence holds exactly
    x, label, baseline, oracle = M.task_series(arrays, "clean_next_level")
    t = 100
    assert label[t] == clean[t + 1] and baseline[t] == clean[t] and abs(oracle[t] - clean[t + 1]) < 1e-9
    x, label, baseline, oracle = M.task_series(arrays, "clean_increment")
    assert label[t] == clean[t + 1] - clean[t] and baseline[t] == 0.0 and abs(oracle[t] - label[t]) < 1e-9
    x, label, baseline, oracle = M.task_series(arrays, "observed_increment")
    obs = arrays["observed"]
    assert label[t] == obs[t + 1] - obs[t] and x is obs
    assert np.isnan(label[-1])                                                             # no label beyond the series
    W = M.windows(x, np.array([t]), 4)
    assert np.array_equal(W[0], obs[t - 3:t + 1])                                          # row t consumes the past up to t only


def test_S1_train_only_scaling_forward_held_out_and_equal_rows_across_models(tmp_path):
    bank = _fake_bank(tmp_path)
    arrays = M.load_unit_arrays(bank, UNIT)
    prep = M.prepare(arrays, "observed_increment", 8, 256)
    P = prep["parts"]
    assert P["train"]["rows"].max() < P["validation"]["rows"].min() - 8 < P["test"]["rows"].min() - 8
    assert P["test"]["rows"].min() == 1664 and P["test"]["rows"].max() == 2046
    xm = P["train"]["X"].mean()
    assert prep["scale"]["x_mean"] == pytest.approx(xm)                                     # not the test's mean
    assert prep["scale"]["x_mean"] != pytest.approx(P["test"]["X"].mean())
    # every model of a cell receives exactly these rows and labels: the preparation is model-free
    prep2 = M.prepare(arrays, "observed_increment", 8, 256)
    assert np.array_equal(prep["parts"]["test"]["rows"], prep2["parts"]["test"]["rows"])
    assert np.array_equal(prep["parts"]["test"]["y"], prep2["parts"]["test"]["y"])


def test_S2_ridge_learns_the_clean_recurrence_exactly_with_two_lags(tmp_path):
    bank = _fake_bank(tmp_path, noise=0.0)
    arrays = M.load_unit_arrays(bank, UNIT)
    prep = M.prepare(arrays, "clean_next_level", 4, 512)
    P, s = prep["parts"], prep["scale"]
    r = M.Ridge(1e-9)
    r.fit(M._scale_x(P["train"]["X"], s), M._scale_y(P["train"]["y"], s), M._scale_x(P["validation"]["X"], s), M._scale_y(P["validation"]["y"], s),
          seed=0, training=D.TRAINING)
    pred = M._unscale_y(r.predict(M._scale_x(P["test"]["X"], s)), s)
    assert M.mae(pred, P["test"]["y"]) < 1e-6 * arrays["meta"]["amplitude"]              # the special case is solvable by two lags


def test_S2_future_perturbation_invariance_and_restart_parity(tmp_path):
    bank = _fake_bank(tmp_path)
    arrays = M.load_unit_arrays(bank, UNIT)
    prep = M.prepare(arrays, "observed_increment", 8, 256)
    P, s = prep["parts"], prep["scale"]
    r = M.Ridge(1.0)
    r.fit(M._scale_x(P["train"]["X"], s), M._scale_y(P["train"]["y"], s), M._scale_x(P["validation"]["X"], s), M._scale_y(P["validation"]["y"], s),
          seed=0, training=D.TRAINING)
    rows = P["test"]["rows"][:50]
    x = arrays["observed"].copy()
    before = r.predict(M._scale_x(M.windows(x, rows, 8), s))
    x[rows[-1] + 1:] += 100.0                                                              # the future is perturbed
    after = r.predict(M._scale_x(M.windows(x, rows, 8), s))
    assert np.array_equal(before, after)
    lstm = M.LSTMModel(8)
    lstm.fit(M._scale_x(P["train"]["X"][:64], s), M._scale_y(P["train"]["y"][:64], s), M._scale_x(P["validation"]["X"], s),
             M._scale_y(P["validation"]["y"], s), seed=1, training={**TRAINING_FAST, "max_updates": 4})
    X = M._scale_x(P["test"]["X"][:40], s)
    whole = lstm.predict(X)
    chunked = np.concatenate([lstm.predict(X[:20]), lstm.predict(X[20:])])
    assert np.allclose(whole, chunked, atol=1e-6)                                          # RESET_PER_WINDOW: no carried state


def test_S2_conv_receptive_field_optimizer_updates_diagnosis_and_reload_parity(tmp_path):
    bank = _fake_bank(tmp_path)
    arrays = M.load_unit_arrays(bank, UNIT)
    assert D.conv_dilations(4) == [1, 2] and D.receptive_field(3, [1, 2]) == 7 and D.conv_dilations(8) == [1, 2, 4]
    assert D.receptive_field(3, D.conv_dilations(256)) >= 256
    prep = M.prepare(arrays, "clean_increment", 8, 256)
    P, s = prep["parts"], prep["scale"]
    conv = M.CausalConv1D(8)
    fit = conv.fit(M._scale_x(P["train"]["X"], s), M._scale_y(P["train"]["y"], s), M._scale_x(P["validation"]["X"], s),
                   M._scale_y(P["validation"]["y"], s), seed=3, training=TRAINING_FAST)
    assert fit["updates"] > 0 and fit["weight_change_norm"] > 0 and fit["receptive_field"] == 15
    assert fit["updates"] <= TRAINING_FAST["max_updates"] and len(fit["curve"]["train"]) == fit["epochs"]
    assert [l["type"] for l in fit["layers"]][:3] == ["Conv1D", "Conv1D", "Conv1D"] and fit["layers"][-1]["type"] == "Dense"
    pred = conv.predict(M._scale_x(P["test"]["X"], s))
    conv.save(tmp_path / "w.weights.h5")
    again = M.CausalConv1D(8)
    again.load(tmp_path / "w.weights.h5", seed=3)
    assert np.allclose(again.predict(M._scale_x(P["test"]["X"], s)), pred, atol=1e-6)      # reload parity
    base = M.mae(M._scale_y(P["train"]["baseline"], s), M._scale_y(P["train"]["y"], s))
    assert M.diagnose(fit, base)["class"] in (M.FITTED, M.UNDERFIT, M.OVERFIT)
    assert M.diagnose({"updates": 0, "weight_change_norm": 0.0, "curve": {"train": [1.0], "validation": [1.0]}}, 1.0)["class"] == M.OPT_FAIL
    assert M.diagnose({"updates": 5, "weight_change_norm": 1.0, "curve": {"train": [1.0] * 12, "validation": [1.0] * 12}}, 1.0)["class"] == M.OPT_FAIL
    assert M.diagnose({"updates": 5, "weight_change_norm": 1.0, "curve": {"train": [1.0, 0.98], "validation": [1.0, 0.99]}}, 1.0)["class"] == M.UNDERFIT
    assert M.diagnose({"updates": 5, "weight_change_norm": 1.0, "curve": {"train": [1.0, 0.5, 0.2, 0.1], "validation": [1.0, 0.5, 0.7, 0.9]}}, 1.0)["class"] == M.OVERFIT


def test_S2_noise_only_control_has_no_skill(tmp_path):
    rng = np.random.default_rng(9)
    n = 2048
    bank = _fake_bank(tmp_path, amplitude=0.0, noise=1.0, n=n)                             # pure white noise
    arrays = M.load_unit_arrays(bank, UNIT)
    prep = M.prepare(arrays, "observed_increment", 8, 1024)
    P, s = prep["parts"], prep["scale"]
    r = M.Ridge(1.0)
    r.fit(M._scale_x(P["train"]["X"], s), M._scale_y(P["train"]["y"], s), M._scale_x(P["validation"]["X"], s), M._scale_y(P["validation"]["y"], s),
          seed=0, training=D.TRAINING)
    pred = M._unscale_y(r.predict(M._scale_x(P["test"]["X"], s)), s)
    skill = 1 - M.mae(pred, P["test"]["y"]) / M.mae(P["test"]["baseline"], P["test"]["y"])
    # increments of a random walk from white-noise levels are MA(1): a little skill is genuine;
    # a control claiming a large one would be a defect
    assert skill < 0.35


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_S2_a_cell_through_the_isolated_child_records_everything_and_recomputes_its_losses(tmp_path):
    bank = _fake_bank(tmp_path)
    cell = {"cell_id": "seed12__observed_increment__ridge__W8__L256__s1", "unit": UNIT, "task": "observed_increment", "model": "ridge",
            "window": 8, "train_length": 256, "seed": 1}
    design = {"horizon": 1, "training": D.TRAINING, "design_sha256": "d" * 64}
    job = RUN._job(design, cell, bank, "t", role="CELL")
    out = RUN.run_isolated(job, attempt_dir=tmp_path / "a", assigned_bytes=2 << 30, wall_seconds=300.0, cpu_seconds=300)
    assert out["outcome"] in (M.FITTED, M.UNDERFIT, M.OVERFIT), out
    rec = out["score"]
    assert rec["losses"]["test"]["rows"] == 383 and rec["consumed_span_over_P"] == pytest.approx(7 / 40.0)
    assert (tmp_path / "a" / "arrays.npz").is_file() and rec["arrays_sha256"]
    arr = np.load(tmp_path / "a" / "arrays.npz")
    assert M.mae(arr["test_pred"], arr["test_y"]) == pytest.approx(rec["losses"]["test"]["model"])
    assert out["cost"]["cpu_seconds"] > 0
    # altered arrays are refused on resume (one truthful outcome)
    np.savez(tmp_path / "a" / "arrays.npz", **{k: arr[k] * (1.5 if k == "test_pred" else 1) for k in arr})
    again = RUN.run_isolated(job, attempt_dir=tmp_path / "a", assigned_bytes=2 << 30, wall_seconds=300.0, cpu_seconds=300)
    assert again["outcome"] == RUN.SCORE_UNVERIFIED and again["score"] is None


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_S3_a_ceiling_hit_is_a_complete_failure_with_cost_and_no_partial_score(tmp_path):
    bank = _fake_bank(tmp_path)
    cell = {"cell_id": "seed12__observed_increment__lstm__W256__L1024__s1", "unit": UNIT, "task": "observed_increment", "model": "lstm",
            "window": 256, "train_length": 768, "seed": 1}
    design = {"horizon": 1, "training": D.TRAINING, "design_sha256": "d" * 64}
    out = RUN.run_isolated(RUN._job(design, cell, bank, "t"), attempt_dir=tmp_path / "f", assigned_bytes=2 << 30, wall_seconds=8.0, cpu_seconds=8)
    assert out["outcome"] == RUN.H.RESOURCE_EXCEEDED and out["score"] is None and out["cost"]["wall_seconds"] is not None, out
