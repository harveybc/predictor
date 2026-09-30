"""Behavioral specification for staged, train-only representation pretraining.

All arrays here are SYNTHETIC fixtures; these are component checks, never
forecasting results.
"""
import copy
import json
import time

import numpy as np
import pytest

from tools.modular_pretrain import (Heartbeat, pretrain_components, regime_config,
                                    run_synthetic_pilot)

FIT = {"max_epochs": 2, "patience": 1, "batch_size": 4, "learning_rate": 0.001, "loss": "mse"}


def populations():
    return ({"split": "train", "dataset_id": "synthetic", "support_start": 0, "provenance": "synthetic_fixture",
             "support_end": 86400, "time_unit": "seconds"},
            {"split": "train_validation", "dataset_id": "synthetic", "provenance": "synthetic_fixture",
             "support_start": 90000, "support_end": 172800, "time_unit": "seconds"})


@pytest.mark.parametrize("mutation", [
    lambda t, v: v.update(split="test"),
    lambda t, v: v.update(split="holdout"),
    lambda t, v: v.update(split="validation"),
    lambda t, v: v.update(support_start=86400),
    lambda t, v: v.update(dataset_id="other"),
    lambda t, v: v.update(provenance="local_file"),
    lambda t, v: t.update(support_end=True),
])
def test_rejects_bad_split_before_model_load(tmp_path, mutation):
    t, v = populations()
    mutation(t, v)
    with pytest.raises(ValueError):
        pretrain_components({}, np.zeros((2, 24, 2)), np.zeros((2, 24, 2)),
                            tmp_path / "run", {}, t, v)
    assert not (tmp_path / "run").exists()


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    from predictor_plugins.modular_temporal import default_config
    config = default_config(["a", "b"])
    original = copy.deepcopy(config)
    rng = np.random.default_rng(91)
    train = rng.normal(size=(8, 24, 2)).astype("float32")
    val = rng.normal(size=(4, 24, 2)).astype("float32")
    t, v = populations()
    out = tmp_path_factory.mktemp("pre") / "run"
    result = pretrain_components(config, train, val, out, FIT, t, v, seed=3)
    assert config == original
    return {"result": result, "train": train, "val": val, "out": out, "config": original}


def test_all_stages_export_reload_and_real_updates(pretrained):
    from predictor_plugins.modular_temporal import build_modular
    result, train, val = pretrained["result"], pretrained["train"], pretrained["val"]
    assert result["schema"] == "modular.pretrain.v2"
    assert result["provenance"] == "synthetic_fixture" and result["label"].startswith("SYNTHETIC")
    assert len(result["branches"]) == 2
    for stage in [*result["branches"], result["core"]]:
        t = stage["training"]
        assert t["observed_updates"] > 0 and t["stop_reason"] in ("patience", "max_epochs")
        assert t["restored_weights_sha256"] == next(
            h["weights_sha256"] for h in t["history"] if h["epoch"] == t["selected_epoch"])
        assert stage["reload_parity"]["passed"]
        for split in ("train", "train_validation"):
            rec = stage["reconstruction"][split]
            # Lossy: an exact zero is not expected, and the reference is reported.
            assert rec["MSE"] > 0 and rec["reference_MSE"] > 0 and rec["relative_MSE"] is not None
    fitted = build_modular(result["fine_tune_config"])
    assert tuple(fitted.encoder_model(train[:1]).shape) == (1, 6, 8)
    frozen = build_modular(regime_config(result["fine_tune_config"], "R1"))
    np.testing.assert_allclose(fitted.encoder_model(val), frozen.encoder_model(val), atol=1e-6)
    assert not frozen.core_model.trainable_weights
    assert (pretrained["out"] / "PRETRAIN.json").exists()
    assert result["fused_train"]["shape"] == [8, 12, 32]


def test_right_edge_grids_and_materialized_row_alignment(pretrained):
    from predictor_plugins.modular_temporal import build_modular
    result, train = pretrained["result"], pretrained["train"]
    grids = result["grids"]
    assert grids["input_right_edges_hours"] == list(range(1, 25))
    assert grids["branch_right_edges_hours"] == list(range(2, 25, 2))
    assert grids["core_right_edges_hours"] == list(range(4, 25, 4))
    assert result["fusion"]["right_edge_grid_hours"] == grids["branch_right_edges_hours"]
    fixed = build_modular(regime_config(result["fine_tune_config"], "R1"))
    fused = np.load(pretrained["out"] / "fused_train.npy")
    for row in (0, 5, 7):  # row i of the materialization is window i, not a shape coincidence
        np.testing.assert_allclose(fused[row], np.asarray(fixed.fusion_model(train[row:row + 1]))[0], atol=1e-6)
    # Right-edge causality: perturbing only the last observation moves only the last block.
    changed = train[:1].copy()
    changed[:, -1, :] += 5.0
    a, b = np.asarray(fixed.fusion_model(train[:1])), np.asarray(fixed.fusion_model(changed))
    np.testing.assert_allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert np.abs(a[:, -1] - b[:, -1]).max() > 1e-4
    la, lb = np.asarray(fixed.encoder_model(train[:1])), np.asarray(fixed.encoder_model(changed))
    np.testing.assert_allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert np.abs(la[:, -1] - lb[:, -1]).max() > 1e-5


def test_core_donor_binds_exact_upstream_and_refuses_mismatch(pretrained, tmp_path):
    from predictor_plugins.modular_temporal import build_modular
    result = pretrained["result"]
    prov = json.loads((pretrained["out"] / "core.provenance.json").read_text())
    assert [b["donor_sha256"] for b in prov["upstream_branch_donors"]] == \
        [b["donor_sha256"] for b in result["branches"]]
    assert prov["fused_materialization"]["train"] == result["fused_train"]["sha256"]
    assert result["core"]["upstream_bound"]["branch_weights_sha256"] == \
        [b["donor_weights_sha256"] for b in result["branches"]]
    build_modular(regime_config(result["fine_tune_config"], "R1"))  # matched upstream loads

    # (a) one branch fresh (R0) under a core donor -> refused before any fit
    bad = regime_config(result["fine_tune_config"], "R1")
    bad["branches"][0].update(regime="R0", donor=None)
    with pytest.raises(ValueError, match="mismatch"):
        build_modular(bad)
    # (b) a different, valid branch donor from another pretraining run -> refused
    t, v = populations()
    other = pretrain_components(pretrained["config"], pretrained["train"] + 0.5, pretrained["val"],
                                tmp_path / "other", FIT, t, v, seed=11)
    swapped = regime_config(result["fine_tune_config"], "R1")
    swapped["branches"][0]["donor"] = other["fine_tune_config"]["branches"][0]["donor"]
    with pytest.raises(ValueError, match="mismatch"):
        build_modular(swapped)
    # (c) the other run's core donor under this run's branches -> refused
    crossed = regime_config(result["fine_tune_config"], "R1")
    crossed["core"]["donor"] = other["fine_tune_config"]["core"]["donor"]
    with pytest.raises(ValueError, match="mismatch"):
        build_modular(crossed)


def test_corrupted_materialization_is_detected(pretrained, tmp_path):
    import shutil
    from tools.modular_pretrain import _array_identity
    copy_path = tmp_path / "fused.npy"
    shutil.copy(pretrained["out"] / "fused_train.npy", copy_path)
    arr = np.load(copy_path, mmap_mode="r+")
    arr[0, 0, 0] += 1.0
    arr.flush()
    assert _array_identity(np.load(copy_path, mmap_mode="r"))["sha256"] != pretrained["result"]["fused_train"]["sha256"]


def _toy(seed=0):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input((4, 2))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(inputs))
    rng = np.random.default_rng(seed)
    return model, rng.normal(size=(16, 4, 2)).astype("float32"), rng.normal(size=(8, 4, 2)).astype("float32")


def test_patience_stop_restores_selected_checkpoint():
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    model, x, vx = _toy()
    r = fit_with_early_stopping(model, x, x, vx, vx, dict(max_epochs=10, patience=2, min_delta=1e9,
                                                          batch_size=4, loss="mse"))
    assert r["stop_reason"] == "patience" and r["stop_class"] == "no_improvement"
    assert r["selected_epoch"] == 1 and r["epochs_completed"] == 3
    assert r["observed_updates"] == 12 and r["selected_updates"] == 4
    assert r["restored_weights_sha256"] == r["history"][0]["weights_sha256"]
    assert r["restored_differs_from_last"]


def test_monitor_cadence_and_budget_stop_class():
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    model, x, vx = _toy(1)
    r = fit_with_early_stopping(model, x, x, vx, vx, dict(max_epochs=5, patience=50, monitor_every=2,
                                                          batch_size=4, loss="mse"))
    assert [h["epoch"] for h in r["history"] if h["monitored"]] == [2, 4, 5]
    assert r["monitor_evaluations"] == 3 and r["selected_epoch"] in (2, 4, 5)
    assert r["stop_reason"] == "max_epochs" and r["stop_class"] == "budget"
    model, x, vx = _toy(2)
    r = fit_with_early_stopping(model, x, x, vx, vx, dict(max_epochs=5, max_updates=6, batch_size=4, loss="mse"))
    assert r["stop_reason"] == "max_updates" and r["stop_class"] == "budget" and r["observed_updates"] == 6


def test_early_stop_rule_identity_separates_variants():
    from tools.modular_candidate_evaluator import _settings, early_stop_identity
    base = early_stop_identity(_settings({}))["sha256"]
    assert early_stop_identity(_settings({"evaluator": {"batch_size": 8}}))["sha256"] == base
    for change in ({"patience": 6}, {"min_delta": 0.01}, {"monitor_every": 2}, {"max_epochs": 21},
                   {"max_updates": 5}, {"max_seconds": 10.0}):
        assert early_stop_identity(_settings({"evaluator": change}))["sha256"] != base
    with pytest.raises(ValueError):
        _settings({"evaluator": {"monitor": "test_loss"}})


def test_heartbeat_cadence_and_bounds(tmp_path):
    with pytest.raises(ValueError):
        Heartbeat(tmp_path / "hb.jsonl", 61)
    path = tmp_path / "hb.jsonl"
    with Heartbeat(path, 0.05) as beat:
        beat.update(stage="unit", last_checkpoint={"stage": "x"})
        time.sleep(0.4)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) >= 4 and lines[-1]["stage"] == "finished"
    gaps = np.diff([line["time_unix"] for line in lines])
    assert gaps.max() < 0.05 + 0.25
    assert {"stage", "last_checkpoint", "resources"} <= set(lines[2])
    assert "vmrss_kib" in lines[2]["resources"]


def test_end_to_end_pretrain_then_real_evaluator_regimes(tmp_path):
    """Real engine + real evaluator on a SYNTHETIC fixture: plumbing, not a result."""
    from predictor_plugins.modular_temporal import weights_hash
    import tensorflow as tf
    summary = run_synthetic_pilot(tmp_path / "pilot", rows=700, features=2, horizons=(1, 3), seed=5,
                                  fit=dict(max_epochs=2, patience=1, batch_size=32, loss="mse"),
                                  heartbeat_interval=0.5)
    assert summary["label"].startswith("SYNTHETIC")
    assert summary["data"]["test_rows_generated"] == 0
    down = summary["downstream"]
    assert set(down) == {"R0", "R1", "R2"}
    assert down["R1"]["initial_weights_sha256"] == down["R2"]["initial_weights_sha256"]
    assert down["R0"]["initial_weights_sha256"] != down["R1"]["initial_weights_sha256"]
    for regime in down.values():
        assert regime["observed_updates"] > 0 and regime["persistence_MAE"] > 0
    pre = json.loads((tmp_path / "pilot" / "pretrain" / "PRETRAIN.json").read_text())
    donor_core = pre["core"]["donor_weights_sha256"]
    r1 = tf.keras.models.load_model(tmp_path / "pilot" / "forecast_R1" / "best.keras", compile=False)
    r2 = tf.keras.models.load_model(tmp_path / "pilot" / "forecast_R2" / "best.keras", compile=False)
    assert weights_hash(r1.get_layer("temporal_core")) == donor_core      # frozen: unchanged
    assert weights_hash(r2.get_layer("temporal_core")) != donor_core      # fine-tuned: updated
    assert (tmp_path / "pilot" / "heartbeat.jsonl").exists()


def test_input_swap_from_train_npz_refuses_non_train(tmp_path):
    from tools.modular_pretrain import build_synthetic_splits, internal_split, pretrain_from_train_npz
    train, validation, *_ = build_synthetic_splits(rows=500, features=2, window=24, horizons=[1], seed=2)
    np.savez(tmp_path / "SYNTHETIC_train.npz", **train)
    np.savez(tmp_path / "SYNTHETIC_validation.npz", **validation)
    with pytest.raises(ValueError, match="TRAIN split only"):
        pretrain_from_train_npz(tmp_path / "SYNTHETIC_validation.npz", tmp_path / "v", FIT,
                                provenance="synthetic_fixture")
    with pytest.raises(ValueError, match="manifest digest"):
        pretrain_from_train_npz(tmp_path / "SYNTHETIC_train.npz", tmp_path / "g", FIT,
                                provenance="governed_resource")
    tr, va, (a, b) = internal_split(train["timestamps"], 24, 1)
    assert va[0] - tr[-1] == 25 and a[1] < b[0]
    result = pretrain_from_train_npz(tmp_path / "SYNTHETIC_train.npz", tmp_path / "ok", FIT,
                                     provenance="synthetic_fixture", seed=1)
    assert result["train_input"]["shape"][0] == len(tr)
    assert result["validation_input"]["shape"][0] == len(va)
    assert result["label"].startswith("SYNTHETIC")
