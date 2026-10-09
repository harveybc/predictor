"""Acceptance tests for the executable I6-D matched-control harness."""

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import tensorflow as tf

from tools import i6d_matched_control as i6d


@pytest.fixture(autouse=True)
def deterministic():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(6102026)
    yield
    tf.keras.backend.clear_session()


def config():
    return {
        "schema": i6d.CONFIG_SCHEMA,
        "experiment_id": "i6d_fixture",
        "feature_names": ["close", "volume", "rate"],
        "feature_groups": [
            {"name": "close", "channels": ["close"]},
            {"name": "volume", "channels": ["volume"]},
            {"name": "rate", "channels": ["rate"]},
        ],
        "window": 24,
        "sample_hours": 1,
        "horizons": [1, 3, 6],
        "target_names": ["close"],
        "seed": 17,
        "branch": {
            "dense": {"hidden_units": [8], "latent_units": 4,
                      "activation": "gelu", "use_bias": True},
            "conv": {"channels": 4, "kernel_size": 3},
        },
        "core": {
            "vector_width": 24,
            "dense_hidden_units": [16],
            "conv": {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 24,
                     "dropout": 0.0, "stage_channels": [12, 10, 4],
                     "time_factors": [2, 2, 1], "kernel_size": 3},
            "conv_output_steps": 6,
            "conv_output_channels": 4,
        },
        "head": {"initializer_seed": 991},
        "fit": {"max_epochs": 3, "patience": 1, "batch_size": 4,
                "learning_rate": 0.001, "weight_decay": 0.0001,
                "loss": "huber", "huber_delta": 1.0, "min_delta": 0.0,
                "max_updates": 20, "max_seconds": 60.0,
                "monitor": "validation_loss", "monitor_every": 1},
    }


def test_matched_arms_share_rows_budget_seed_and_exact_initial_head():
    harness = i6d.build_matched_control(config())

    assert harness.dense.model.input_shape == harness.conv.model.input_shape == (None, 24, 3)
    assert harness.dense.model.output_shape == harness.conv.model.output_shape == (None, 3, 1)
    assert harness.report["matched_contract"] == {
        "feature_names": ["close", "volume", "rate"],
        "selected_feature_union": ["close", "volume", "rate"],
        "window_hours": 24,
        "horizons": [1, 3, 6],
        "target_names": ["close"],
        "seed": 17,
        "fit_sha256": i6d.canonical_sha256(config()["fit"]),
        "predictive_head_sha256": harness.report["arms"]["DENSE"]["head_sha256"],
    }
    assert (harness.report["arms"]["DENSE"]["head_sha256"] ==
            harness.report["arms"]["CONV"]["head_sha256"])
    assert (harness.report["arms"]["DENSE"]["head_parameters"] ==
            harness.report["arms"]["CONV"]["head_parameters"])
    assert harness.report["arms"]["DENSE"]["representation_shape"] == [24]
    assert harness.report["arms"]["DENSE"]["representation_semantics"] == "unordered_latent_vector"
    assert harness.report["arms"]["CONV"]["representation_shape"] == [6, 4]
    assert harness.report["arms"]["CONV"]["representation_semantics"] == "ordered_temporal_sequence"
    assert harness.report["arms"]["DENSE"]["parameters"] != harness.report["arms"]["CONV"]["parameters"]
    assert harness.report["arms"]["DENSE"]["branches"] == 3
    assert harness.report["arms"]["CONV"]["branches"] == 3


def test_dense_arm_never_invents_a_temporal_axis_and_conv_keeps_one():
    harness = i6d.build_matched_control(config())

    dense_shapes = [tuple(layer.output.shape) for layer in harness.dense.representation.layers
                    if "dense_core" in layer.name]
    assert dense_shapes and all(len(shape) == 2 for shape in dense_shapes)
    assert not any("time" in key for key in harness.report["arms"]["DENSE"])
    assert harness.conv.representation.output_shape == (None, 6, 4)
    assert harness.conv.time_grid == (4, 8, 12, 16, 20, 24)


def test_train_only_plumbing_uses_identical_rows_without_fitting_or_split_reads():
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(5, 24, 3)).astype("float32")
    targets = rng.normal(size=(5, 3, 1)).astype("float32")
    row_ids = np.array([f"train-{index}" for index in range(5)])

    result = i6d.train_only_plumbing(config(), windows, targets, row_ids)

    assert result["status"] == "TRAIN_ONLY_PLUMBING_OK"
    assert result["rows"] == 5
    assert result["row_identity_sha256"] == i6d.array_sha256(row_ids)
    assert result["arms"]["DENSE"]["output_shape"] == [5, 3, 1]
    assert result["arms"]["CONV"]["output_shape"] == [5, 3, 1]
    assert result["fit_invocations"] == 0
    assert result["validation_read"] is False
    assert result["test_read"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update(window=23), "24"),
    (lambda c: c.update(seed=True), "seed"),
    (lambda c: c.update(feature_names=["close", "close"]), "feature"),
    (lambda c: c["feature_groups"][0]["channels"].append("volume"), "exactly once"),
    (lambda c: c["core"].update(vector_width=11), "vector_width"),
    (lambda c: c["fit"].update(max_updates=0), "max_updates"),
])
def test_design_refuses_unmatched_or_invalid_contracts(mutation, match):
    candidate = config()
    mutation(candidate)
    with pytest.raises(ValueError, match=match):
        i6d.build_matched_control(candidate)


def test_campaign_init_status_and_terminal_reconciliation_are_resumable(tmp_path):
    design = i6d.initialize_campaign(config(), tmp_path / "campaign",
                                     train_path="/data/train.npz",
                                     validation_path="/data/validation.npz")
    status = i6d.campaign_status(tmp_path / "campaign")
    assert design["status"] == "READY"
    assert status["state"] == "READY"
    assert status["complete_arms"] == []
    assert status["pending_arms"] == ["CONV", "DENSE"]
    assert status["test_read"] is False

    for arm in ("DENSE", "CONV"):
        terminal = {
            "schema": i6d.TERMINAL_SCHEMA,
            "arm": arm,
            "design_sha256": design["design_sha256"],
            "data_identity_sha256": "a" * 64,
            "fit_sha256": design["fit_sha256"],
            "seed": 17,
            "status": "COMPLETED",
        }
        i6d.write_terminal_for_test(tmp_path / "campaign", terminal)
    closed = i6d.close_campaign(tmp_path / "campaign")
    assert closed["state"] == "COMPLETE"
    assert closed["matched"]["data_identity_sha256"] == "a" * 64
    assert i6d.campaign_status(tmp_path / "campaign")["state"] == "COMPLETE"


def test_cli_plan_and_status_are_machine_readable(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    assert i6d.main(["plan", "--config", str(path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["arms"]["DENSE"]["representation_semantics"] == "unordered_latent_vector"

    assert i6d.main(["init", "--config", str(path), "--output", str(tmp_path / "run"),
                     "--train", "/data/train.npz", "--validation", "/data/validation.npz"]) == 0
    capsys.readouterr()
    assert i6d.main(["status", "--output", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "READY"


def test_executable_eurusd_config_binds_selected20_without_dropping_masks():
    path = Path("examples/config/i6d/i6d_eurusd_y_s_1h_matched_control.json")
    harness = i6d.build_matched_control(json.loads(path.read_text()))
    contract = harness.report["matched_contract"]

    assert len(contract["selected_feature_union"]) == 20
    assert len(contract["feature_names"]) == 40
    assert harness.report["arms"]["DENSE"]["branches"] == 20
    assert harness.report["arms"]["CONV"]["branches"] == 20
    assert all(name.endswith(("__value", "__observed")) for name in contract["feature_names"])


def test_split_contract_keeps_external_target_and_explicit_same_row_baseline(tmp_path):
    candidate = config()
    candidate["target_names"] = ["future_return"]
    n = 5
    origins = np.arange(n, dtype="int64") * 3600 + 100000
    content = {
        "windows": np.ones((n, 24, 3), dtype="float32"),
        "targets": np.full((n, 3, 1), 0.2, dtype="float32"),
        "baseline": np.zeros((n, 3, 1), dtype="float32"),
        "row_ids": np.array([f"row-{i}" for i in range(n)]),
        "timestamps": origins,
        "target_timestamps": origins[:, None] + np.array([1, 3, 6]) * 3600,
        "dataset_id": np.array("fixture"), "split": np.array("train"),
        "feature_names": np.array(candidate["feature_names"]),
        "target_names": np.array(candidate["target_names"]),
        "horizons": np.array(candidate["horizons"]),
        "timestamp_unit": np.array("seconds"), "metric_space": np.array("return"),
        "scaler_identity": np.array("train-scaler"), "scaler_scale": np.array([1.0]),
    }
    path = tmp_path / "train.npz"
    np.savez(path, **content)
    loaded = i6d._load_split(path, "train", i6d.normalize_matched_config(candidate))

    assert loaded["target_names"].tolist() == ["future_return"]
    np.testing.assert_array_equal(loaded["baseline"], 0.0)
    content["baseline"] = np.zeros((n, 1, 1), dtype="float32")
    np.savez(path, **content)
    with pytest.raises(ValueError, match="baseline"):
        i6d._load_split(path, "train", i6d.normalize_matched_config(candidate))
