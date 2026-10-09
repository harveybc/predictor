"""Acceptance tests for the I6-D branch-only Dense-versus-Conv control."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from predictor_plugins import modular_temporal as mt


def _control():
    config = mt.default_config(["close", "volume"])
    config["alignment_probe"] = False
    config["horizons"] = [1, 6]
    return mt.build_branch_only_control(
        config,
        seed=17,
        conv_params={"channels": 8, "kernel_size": 3},
        dense_params={
            "context": 3,
            "hidden_units": [8],
            "channels": 8,
            "activation": "gelu",
            "use_bias": True,
        },
    )


def test_branch_only_control_changes_only_branch_implementation_and_parameters():
    control = _control()

    assert control.report["only_branch_configuration_differs"] is True
    assert all(
        path.startswith("branches[") and (
            path.endswith(".plugin") or ".params." in path
        )
        for path in control.report["configuration_differences"]
    )
    assert {
        branch["plugin"] for branch in control.conv.config["branches"]
    } == {"causal_conv1d"}
    assert {
        branch["plugin"] for branch in control.dense.config["branches"]
    } == {"causal_dense_sequence"}


def test_branch_only_control_has_identical_fusion_core_head_and_temporal_shapes():
    control = _control()

    assert control.conv.branch_time_grid == control.dense.branch_time_grid
    assert control.conv.core_time_grid == control.dense.core_time_grid
    assert control.conv.encoder_model.output_shape == control.dense.encoder_model.output_shape
    assert control.conv.forecast_model.output_shape == control.dense.forecast_model.output_shape
    assert control.report["shared"]["fusion_identity_sha256"]
    assert control.report["shared"]["core_initial_weights_sha256"]
    assert control.report["shared"]["head_initial_weights_sha256"]
    assert control.report["shared"]["core_config_sha256"]
    assert control.report["shared"]["head_config_sha256"]


def test_branch_only_control_refuses_width_or_regime_mismatch():
    config = mt.default_config(["close"])
    config["alignment_probe"] = False

    try:
        mt.build_branch_only_control(
            config, seed=3,
            conv_params={"channels": 8, "kernel_size": 3},
            dense_params={"channels": 7},
        )
    except ValueError as exc:
        assert "same channel width" in str(exc)
    else:
        raise AssertionError("mismatched branch widths were accepted")

    config["branches"][0]["regime"] = "R1"
    config["branches"][0]["donor"] = "/tmp/not-a-real-donor"
    try:
        mt.build_branch_only_control(config, seed=3)
    except ValueError as exc:
        assert "R0" in str(exc)
    else:
        raise AssertionError("donor-backed branch control was accepted")
