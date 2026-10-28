"""I6-D contract tests for the non-temporal causal-window Dense control."""

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt


@pytest.fixture(autouse=True)
def deterministic():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(6102026)
    yield
    tf.keras.backend.clear_session()


def test_dense_control_consumes_one_24_hour_window_and_emits_a_vector():
    component = mt.causal_window_dense(
        input_shape=(24, 1),
        support_grid=tuple(range(-23, 1)),
        name="dense_control",
        params={"hidden_units": [16, 8], "latent_units": 8},
    )

    values = np.ones((3, 24, 1), dtype="float32")
    assert component.model(values).shape == (3, 8)
    assert component.model.input_shape == (None, 24, 1)
    assert component.model.output_shape == (None, 8)
    assert component.semantics == mt.UNORDERED_LATENT_VECTOR
    assert component.support_grid == tuple(range(-23, 1))
    assert component.available_at == 0


def test_dense_control_can_keep_one_semantic_feature_with_value_and_observation_channels():
    component = mt.causal_window_dense(
        input_shape=(24, 2),
        support_grid=tuple(range(-23, 1)),
        name="dense_control_with_mask",
        params={"hidden_units": [8], "latent_units": 4},
    )
    assert component.model.input_shape == (None, 24, 2)
    assert component.model.output_shape == (None, 4)
    assert component.semantics == mt.UNORDERED_LATENT_VECTOR


def test_dense_control_has_exact_causal_support_and_uses_every_window_position():
    component = mt.causal_window_dense(
        input_shape=(24, 1),
        support_grid=tuple(range(-23, 1)),
        name="dense_control",
        params={"hidden_units": [], "latent_units": 1, "activation": "linear"},
    )
    component.model.layers[-1].set_weights(
        [np.ones((24, 1), dtype="float32"), np.zeros((1,), dtype="float32")]
    )
    baseline = np.zeros((1, 24, 1), dtype="float32")
    for position in range(24):
        changed = baseline.copy()
        changed[0, position, 0] = 1.0
        assert float(component.model(changed)[0, 0]) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="24-hour"):
        mt.causal_window_dense(
            input_shape=(23, 1), support_grid=tuple(range(-22, 1)),
            name="too_short", params={},
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        mt.causal_window_dense(
            input_shape=(24, 1), support_grid=(0,) * 24,
            name="bad_grid", params={},
        )


def test_dense_control_parameter_count_is_explicit_and_stable():
    component = mt.causal_window_dense(
        input_shape=(24, 1),
        support_grid=tuple(range(-23, 1)),
        name="dense_control",
        params={"hidden_units": [12, 6], "latent_units": 4},
    )
    # 24*12+12, 12*6+6, 6*4+4.
    assert component.model.count_params() == 406


def test_dense_control_config_round_trips_canonically():
    config = mt.dense_control_config(
        feature_name="eurusd_close",
        hidden_units=[12, 6],
        latent_units=4,
        activation="relu",
    )
    encoded = mt.canonical_dense_control_json(config)
    decoded = mt.load_dense_control_config(encoded)

    assert decoded == config
    assert json.loads(encoded)["semantic_type"] == mt.UNORDERED_LATENT_VECTOR
    assert json.loads(encoded)["window_hours"] == 24
    assert encoded == mt.canonical_dense_control_json(dict(reversed(list(config.items()))))


def test_non_temporal_latents_cannot_claim_or_enter_a_temporal_path():
    component = mt.causal_window_dense(
        input_shape=(24, 1),
        support_grid=tuple(range(-23, 1)),
        name="dense_control",
        params={},
    )
    adapter = mt.DenseControlAdapter(latent_units=6)
    combined = adapter.combine([component], name="dense_control_adapter")

    assert combined.model.output_shape == (None, 6)
    assert combined.semantics == mt.UNORDERED_LATENT_VECTOR
    with pytest.raises(TypeError, match="fake timestamps"):
        adapter.as_temporal(component)
    with pytest.raises(ValueError, match="temporal-preservation claim"):
        mt.dense_control_config(
            feature_name="eurusd_close",
            preserves_temporal_axis=True,
        )
    with pytest.raises(TypeError, match="TemporalComponent"):
        adapter.combine([
            mt.causal_conv1d(
                input_shape=(24, 1), time_grid=tuple(range(1, 25)),
                output_steps=24, name="temporal", params={},
            )
        ], name="invalid_adapter")


def test_dense_control_is_a_declared_plugin_not_the_temporal_branch_default():
    declared = mt.describe_component("control_branch", "causal_window_dense")
    assert declared["version"] == "1.1.0"
    assert "unordered latent vector" in declared["contract"]
    assert mt.registry.DEFAULTS["modular.branch"] == "causal_conv1d"
