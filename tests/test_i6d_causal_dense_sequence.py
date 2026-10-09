"""I6-D branch-only Dense control: temporal support and plugin contract."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

from predictor_plugins import modular_temporal as mt


def test_causal_dense_sequence_preserves_time_and_blocks_future_perturbations():
    component = mt.causal_dense_sequence(
        input_shape=(24, 2), time_grid=tuple(range(24)), output_steps=24,
        name="dense_sequence", params={"context": 3, "hidden_units": [7], "channels": 5},
    )
    model = component.model
    base = np.zeros((1, 24, 2), dtype="float32")
    perturbed = base.copy()
    perturbed[:, 12, :] = 100.0

    before = model(base, training=False).numpy()
    after = model(perturbed, training=False).numpy()

    assert model.output_shape == (None, 24, 5)
    assert component.time_grid == tuple(range(24))
    np.testing.assert_array_equal(before[:, :12], after[:, :12])
    assert not np.array_equal(before[:, 12:15], after[:, 12:15])


def test_causal_dense_sequence_round_trips_without_custom_objects(tmp_path):
    component = mt.causal_dense_sequence(
        input_shape=(24, 1), time_grid=tuple(range(24)), output_steps=24,
        name="dense_sequence", params={},
    )
    values = np.arange(24, dtype="float32").reshape(1, 24, 1)
    expected = component.model(values, training=False).numpy()
    path = tmp_path / "branch.keras"
    component.model.save(path)

    restored = tf.keras.models.load_model(path, compile=False)

    np.testing.assert_allclose(restored(values, training=False).numpy(), expected)


def test_causal_dense_sequence_is_a_declared_optional_branch_plugin():
    declared = mt.describe_component("branch", "causal_dense_sequence")
    assert declared["group"] == "modular.branch"
    assert declared["name"] == "causal_dense_sequence"
    assert declared["version"] == "1.0.0"
    assert "preserves" in declared["contract"]
    assert mt.registry.DEFAULTS["modular.branch"] == "causal_conv1d"
