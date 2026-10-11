"""Real Keras contracts for sparse-clock forecasting, independent of fit quality."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tools.i6e_masked_model import build_model


def test_placeholder_invariance_shapes_and_capacity(tmp_path):
    counts = []
    for steps in (3, 5, 9):
        tf.keras.utils.set_random_seed(6102026)
        model = build_model(steps, 2, 144 / (steps - 1))
        counts.append(model.count_params())
        x = np.ones((2, steps, 5), dtype=np.float32)
        x[:, 1, -1] = 0
        before = model(x, training=False).numpy()
        x[:, 1, :-1] = 999
        np.testing.assert_array_equal(before, model(x, training=False).numpy())
        assert before.shape == (2, 1)
        assert model.get_layer("core_mask_2").output.shape[1:] == (steps, 8)
        path = tmp_path / f"model_{steps}.keras"
        model.save(path)
        loaded = tf.keras.models.load_model(path)
        np.testing.assert_array_equal(before, loaded(x, training=False).numpy())
    assert len(set(counts)) == 1


def test_future_tokens_do_not_change_past_core_outputs():
    model = build_model(9, 2, 18)
    core = tf.keras.Model(model.input, model.get_layer("core_mask_2").output)
    x = np.ones((1, 9, 5), dtype=np.float32)
    before = core(x, training=False).numpy()
    x[:, 6:, :-1] = 1000
    np.testing.assert_array_equal(before[:, :6], core(x, training=False).numpy()[:, :6])
