import numpy as np
import tensorflow as tf

from predictor_plugins.common.bayesian import predict_mc_welford
from predictor_plugins.common.base import BaseDeterministicKerasPredictor


def test_predict_mc_welford_matches_deterministic_predict_two_outputs():
    tf.random.set_seed(123)
    np.random.seed(123)

    inp = tf.keras.Input(shape=(4,), name="x")
    h = tf.keras.layers.Dense(8, activation="relu")(inp)
    o1 = tf.keras.layers.Dense(1, activation="linear", name="out1")(h)
    o2 = tf.keras.layers.Dense(1, activation="linear", name="out2")(h)
    model = tf.keras.Model(inp, [o1, o2])

    x = np.random.randn(32, 4).astype(np.float32)

    # Deterministic baseline
    p_det = model.predict(x, batch_size=16, verbose=0)
    assert isinstance(p_det, list) and len(p_det) == 2

    means, stds = predict_mc_welford(model, x, mc_samples=5, batch_size=16, training=False)
    assert isinstance(means, list) and isinstance(stds, list)
    assert len(means) == 2 and len(stds) == 2

    for i in range(2):
        np.testing.assert_allclose(means[i], p_det[i], rtol=1e-6, atol=1e-6)
        assert float(np.max(stds[i])) < 1e-6


def test_predict_mc_welford_supports_tuple_outputs_and_mixed_dims():
    tf.random.set_seed(123)
    np.random.seed(123)

    inp = tf.keras.Input(shape=(3,), name="x")
    h = tf.keras.layers.Dense(7, activation="relu")(inp)
    o1 = tf.keras.layers.Dense(2, activation="linear", name="out1")(h)
    o2 = tf.keras.layers.Dense(1, activation="linear", name="out2")(h)
    # Use tuple outputs on purpose (some Keras code paths return tuples).
    model = tf.keras.Model(inp, (o1, o2))

    x = np.random.randn(17, 3).astype(np.float32)  # not divisible by batch_size

    p_det = model.predict(x, batch_size=8, verbose=0)
    assert len(p_det) == 2
    assert p_det[0].shape == (17, 2)
    assert p_det[1].shape == (17, 1)

    means, stds = predict_mc_welford(model, x, mc_samples=3, batch_size=8, training=False)
    assert len(means) == 2 and len(stds) == 2
    assert means[0].shape == (17, 2)
    assert means[1].shape == (17, 1)

    np.testing.assert_allclose(means[0], p_det[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(means[1], p_det[1], rtol=1e-6, atol=1e-6)
    assert float(np.max(stds[0])) < 1e-6
    assert float(np.max(stds[1])) < 1e-6


def test_deterministic_predictor_bypasses_mc_uncertainty():
    class _Det(BaseDeterministicKerasPredictor):
        def build_model(self, input_shape, x_train, config):
            inp = tf.keras.Input(shape=input_shape)
            out1 = tf.keras.layers.Dense(1, name="output_horizon_1")(inp)
            out2 = tf.keras.layers.Dense(1, name="output_horizon_2")(inp)
            self.model = tf.keras.Model(inp, [out1, out2])
            self.output_names = ["output_horizon_1", "output_horizon_2"]

    plugin = _Det({"predicted_horizons": [1, 2]})
    plugin.build_model((4,), None, {})

    x = np.random.randn(10, 4).astype(np.float32)
    preds, unc = plugin.predict_with_uncertainty(x, mc_samples=10)
    assert isinstance(preds, list) and isinstance(unc, list)
    assert len(preds) == 2 and len(unc) == 2
    for p, u in zip(preds, unc):
        assert p.shape == u.shape
        assert float(np.max(np.abs(u))) == 0.0
