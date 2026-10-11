"""N0 requirements: right-anchored reduction and per-head label masking."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6n_contracts.temporal_ops import (
    RightAnchoredSubsample, masked_huber, retained_indices,
)


@pytest.mark.parametrize("length", [3, 4, 5, 9, 14, 28, 55, 241, 433])
@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_every_reduction_includes_current_tick(length, factor):
    indices = retained_indices(length, factor)
    x = tf.reshape(tf.range(length, dtype=tf.float32), (1, length, 1))
    y = RightAnchoredSubsample(factor)(x)
    np.testing.assert_array_equal(y.numpy()[0, :, 0], indices)
    assert indices[-1] == length - 1
    assert y.shape[1] == (length + factor - 1) // factor


def test_cascade_keeps_origin_and_save_load(tmp_path):
    inputs = tf.keras.Input((55, 1))
    value = RightAnchoredSubsample(2)(inputs)
    value = RightAnchoredSubsample(2)(value)
    model = tf.keras.Model(inputs, value)
    x = np.zeros((1, 55, 1), dtype=np.float32)
    x[0, -1, 0] = 7
    assert model(x).numpy()[0, -1, 0] == 7
    assert model.output_shape == (None, 14, 1)
    model.save(tmp_path / "model.keras")
    loaded = tf.keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_array_equal(model(x), loaded(x))


def test_subsample_dynamic_time_axis():
    inp = tf.keras.Input((None, 1))
    model = tf.keras.Model(inp, RightAnchoredSubsample(2)(inp))
    for n in (4, 5):
        x = tf.reshape(tf.range(n, dtype=tf.float32), (1, n, 1))
        assert model(x).numpy()[0, -1, 0] == n - 1


@pytest.mark.parametrize("bad", [True, 0, -1, 2.5])
def test_bad_factors_rejected(bad):
    with pytest.raises(ValueError):
        RightAnchoredSubsample(bad)
    with pytest.raises(ValueError):
        retained_indices(5, bad)


def test_nan_masked_before_residual_and_gradient():
    y = tf.constant([[1., np.nan], [np.nan, 4.]])
    pred = tf.Variable([[0., np.nan], [np.nan, 2.]])
    mask = tf.constant([[True, False], [False, True]])
    with tf.GradientTape() as tape:
        loss = masked_huber(y, pred, mask, [1., 1.])
    assert float(loss) == pytest.approx(1.)  # (.5 + 1.5) / 2 heads
    np.testing.assert_array_equal(tape.gradient(loss, pred).numpy(), [[-.5, 0.], [0., -.5]])


def test_per_head_counts_not_zero_padded_batch_mean():
    y = tf.constant([[1., 2.], [1., np.nan], [1., np.nan]])
    p = tf.zeros_like(y)
    m = tf.math.is_finite(y)
    assert float(masked_huber(y, p, m, [1., 3.])) == pytest.approx(1.25)


def test_empty_head_and_empty_batch_do_not_contribute():
    assert float(masked_huber([[1., np.nan]], [[0., np.nan]],
                              [[True, False]], [1., 10.])) == pytest.approx(.5)
    assert float(masked_huber([[np.nan]], [[np.nan]], [[False]], [1.])) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_valid_nonfinite_target_or_prediction_rejected(bad):
    with pytest.raises(tf.errors.InvalidArgumentError):
        masked_huber([[bad]], [[0.]], [[True]], [1.])
    with pytest.raises(tf.errors.InvalidArgumentError):
        masked_huber([[0.]], [[bad]], [[True]], [1.])


def test_mask_shape_cannot_broadcast_silently():
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        masked_huber([[1., 2.]], [[0., 0.]], [[True]], [1., 1.])


@pytest.mark.parametrize("weights", [[-1., 1.], [0., 1.], [np.nan, 1.], [1.]])
def test_invalid_head_weights_rejected(weights):
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        masked_huber([[1., 2.]], [[0., 0.]], [[True, True]], weights)
