"""Lane A model/training contracts on synthetic tensors (CPU only).

Placement logic is tested on CPU by contract; a real GPU probe stays gated
in tests/test_i6n_attention_mask.py (I6N_GPU_MASK_PROBE=1, admitted host).
"""

import copy

import numpy as np
import pytest

from tools.i6n_contracts.temporal_ops import reduction_plan
from tools.i6n_model import DEFAULT_CONFIG, build_model, verify_placement
from tools.i6n_training import fit_masked


def config(**updates):
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(updates)
    return result


PLAN = dict(max_epochs=50, patience=5, min_delta=1e-6, batch_size=4, learning_rate=.01,
            weight_decay=0., huber_delta=1., seed=6102026)


@pytest.fixture
def keras():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(6102026)
    yield tf.keras
    tf.keras.backend.clear_session()


@pytest.mark.parametrize("samples,strides,lengths", [
    (9, [2, 2, 2], [5, 3, 2]),
    (13, [2, 1, 2], [7, 7, 4]),
    (25, [2, 2], [13, 7]),
    (3, [2], [2]),
    (433, [2, 2, 2], [217, 109, 55]),
])
def test_odd_cascades_keep_origin(samples, strides, lengths):
    plan = reduction_plan(samples, strides)
    assert [len(stage) for stage in plan] == lengths
    for stage in plan:
        assert stage[-1] == samples - 1  # original index of t survives every stage
        assert list(stage) == sorted(set(stage))


@pytest.mark.parametrize("bad", [(0, [2]), (5, []), (5, [3]), (5, [0]), (5, [True])])
def test_reductions_that_lose_t_or_empty_rejected(bad):
    with pytest.raises(ValueError):
        reduction_plan(*bad)


def set_current_taps(keras, layer):
    if isinstance(layer, keras.Model):
        for child in layer.layers:
            set_current_taps(keras, child)
    elif isinstance(layer, keras.layers.Conv1D):
        kernel, bias = layer.get_weights()
        kernel[:] = 0
        kernel[-1, 0, 0] = 1
        layer.set_weights([kernel, np.zeros_like(bias)])
    elif isinstance(layer, keras.layers.Dense):
        kernel, bias = layer.get_weights()
        kernel[:] = 0
        kernel[0, :] = 1
        layer.set_weights([kernel, np.zeros_like(bias)])


@pytest.mark.parametrize("samples,strides", [(9, [2, 2, 2]), (13, [2, 1, 2]), (433, [2, 2])])
def test_t_reaches_every_trajectory_head_after_all_reductions(keras, samples, strides):
    model = build_model(samples, 1, 1, config(reducer_filters=[4] * len(strides),
                                             reducer_strides=strides, n_horizons=3))
    set_current_taps(keras, model)
    data = np.zeros((1, samples, 3), dtype="float32")
    data[:, :, 1:] = 1
    base = model(data).numpy()
    data[:, -1, 0] = 4
    assert base.shape == (1, 3) and np.all(model(data).numpy() - base > .1)
    data[:, -1, -1] = 0  # missing t zeroes the head input, never a stale tick
    np.testing.assert_array_equal(model(data), 0)


@pytest.mark.parametrize("entrance,blocks,adapter", [
    ("conv", 0, "none"), ("transformer", 1, "lstm"), ("conv_transformer", 2, "causalconv"),
])
def test_no_flatten_sequence_core_multihead(keras, entrance, blocks, adapter):
    model = build_model(9, 2, 3, config(entrance=entrance, transformer_blocks=blocks, width=8,
                                        reducer_filters=[8, 4], reducer_strides=[2, 2],
                                        adapter=adapter,
                                        adapter_width=None if adapter == "none" else 4,
                                        n_horizons=4))
    for layer in model.layers:
        assert not isinstance(layer, (keras.layers.Flatten, keras.layers.Lambda,
                                      keras.layers.GlobalAveragePooling1D))
    assert len(model.get_layer("core_output").output.shape) == 3
    if adapter == "lstm":
        assert model.get_layer("final_adapter").return_sequences is True
    assert model.output.shape[-1] == 4


def test_future_slots_never_change_earlier_core_steps_multihead(keras):
    model = build_model(13, 1, 1, config(entrance="transformer", transformer_blocks=1,
                                         reducer_strides=[1], reducer_filters=[4],
                                         adapter="lstm", adapter_width=4, n_horizons=3))
    core = keras.Model(model.input, model.get_layer("core_output").output)
    data = np.random.default_rng(1).normal(size=(2, 13, 3)).astype("float32")
    data[:, :, 1:] = 1
    changed = data.copy()
    changed[:, 7:, 0] += 50
    np.testing.assert_allclose(core(data)[:, :7], core(changed)[:, :7], atol=1e-5)


def test_build_433_samples_trajectory_without_truncation(keras):
    model = build_model(433, 2, 1, config(width=8, reducer_filters=[4], reducer_strides=[2],
                                          n_horizons=3))
    assert model.input.shape[1] == 433
    data = np.zeros((1, 433, 5), "float32")
    data[:, :, 1::2] = 1
    data[:, :, -1] = 1
    assert np.isfinite(model(data)).all() and model(data).shape == (1, 3)


def test_strict_placement_contract_on_cpu(keras):
    import tensorflow as tf
    previous = tf.config.get_soft_device_placement()
    try:
        tf.config.set_soft_device_placement(False)
        with tf.device("/CPU:0"):
            model = build_model(5, 2, 1, config(entrance="transformer", transformer_blocks=1,
                                                width=8, reducer_filters=[4], reducer_strides=[1], n_horizons=2))
            probe = np.ones((1, 5, 5), "float32")
            report = verify_placement(model, "/CPU:0", probe)
        assert report == dict(soft_device_placement=False, variables_on_device=True,
                              attention_variables_on_device=True,
                              output_device=report["output_device"])
        assert report["output_device"].endswith("/device:CPU:0")
        with pytest.raises(ValueError, match="PLACEMENT_MISMATCH"):
            verify_placement(model, "/GPU:0", probe)
        tf.config.set_soft_device_placement(True)
        with pytest.raises(ValueError, match="SOFT_PLACEMENT_FORBIDDEN"):
            verify_placement(model, "/CPU:0", probe)
    finally:
        tf.config.set_soft_device_placement(previous)


def test_unsupported_head_gets_no_gradient_no_nan_equal_weights(keras):
    tf = pytest.importorskip("tensorflow")
    m = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(3)])
    x = np.random.default_rng(3).normal(size=(12, 2)).astype("float32")
    y = np.random.default_rng(4).normal(size=(12, 3)).astype("float32")
    mask = np.ones_like(y, bool)
    mask[:, 0] = False          # head without any support
    y[:, 0] = np.nan            # never a residual, never a gradient
    mask[::3, 2] = False        # missing endpoint labels on some rows
    y[::3, 2] = np.nan
    before = [w.copy() for w in m.get_weights()]
    result = fit_masked(m, x, y, mask, x, y, mask, dict(PLAN, head_weights=[1., 1., 1.]))
    kernel, bias = m.get_weights()
    np.testing.assert_array_equal(kernel[:, 0], before[0][:, 0])
    assert bias[0] == before[1][0]
    assert not np.array_equal(kernel[:, 1], before[0][:, 1])
    assert result["counts"]["train"] == [0, 12, 8]
    assert result["early_stop"]["rule"]["patience"] == 5
    assert result["early_stop"]["rule"]["max_epochs"] == 50
    assert result["early_stop"]["rule"]["min_delta"] == 1e-6
    for row in result["history"]:
        assert row["train"]["counts"][0] == 0 and row["train"]["per_head"][0] == 0.


def test_plan_patience_five_stops_and_restores_best(keras):
    tf = pytest.importorskip("tensorflow")
    m = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    x = np.ones((4, 1), "float32")
    y = np.full((4, 1), 3., "float32")
    result = fit_masked(m, x, y, np.ones_like(y, bool), x, y, np.ones_like(y, bool),
                        dict(PLAN, learning_rate=1e-12, head_weights=[1.]))
    assert result["stop_reason"] == "patience" and result["epochs_completed"] == 6
    assert result["restored_best_weights"] is True
    assert result["selected_weights_sha256"] == result["restored_weights_sha256"]
