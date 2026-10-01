"""Hourly default architecture, built with real Keras (owner-approved shapes, 2026-09-30).

Each branch (B,24,1)->(B,24,16) with no time reduction; channel concatenation
(B,24,16F); positional encoding immediately after fusion; per-step projection to
64; two causal Transformer blocks; residual Conv1D stages (B,12,32)->(B,6,16)->
(B,6,8); no flatten before the head; FeatureSelect is fixed tf.gather routing.
Shapes alone do not establish time alignment, so the grids are also checked
behaviourally. Synthetic component checks, not forecasting results.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(1)
    yield


def bundle(f):
    return mt.build_modular(mt.default_config([f"f{i}" for i in range(f)]))


@pytest.mark.parametrize("f", [3, 7])
def test_default_shapes_at_every_stage(f):
    b = bundle(f)
    assert tuple(b.forecast_model.input.shape) == (None, 24, f)
    assert all(tuple(m.output.shape) == (None, 24, 16) for m in b.branch_models.values())
    assert len(b.branch_models) == f
    assert tuple(b.fusion_model.output.shape) == (None, 24, 16 * f)
    core = b.core_model
    assert isinstance(core.layers[1], mt.PositionalEncoding)          # PE immediately after fusion input
    assert core.get_layer("model_projection").units == 64
    attention = [l for l in core.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    assert len(attention) == 2 and all(l.get_config()["num_heads"] == 4 for l in attention)
    for i in range(2):                                                 # attention + FFN, residuals, norms
        for name in (f"attention_norm_{i}", f"ffn_expand_{i}", f"ffn_project_{i}", f"ffn_norm_{i}"):
            core.get_layer(name)
    stages = [tuple(core.get_layer(f"stage_{i}_output").output.shape) for i in range(3)]
    assert stages == [(None, 12, 32), (None, 6, 16), (None, 6, 8)]
    for i, factor in enumerate((2, 2, 1)):
        for part in ("_skip_downsample", "_block_projection"):
            layer = core.get_layer(f"stage_{i}{part}")
            assert layer.strides == (factor,) and layer.kernel_size == (factor,) and layer.padding == "valid"
    assert tuple(b.encoder_model.output.shape) == (None, 6, 8)
    assert tuple(b.forecast_model.output.shape) == (None, 1, 1)
    for model in [*b.branch_models.values(), b.fusion_model, core]:
        assert all(len(l.output.shape) == 3 for l in model.layers)       # no temporal collapse
        assert not any(isinstance(l, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D))
                       for l in model.layers)
    assert any(isinstance(l, tf.keras.layers.Flatten) for l in b.forecast_model.get_layer("forecast_head").layers)
    assert b.branch_time_grid == tuple(range(1, 25)) and b.core_time_grid == tuple(range(4, 25, 4))


def test_feature_select_is_fixed_tf_gather_routing():
    b = bundle(3)
    selects = [l for l in b.forecast_model.layers if isinstance(l, mt.FeatureSelect)]
    assert len(selects) == 3 and all(not l.weights for l in selects)
    x = np.random.default_rng(0).normal(size=(2, 24, 3)).astype("float32")
    for i, layer in enumerate(selects):
        np.testing.assert_array_equal(layer(x), tf.gather(x, [i], axis=-1))
        np.testing.assert_array_equal(layer(x), x[:, :, [i]])


def _first_changed(model, x, i):
    changed = x.copy()
    changed[:, i, :] += 5.0
    moved = np.max(np.abs(np.asarray(model(changed)) - np.asarray(model(x))), axis=(0, 2))
    return int(np.argmax(moved > 1e-6)), moved


def test_time_grids_are_behavioural_at_every_stage():
    b = bundle(3)
    x = np.random.default_rng(2).normal(size=(2, 24, 3)).astype("float32")
    core = b.core_model
    fused = b.fusion_model(x).numpy()
    stage_models = [tf.keras.Model(core.input, core.get_layer(f"stage_{i}_output").output) for i in range(3)]
    for i in range(24):
        first, moved = _first_changed(b.fusion_model, x, i)             # branch grid = input grid
        assert first == i and np.all(moved[:i] <= 1e-6)
        for model, block in zip(stage_models, (2, 4, 4)):              # right edges 2..24, 4..24, 4..24
            first, moved = _first_changed(model, fused, i)
            assert first == i // block and np.all(moved[:i // block] <= 1e-6)
        first, moved = _first_changed(b.encoder_model, x, i)           # causal Transformer: no look-ahead
        assert first == i // 4 and np.all(moved[:i // 4] <= 1e-6)
