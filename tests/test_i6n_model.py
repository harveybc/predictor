"""Synthetic candidate contracts, not campaign or historical-data evidence.

Traceability: schema -> invalid/config tests; topology -> entrance/core tests;
temporal safety -> masks/prefix/origin; portability -> safe serialization.
"""

import copy
import subprocess
import sys

import numpy as np
import pytest

from tools.i6n_model import DEFAULT_CONFIG, build_model, normalize_config


def config(**updates):
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(updates)
    return result


@pytest.fixture
def keras():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(611)
    yield tf.keras
    tf.keras.backend.clear_session()


def clock(samples, features=2):
    data = np.random.default_rng(71).normal(size=(2, samples, 2 * features + 1))
    data[:, :, 1::2] = 1
    data[:, :, -1] = 1
    return data.astype("float32")


def test_import_and_validation_do_not_import_tensorflow():
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; from tools.i6n_model import "
         "DEFAULT_CONFIG, normalize_config; normalize_config(DEFAULT_CONFIG); "
         "assert 'tensorflow' not in sys.modules"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_default_schema_is_explicit_and_normalization_is_detached():
    cfg = config()
    normalized = normalize_config(cfg)
    assert normalized == cfg
    assert normalized["entrance"] == "conv"
    assert normalized["transformer_blocks"] == 0
    assert normalized["reducer_filters"] == [8, 4]
    normalized["reducer_filters"][0] = 99
    assert cfg["reducer_filters"] == [8, 4]


@pytest.mark.parametrize("key", list(DEFAULT_CONFIG))
def test_missing_schema_fields_rejected(key):
    cfg = config()
    del cfg[key]
    with pytest.raises(ValueError):
        normalize_config(cfg)


@pytest.mark.parametrize("updates", [
    {"unknown": 1}, {"entrance": "dense"}, {"entrance": True},
    {"width": True}, {"width": 0}, {"width": -1}, {"width": 3.5},
    {"width": float("inf")}, {"width": float("nan")}, {"width": 129},
    {"transformer_blocks": True}, {"transformer_blocks": -1},
    {"transformer_blocks": 3}, {"transformer_blocks": 1.0},
    {"transformer_blocks": 1},
    {"entrance": "transformer", "transformer_blocks": 0},
    {"entrance": "conv_transformer", "transformer_blocks": 0},
    {"entrance": "transformer", "transformer_blocks": 1, "width": 3},
    {"reducer_filters": []}, {"reducer_filters": [1, 2, 3, 4]},
    {"reducer_filters": [True, 4]}, {"reducer_filters": [0, 4]},
    {"reducer_filters": [float("nan"), 4]}, {"reducer_filters": [129, 4]},
    {"reducer_filters": "8,4"}, {"reducer_strides": [2]},
    {"reducer_strides": [True, 1]}, {"reducer_strides": [3, 1]},
    {"reducer_strides": [1.0, 1]}, {"adapter": "dense"},
    {"adapter_width": 4}, {"adapter": "lstm"},
    {"adapter": "causalconv", "adapter_width": True},
    {"adapter": "lstm", "adapter_width": 0},
    {"n_horizons": True}, {"n_horizons": 0},
    {"n_horizons": float("inf")}, {"n_horizons": 1.0},
    {"heads": 2}, {"ff_multiplier": 2},
])
def test_invalid_or_inactive_genes_rejected(updates):
    with pytest.raises(ValueError):
        normalize_config(config(**updates))


@pytest.mark.parametrize("cfg", [None, [], "conv"])
def test_non_mapping_config_rejected(cfg):
    with pytest.raises(ValueError):
        normalize_config(cfg)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, float("nan"), float("inf")])
def test_invalid_dimensions_fail_before_tensorflow(axis, bad):
    dims = [5, 2, 1]
    dims[axis] = bad
    with pytest.raises(ValueError):
        build_model(*dims, config())


@pytest.mark.parametrize("samples", [5, 55, 433])
@pytest.mark.parametrize("heads", [1, 4])
def test_variable_lengths_and_head_shapes(keras, samples, heads):
    model = build_model(samples, 2, 3, config(n_horizons=heads))
    assert model.input_shape == (None, samples, 5)
    assert model.output_shape == (None, heads)
    assert model.get_layer("core_output").output.shape == (None, samples, 4)
    assert np.isfinite(model(clock(samples)).numpy()).all()
    assert not getattr(model, "compiled", False)
    assert not any(isinstance(layer, keras.layers.MultiHeadAttention)
                   for layer in model.layers)


@pytest.mark.parametrize("entrance,blocks", [
    ("conv", 0), ("transformer", 1), ("transformer", 2),
    ("conv_transformer", 1), ("conv_transformer", 2),
])
def test_entrance_topology_and_no_temporal_collapse(keras, entrance, blocks):
    cfg = config(entrance=entrance, transformer_blocks=blocks, width=8)
    model = build_model(5, 2, 2, cfg)
    assert model.get_layer("fusion").output.shape == (None, 5, 32)
    position = model.get_layer("physical_position")
    assert position.input is model.get_layer("fusion").output
    assert position.spacing_hours == 2
    projection = model.get_layer("core_projection")
    assert projection.kernel_size == ((1,) if entrance == "transformer" else (3,))
    assert projection.padding == "causal"
    attention = [layer for layer in model.layers
                 if isinstance(layer, keras.layers.MultiHeadAttention)]
    assert len(attention) == blocks
    for index, layer in enumerate(attention):
        assert layer.get_config()["num_heads"] == 2
        assert layer.get_config()["key_dim"] == 4
        assert model.get_layer(f"ff_expand_{index}").units == 16
        for prefix in ("attention", "ff"):
            assert isinstance(model.get_layer(f"{prefix}_add_{index}"), keras.layers.Add)
            assert isinstance(model.get_layer(f"{prefix}_norm_{index}"),
                              keras.layers.LayerNormalization)
    for layer in model.layers:
        assert not isinstance(layer, (keras.layers.Flatten, keras.layers.Lambda))
        if isinstance(layer, keras.layers.Dense) and layer.name != "forecast_head":
            assert len(layer.input.shape) == len(layer.output.shape) == 3
    assert model.get_layer("head_input").input is model.get_layer("core_output").output
    assert np.isfinite(model(clock(5))).all()


@pytest.mark.parametrize("strides", [[1], [2], [1, 2], [2, 1], [2, 2, 2]])
def test_independent_reductions_preserve_mask_and_last_tick(keras, strides):
    from tools.i6n_contracts.temporal_ops import RightAnchoredSubsample
    model = build_model(55, 1, 1, config(
        reducer_filters=[4] * len(strides), reducer_strides=strides))
    data = clock(55, 1)
    data[:, ::3, -1] = 0
    expected = data[:, :, -1:] > .5
    length = 55
    for stage, factor in enumerate(strides):
        length = (length + factor - 1) // factor
        expected = expected[:, (expected.shape[1] - 1) % factor::factor]
        assert isinstance(model.get_layer(f"reducer_subsample_{stage}"), RightAnchoredSubsample)
        assert model.get_layer(f"reducer_mask_{stage}").output.dtype == "bool"
        assert model.get_layer(f"reducer_conv_{stage}").strides == (1,)
        probe = keras.Model(model.input, [model.get_layer(f"reducer_mask_{stage}").output,
                                        model.get_layer(f"reducer_output_{stage}").output])
        mask, values = [x.numpy() for x in probe(data)]
        assert values.shape == (2, length, 4)
        np.testing.assert_array_equal(mask, expected)
        assert np.all(values[~mask[:, :, 0]] == 0)


@pytest.mark.parametrize("adapter", ["none", "causalconv", "lstm"])
def test_market_placeholders_and_all_missing_are_ignored(keras, adapter):
    cfg = config(entrance="conv_transformer", transformer_blocks=1,
                 adapter=adapter, adapter_width=None if adapter == "none" else 4,
                 reducer_strides=[2, 1])
    model = build_model(9, 2, 1, cfg)
    data = clock(9)
    data[:, [0, 2, 5, 8], -1] = 0
    changed = data.copy()
    changed[:, [0, 2, 5, 8], :-1] = np.nan
    np.testing.assert_allclose(model(data), model(changed), atol=1e-6)
    core = keras.Model(model.input, model.get_layer("core_output").output)
    assert np.all(core(data).numpy()[:, -1] == 0)
    data[:, :, -1] = 0
    data[:, :, :-1] = np.nan
    assert np.all(core(data).numpy() == 0)
    assert np.isfinite(model(data)).all()
    if adapter == "lstm":
        assert model.get_layer("final_adapter").return_sequences
        assert model.get_layer("final_adapter").use_cudnn is False


@pytest.mark.parametrize("entrance,blocks", [("conv", 0), ("transformer", 1)])
@pytest.mark.parametrize("adapter", ["none", "causalconv", "lstm"])
def test_core_future_perturbation_is_causal(keras, entrance, blocks, adapter):
    model = build_model(9, 1, 1, config(
        entrance=entrance, transformer_blocks=blocks, adapter=adapter,
        adapter_width=None if adapter == "none" else 4))
    core = keras.Model(model.input, model.get_layer("core_output").output)
    data = clock(9, 1)
    changed = data.copy()
    changed[:, 5:, 0] += 100
    np.testing.assert_allclose(core(data)[:, :5], core(changed)[:, :5], atol=1e-6)


@pytest.mark.parametrize("samples", [5, 55, 433])
def test_controllable_origin_impulse_reaches_head(keras, samples):
    model = build_model(samples, 1, 1, config(reducer_strides=[2, 2]))
    # Make the path deterministic: current-tap-only positive kernels.
    def set_current_taps(layer):
        if isinstance(layer, keras.Model):
            for child in layer.layers:
                set_current_taps(child)
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
    set_current_taps(model)
    data = np.zeros((1, samples, 3), dtype="float32")
    data[:, :, 1:] = 1
    base = model(data).numpy()
    data[:, -1, 0] = 4
    assert np.all(model(data).numpy() - base > .1)
    data[:, -1, -1] = 0
    np.testing.assert_array_equal(model(data), 0)


@pytest.mark.parametrize("adapter", ["none", "causalconv", "lstm"])
def test_safe_save_load_parity(keras, tmp_path, adapter):
    model = build_model(5, 1, 1, config(
        entrance="conv_transformer", transformer_blocks=2,
        reducer_strides=[2, 2], n_horizons=3, adapter=adapter,
        adapter_width=None if adapter == "none" else 4))
    data = clock(5, 1)
    data[:, 2, -1] = 0
    path = tmp_path / "candidate.keras"
    model.save(path)
    restored = keras.models.load_model(path, safe_mode=True)
    np.testing.assert_allclose(model(data), restored(data), rtol=1e-6, atol=1e-6)
    assert "__lambda__" not in model.to_json()
    assert "Lambda" not in model.to_json()
