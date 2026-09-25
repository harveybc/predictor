# -*- coding: utf-8 -*-
"""WP24 (M5PHET work plan 2026-09-24, revision 2): the fusing core `fused_branches`.

WP18 step 5 needs a `predictor.plugins` entry that accepts SEVERAL input branches; the probe
(`tests/test_wp18_branch_capability.py`) found none. These tests are what `predictor_plugins/
fused_branches.py` is held to, and nothing more:

* it builds with two and with three branches;
* the two input declarations it accepts -- the framework's single `(window, channels)` tensor, sliced
  per branch inside the graph, and one Keras `Input` per branch -- describe the SAME model: with the
  same weights they produce the same numbers on the same data;
* a branch naming a column the input does not have is refused by name (`BRANCH_COLUMN_UNKNOWN`);
* a branch naming an encoder the plugin does not implement is refused by name (`UNKNOWN_ENCODER`);
* one epoch on 64 synthetic rows runs and predicts the declared shape.

Everything here is synthetic and on the CPU (`CUDA_VISIBLE_DEVICES=""` is set before TensorFlow is
imported): no dataset under `examples/` is read, no GPU is touched, and nothing here measures quality.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from predictor_plugins.fused_branches import ENCODERS, Plugin

WINDOW = 6
FEATURE_NAMES = ["open", "high", "low", "close"]
ROWS = 64

BASE = {
    "predicted_horizons": [1],
    "plotted_horizon": 1,
    "feature_names": FEATURE_NAMES,
    "encoder_units": 4,
    "encoder_layers": 1,
    "head_units": [8],
    "batch_size": 8,
    "mc_samples": 2,
    "quiet": True,
}


def config(**overrides):
    merged = dict(BASE)
    merged.update(overrides)
    return merged


def two_branches():
    return [{"name": "prices", "columns": ["open", "close"], "encoder": "cnn"},
            {"name": "range", "columns": ["high", "low"], "encoder": "dense"}]


def three_branches():
    return [{"name": "prices", "columns": ["open", "close"], "encoder": "cnn"},
            {"name": "range", "columns": ["high", "low"], "encoder": "lstm"},
            {"name": "close_only", "columns": ["close"], "encoder": "tcn"}]


def synthetic(rows=ROWS, channels=len(FEATURE_NAMES), seed=20260925):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, WINDOW, channels)).astype("float32")


# --- it builds with several branches ------------------------------------------------------------

@pytest.mark.parametrize("branches, expected", [(two_branches(), 2), (three_branches(), 3)])
def test_it_builds_with_two_and_with_three_branches(branches, expected):
    """One encoder per declared branch, fused, over the framework's single input tensor."""
    plugin = Plugin(config(branches=branches))
    model = plugin.build_model((WINDOW, len(FEATURE_NAMES)), synthetic(rows=8), {})
    assert len(model.inputs) == 1, "the single-tensor path keeps the pipeline's one input"
    fusion = [layer for layer in model.layers if layer.name.startswith("fusion_")]
    assert len(fusion) == 1 and len(fusion[0].input) == expected
    assert plugin.output_names == ["output_horizon_1"]


def test_it_builds_one_keras_input_per_branch_when_handed_per_branch_shapes():
    """The shape WP18's probe hands a core: a list of per-branch shapes becomes a multi-input model."""
    plugin = Plugin(config(branches=two_branches()))
    model = plugin.build_model([(WINDOW, 2), (WINDOW, 2)], None, {})
    assert len(model.inputs) == 2


def test_every_declared_encoder_builds():
    """The option labels this plugin declares are options that exist."""
    for encoder in sorted(ENCODERS):
        plugin = Plugin(config(branches=[
            {"name": "a", "columns": ["open", "close"], "encoder": encoder},
            {"name": "b", "columns": ["high", "low"], "encoder": encoder}]))
        assert len(plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {}).inputs) == 1


# --- the two input paths are the same model -----------------------------------------------------

def _copy_weights_by_layer_name(source, target):
    """Same architecture, same weights -- so any difference below is the graph, not the initializer."""
    by_name = {layer.name: layer for layer in source.layers}
    copied = 0
    for layer in target.layers:
        if not layer.weights:
            continue
        assert layer.name in by_name, f"{layer.name} exists on one path only"
        layer.set_weights(by_name[layer.name].get_weights())
        copied += 1
    assert copied > 0
    return copied


def test_the_single_input_path_and_the_list_input_path_produce_the_same_output():
    """Slicing the channels inside the graph is the same computation as feeding the slices in."""
    branches = two_branches()
    shared = Plugin(config(branches=branches, bayesian_head=False))
    shared_model = shared.build_model((WINDOW, len(FEATURE_NAMES)), None, {})

    split = Plugin(config(branches=branches, bayesian_head=False))
    split_model = split.build_model([(WINDOW, 2), (WINDOW, 2)], None, {})

    _copy_weights_by_layer_name(shared_model, split_model)

    x = synthetic()
    per_branch = [x[:, :, [0, 3]], x[:, :, [1, 2]]]     # open/close and high/low, in declaration order
    shared_out = np.asarray(shared_model.predict(x, verbose=0))
    split_out = np.asarray(split_model.predict(per_branch, verbose=0))
    assert shared_out.shape == split_out.shape
    np.testing.assert_allclose(shared_out, split_out, rtol=1e-5, atol=1e-6)


# --- refusals, by name --------------------------------------------------------------------------

def test_a_branch_naming_a_column_outside_the_input_is_refused_by_name():
    plugin = Plugin(config(branches=[
        {"name": "prices", "columns": ["open", "close"], "encoder": "cnn"},
        {"name": "ghost", "columns": ["volume"], "encoder": "dense"}]))
    with pytest.raises(ValueError) as refusal:
        plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {})
    assert "BRANCH_COLUMN_UNKNOWN" in str(refusal.value)
    assert "volume" in str(refusal.value)


def test_a_column_index_beyond_the_channels_is_refused_by_the_same_name():
    plugin = Plugin(config(feature_names=None, branches=[
        {"name": "a", "columns": [0, 1], "encoder": "dense"},
        {"name": "b", "columns": [2, 9], "encoder": "dense"}]))
    with pytest.raises(ValueError) as refusal:
        plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {})
    assert "BRANCH_COLUMN_UNKNOWN" in str(refusal.value)


def test_an_unknown_encoder_is_refused_by_name():
    plugin = Plugin(config(branches=[
        {"name": "prices", "columns": ["open", "close"], "encoder": "cnn"},
        {"name": "range", "columns": ["high", "low"], "encoder": "wavelet_oracle"}]))
    with pytest.raises(ValueError) as refusal:
        plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {})
    assert "UNKNOWN_ENCODER" in str(refusal.value)
    assert "wavelet_oracle" in str(refusal.value)


def test_an_unknown_fusion_is_refused_by_name():
    plugin = Plugin(config(branches=two_branches(), fusion="attention"))
    with pytest.raises(ValueError) as refusal:
        plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {})
    assert "UNKNOWN_FUSION" in str(refusal.value)


# --- it trains one epoch on synthetic rows ------------------------------------------------------

def test_one_epoch_on_64_synthetic_rows_runs_and_predicts_the_declared_shape():
    """A plumbing test: it fits and predicts. It says nothing about the quality of what it fitted."""
    horizons = [1, 3]
    plugin = Plugin(config(branches=two_branches(), predicted_horizons=horizons, epochs=1))
    plugin.build_model((WINDOW, len(FEATURE_NAMES)), None, {})

    rng = np.random.default_rng(4242)
    x_train, x_val = synthetic(seed=1), synthetic(rows=16, seed=2)
    y_train = {name: rng.normal(size=(ROWS, 1)).astype("float32") for name in plugin.output_names}
    y_val = {name: rng.normal(size=(16, 1)).astype("float32") for name in plugin.output_names}

    history, train_preds, train_unc, val_preds, val_unc = plugin.train(
        x_train, y_train, epochs=1, batch_size=8, threshold_error=0.0,
        x_val=x_val, y_val=y_val, config={})

    assert len(history.history["loss"]) == 1
    assert len(train_preds) == len(horizons) and len(val_preds) == len(horizons)
    assert np.asarray(train_preds[0]).shape == (ROWS, 1)
    assert np.asarray(val_unc[0]).shape == (16, 1)

    predictions = plugin.model.predict(x_val, verbose=0)
    assert len(predictions) == len(horizons)
    assert all(np.asarray(head).shape == (16, 1) for head in predictions)


# --- the registry declares it -------------------------------------------------------------------

def test_setup_py_declares_the_entry_point_and_the_label_is_one_truthful_line():
    """Laya reads the docstring's first line as the option label, so it must be one sentence."""
    from pathlib import Path

    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    assert "fused_branches=predictor_plugins.fused_branches:Plugin" in setup_py.read_text(encoding="utf-8")
    label = (Plugin.__doc__ or "").strip().splitlines()[0]
    assert label and len(label) <= 60 and label.endswith(".")
