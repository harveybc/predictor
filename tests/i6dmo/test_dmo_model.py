"""DMO01 (six named independent heads, one shared backbone) and serialization."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import model as M  # noqa: E402
from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4)))


def _x(rows=5, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, 13, 19)).astype("float32")


def test_dmo01_six_named_outputs_in_physical_hour_order(bundle):
    model = bundle.forecast_model
    assert list(model.output_names) == ["h2", "h3", "h4", "h5", "h6", "h12"]
    assert bundle.config["horizons_hours"] == [2, 3, 4, 5, 6, 12]
    assert tuple(model.input_shape) == (None, 13, 19)
    out = M.predict_matrix(model, _x())
    assert out.shape == (5, 6) and out.dtype == np.float64


def test_dmo01_branch_fusion_encoding_core_identity(bundle):
    assert len(bundle.branch_models) == 19
    for branch in bundle.branch_models.values():
        assert tuple(branch.input_shape) == (None, 13, 1)
        convs = [l for l in branch.layers if isinstance(l, tf.keras.layers.Conv1D)]
        assert len(convs) == 1
        c = convs[0].get_config()
        assert (c["filters"], c["kernel_size"], c["strides"], c["padding"], c["activation"]) == \
            (4, (3,), (1,), "causal", "relu")
    assert tuple(bundle.fusion_model.output_shape) == (None, 13, 76)
    assert not bundle.fusion_model.layers[-1].weights
    # Positional encoding consumes the fusion output directly, with physical offsets.
    pe = bundle.forecast_model.get_layer("positional_encoding")
    assert isinstance(pe, M.PhysicalOffsetEncoding)
    assert pe.offsets_hours == tuple(range(-24, 1, 2))
    fusion = bundle.forecast_model.get_layer("sequence_fusion")
    assert pe.input is fusion.output
    assert bundle.forecast_model.get_layer("temporal_core").input is pe.output
    convs = [l for l in bundle.core_model.layers if isinstance(l, tf.keras.layers.Conv1D)]
    assert [(l.filters, l.kernel_size, l.strides, l.padding) for l in convs] == \
        [(8, (3,), (1,), "causal"), (4, (3,), (1,), "causal")]
    assert tuple(bundle.core_model.output_shape) == (None, 13, 4)


def test_dmo01_no_forbidden_layers(bundle):
    forbidden = (tf.keras.layers.LSTM, tf.keras.layers.Bidirectional, tf.keras.layers.GRU,
                 tf.keras.layers.MultiHeadAttention, tf.keras.layers.Dropout)
    seen = []

    def walk(model):
        for layer in model.layers:
            seen.append(layer)
            if isinstance(layer, tf.keras.Model):
                walk(layer)
    walk(bundle.forecast_model)
    assert not [l.name for l in seen if isinstance(l, forbidden)]
    assert not [l.name for l in seen if "bayes" in type(l).__name__.lower()
                or "transformer" in l.name.lower()]


def test_dmo01_heads_have_independent_weights_and_share_one_backbone(bundle):
    heads = bundle.head_models
    assert list(heads) == ["h2", "h3", "h4", "h5", "h6", "h12"]
    ids = [{id(w) for w in head.weights} for head in heads.values()]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not a & b
    for head in heads.values():
        dense = [l for l in head.layers if isinstance(l, tf.keras.layers.Dense)]
        assert [(l.units, l.get_config()["activation"]) for l in dense] == [(8, "relu"), (1, "linear")]
        assert any(isinstance(l, tf.keras.layers.Flatten) for l in head.layers)
        assert tuple(head.input_shape) == (None, 13, 4)
    # Perturbing one head's weights changes only that output.
    x = _x(seed=3)
    before = M.predict_matrix(bundle.forecast_model, x)
    saved = heads["h5"].get_weights()
    heads["h5"].set_weights([w + 0.5 for w in saved])
    after = M.predict_matrix(bundle.forecast_model, x)
    heads["h5"].set_weights(saved)
    changed = np.abs(after - before).max(axis=0) > 0
    assert changed.tolist() == [False, False, False, True, False, False]
    # The gradient of one head's output reaches the shared core but not other heads.
    with tf.GradientTape() as tape:
        outputs = bundle.forecast_model(tf.constant(x), training=False)
        target = tf.reduce_sum(outputs[0])
    grads = tape.gradient(target, [*heads["h3"].trainable_weights, *bundle.core_model.trainable_weights])
    n3 = len(heads["h3"].trainable_weights)
    assert all(g is None or float(tf.reduce_max(tf.abs(g))) == 0.0 for g in grads[:n3])
    assert any(g is not None and float(tf.reduce_max(tf.abs(g))) > 0 for g in grads[n3:])
    total = bundle.forecast_model.count_params()
    shared = sum(m.count_params() for m in bundle.branch_models.values()) + bundle.core_model.count_params()
    assert total == shared + sum(h.count_params() for h in heads.values())


def test_causality_future_input_does_not_move_earlier_core_steps(bundle):
    x = _x(rows=1, seed=7)
    base = np.asarray(bundle.core_model(bundle.fusion_model(x)))
    y = x.copy()
    y[:, 8, :] += 5.0
    moved = np.asarray(bundle.core_model(bundle.fusion_model(y)))
    assert np.allclose(base[:, :8], moved[:, :8])
    assert not np.allclose(base[:, 8:], moved[:, 8:])


def test_same_seed_same_initial_weights_and_config_is_strict():
    from predictor_plugins.modular_temporal import weights_hash
    cfg = M.model_config(list(MANIFEST_NO_STLFSI4))
    a, b = M.build_dmo(cfg), M.build_dmo(cfg)
    assert weights_hash(a.forecast_model) == weights_hash(b.forecast_model)
    bad = dict(cfg, surprise=1)
    with pytest.raises(ValueError, match="UNKNOWN_FIELD|schema"):
        M.build_dmo(bad)
    bad = dict(cfg, regime="R2")
    with pytest.raises(ValueError, match="R0"):
        M.build_dmo(bad)


def test_dmo10_save_load_identical_predictions_and_tamper_refusal(bundle, tmp_path):
    x = _x(rows=17, seed=11)
    doc = M.save_dmo(bundle, tmp_path / "m")
    loaded, document = M.load_dmo(tmp_path / "m")
    assert document["weights_sha256"] == doc["weights_sha256"]
    assert np.array_equal(M.predict_matrix(bundle.forecast_model, x),
                          M.predict_matrix(loaded.forecast_model, x))
    assert list(loaded.forecast_model.output_names) == ["h2", "h3", "h4", "h5", "h6", "h12"]
    archive = tmp_path / "m" / "forecast_model.keras"
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="archive"):
        M.load_dmo(tmp_path / "m")
